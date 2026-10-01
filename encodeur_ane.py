"""Chemin encode de LeWM réécrit pour le Neural Engine (Core ML), construit à partir des poids du modèle chargé.

Même calcul que model.encode(info) pour les images : ViT-tiny de HF transformers (patchs de 14, 12 couches, 3 têtes
de 64, 192 canaux, 257 jetons à 224 x 224), jeton CLS de la dernière couche après la LayerNorm finale, puis le
projecteur (MLP avec BatchNorm). L'encodeur d'actions (Embedder, appelé par rollout sur les 300 candidats à chaque
itération de la CEM) a aussi sa version. Poids copiés, aucun réentraînement, modèle en mode évaluation (dropout nul).

Positions : model.encode appelle le ViT avec interpolate_pos_encoding=True. À 224 x 224 la grille des positions
apprises est déjà de 16 x 16 : en mode normal, HF rend les positions telles quelles ; sous torch.jit.trace, il passe
quand même par une interpolation bicubique de 16 x 16 vers 16 x 16 (identité), que coremltools 9.0 ne sait pas
convertir (upsample_bicubic2d). La version réécrite range les positions comme constantes (calculées une fois par
la méthode de HF, hors trace).

Règles d'Apple (« Deploying Transformers on the Apple Neural Engine », K-019) appliquées :
- disposition (B, C, 1, S), canaux en deuxième dimension, S = 257 jetons sur le dernier axe (514 octets en float16,
  complétés à 576 par l'alignement sur 64 octets : 12 % de place perdue au plus, contre un facteur 32 pour un axe
  de 1 ou 2 éléments) ;
- couches linéaires en convolutions 1 x 1 ;
- LayerNorm calculée sur l'axe des canaux (coremltools reconnaît le motif et le fusionne en layer_norm) ;
- attention découpée par tête avec einsum (code de ml-ane-transformers) ;
- un seul reshape, celui des patchs (B, C, 16, 16) vers (B, C, 1, 256), et une concaténation pour le CLS.

Dernière couche : seul le jeton CLS sert à la sortie. Les clés et les valeurs y sont calculées pour les 257 jetons ;
la requête, la sortie de l'attention, le MLP et la LayerNorm finale pour le seul CLS (option cls_seul, vraie par
défaut). Même résultat (chaque jeton ne dépend que de sa requête et de toutes les clés et valeurs) ; 1,59 milliard
de multiplications-additions par image au lieu de 1,70 (6,5 % en moins ; q reste calculé pour tous les jetons par la
convolution commune q, k, v).

Encodeur d'actions : disposition (1, C, T, B) des actions, le lot en dernier axe (comme le prédicteur "1ctb" de
predicteur_ane.py) ; la convolution 1D de noyau 1 et les deux couches linéaires deviennent des convolutions 1 x 1.

Usage :
    enc = EncodeurANE(model)
    info = enc.encode(info)                     # comme model.encode(info)
    model.encode = enc.encode                   # pour le brancher dans le planificateur
    act = ActionsANE(model)
    model.action_encoder.forward = act.forward  # même interface que model.action_encoder(x)
    del model.encode, model.action_encoder.forward   # retour aux méthodes d'origine

Versions Core ML (formes fixes) : encode_coreml(model, ml) et actions_coreml(ml) rendent les fonctions à brancher
de la même façon (images une à une en (1, 3, 224, 224) ; actions complétées par des zéros jusqu'à (300, 5)).
"""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from predicteur_ane import NormCanaux, NormLot, conv_de, depuis_disposition, vers_disposition

TAILLE = 224                 # côté des images données par le planificateur
LOT_ACTIONS, T_ACTIONS = 300, 5   # rollout : 300 candidats, horizon de 5 blocs d'actions


def conv_empilee(lins, facteurs):
    """Une convolution 1 x 1 pour plusieurs nn.Linear de même entrée (sorties empilées), chaque bloc multiplié par
    son facteur, poids et biais compris (y = f * (W x + b))."""
    w = torch.cat([l.weight.detach() * f for l, f in zip(lins, facteurs)])
    avec_biais = any(l.bias is not None for l in lins)
    conv = nn.Conv2d(lins[0].in_features, w.shape[0], 1, bias=avec_biais)
    with torch.no_grad():
        conv.weight.copy_(w[:, :, None, None])
        if avec_biais:
            conv.bias.copy_(torch.cat([(l.bias.detach() if l.bias is not None else torch.zeros(l.out_features)) * f
                                       for l, f in zip(lins, facteurs)]))
    return conv


def gelu_exacte(fn):
    """Vrai si fn est la GELU exacte (erf) : nn.GELU(approximate="none") ou GELUActivation de HF ("gelu")."""
    if isinstance(fn, nn.GELU):
        return fn.approximate == "none"
    return getattr(fn, "act", None) is F.gelu


def mlp_canaux(mlp):
    """MLP de LeWM (Linear, BatchNorm1d ou LayerNorm, GELU, Linear) en version 4D, normalisation sur les canaux."""
    if isinstance(mlp, nn.Identity):
        return nn.Identity()
    net = mlp.net
    assert gelu_exacte(net[2]), net[2]
    norme = NormLot(net[1]) if isinstance(net[1], nn.BatchNorm1d) else (
        NormCanaux(net[1]) if isinstance(net[1], nn.LayerNorm) else None)
    assert norme is not None or isinstance(net[1], nn.Identity), net[1]
    return nn.Sequential(conv_de(net[0]), norme or nn.Identity(), nn.GELU(), conv_de(net[3]))


class EncodeurOrigine(nn.Module):
    """Graphe d'origine de model.encode pour les images : (B, 3, 224, 224) vers (B, D).

    interpolation=True est l'appel du planificateur ; sous torch.jit.trace il garde l'interpolation bicubique des
    positions (non convertible). interpolation=False donne le même calcul à 224 x 224 sans cette opération."""

    def __init__(self, model, interpolation=True):
        super().__init__()
        self.vit, self.proj = model.encoder, model.projector
        self.interpolation = interpolation

    def forward(self, px):
        sortie = self.vit(px, interpolate_pos_encoding=self.interpolation)
        return self.proj(sortie.last_hidden_state[:, 0])


class BlocViTANE(nn.Module):
    """ViTLayer de HF en (B, C, 1, S) : LayerNorm, attention à têtes séparées, MLP, deux résidus."""

    def __init__(self, couche):
        super().__init__()
        at, so = couche.attention.attention, couche.attention.output
        assert not at.is_causal, "attention attendue sans masque"
        assert gelu_exacte(couche.intermediate.intermediate_act_fn), couche.intermediate.intermediate_act_fn
        self.h, self.d = at.num_attention_heads, at.attention_head_size
        self.norm1, self.norm2 = NormCanaux(couche.layernorm_before), NormCanaux(couche.layernorm_after)
        # échelle 1/sqrt(64) = 1/8 rangée dans les poids et le biais de q : puissance de 2, produit exact
        self.exacte = math.log2(at.scaling).is_integer()
        self.qkv = conv_empilee([at.query, at.key, at.value], [at.scaling, 1.0, 1.0])
        self.sortie = conv_de(so.dense)
        self.fc1, self.fc2 = conv_de(couche.intermediate.dense), conv_de(couche.output.dense)

    def attention(self, q, k, v):
        """q : (B, HD, 1, Sq), k et v : (B, HD, 1, S). Code d'Apple (ml-ane-transformers), une tête à la fois."""
        sorties = []
        for qi, ki, vi in zip(q.split(self.d, dim=1), k.transpose(1, 3).split(self.d, dim=3), v.split(self.d, dim=1)):
            w = torch.einsum("bchq,bkhc->bkhq", qi, ki).softmax(dim=1)  # (B, clés, 1, requêtes)
            sorties.append(torch.einsum("bkhq,bchk->bchq", w, vi))      # (B, d, 1, requêtes)
        return torch.cat(sorties, dim=1)

    def forward(self, x, cls_seul=False):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=1)
        if cls_seul:  # dernière couche : la requête et le résidu du seul jeton CLS
            q, x = q[..., :1], x[..., :1]
        x = x + self.sortie(self.attention(q, k, v))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class CoeurEncodeurANE(nn.Module):
    """Pixels (B, 3, 224, 224), normalisés comme les donne le planificateur, vers emb en (B, D, 1, 1).
    C'est ce module qu'on convertit en Core ML."""

    def __init__(self, model, cls_seul=True, taille=TAILLE):
        super().__init__()
        vit = model.encoder
        assert type(vit).__name__ == "ViTModel", type(vit)
        assert not model.training and not vit.training, "modèle attendu en mode évaluation"
        e = vit.embeddings
        D = vit.config.hidden_size
        self.patch = copy.deepcopy(e.patch_embeddings.projection).eval()  # Conv2d(3, D, 14, stride 14)
        self.grille = taille // vit.config.patch_size
        n = self.grille ** 2
        with torch.no_grad():  # positions comme les rend HF pour cette taille, hors trace (sans interpolation à 224)
            pos = e.interpolate_pos_encoding(torch.zeros(1, n + 1, D), taille, taille).detach().clone()[0].t()
        self.register_buffer("cls", (e.cls_token.detach()[0, 0] + pos[:, 0]).view(1, D, 1, 1).clone())
        self.register_buffer("pos", pos[:, 1:].reshape(1, D, 1, n).clone())
        self.blocs = nn.ModuleList(BlocViTANE(c) for c in vit.encoder.layer)
        self.norm = NormCanaux(vit.layernorm)
        self.proj = mlp_canaux(model.projector)
        self.cls_seul = cls_seul
        self.exacte = all(b.exacte for b in self.blocs)
        self.eval()

    def forward(self, px):
        B = px.shape[0]
        x = self.patch(px)                                       # (B, D, 16, 16)
        x = x.reshape(B, x.shape[1], 1, -1) + self.pos           # (B, D, 1, 256), jeton h * 16 + w comme HF
        x = torch.cat([self.cls.expand(B, -1, -1, -1), x], dim=3)  # (B, D, 1, 257), CLS en tête
        n = len(self.blocs)
        for i, b in enumerate(self.blocs):
            x = b(x, cls_seul=self.cls_seul and i == n - 1)
        return self.proj(self.norm(x[..., :1]))                  # (B, D, 1, 1)


class EncodeurANE(nn.Module):
    """Même interface que model.encode(info) : info["pixels"] (b, t, 3, 224, 224) vers info["emb"] (b, t, D), et
    info["act_emb"] par l'encodeur d'actions du modèle (tel qu'il est branché au moment de l'appel) si "action"."""

    def __init__(self, model, cls_seul=True):
        super().__init__()
        self.coeur = CoeurEncodeurANE(model, cls_seul)
        self._modele = (model,)  # référence sans enregistrer le modèle comme sous-module

    def forward(self, px):
        """(N, 3, 224, 224) vers (N, D)."""
        return self.coeur(px.to(torch.float32)).flatten(1)

    def encode(self, info):
        pixels = info["pixels"].to(torch.float32)  # LeWM.encode : type des poids de l'encodeur (float32)
        b = pixels.size(0)
        emb = self(pixels.flatten(0, 1))             # (b t) ... comme rearrange
        info["emb"] = emb.view(b, -1, emb.shape[-1])
        if "action" in info:
            info["act_emb"] = self._modele[0].action_encoder(info["action"])
        return info


class CoeurActionsANE(nn.Module):
    """Embedder en disposition (1, C, T, B) : actions (1, 10, T, B) vers act_emb (1, D, T, B). Module converti."""

    def __init__(self, model):
        super().__init__()
        e = model.action_encoder
        pe = e.patch_embed
        assert isinstance(pe, nn.Conv1d) and pe.kernel_size == (1,) and pe.stride == (1,) and pe.padding == (0,) \
            and pe.dilation == (1,) and pe.groups == 1, pe
        self.lisse = nn.Conv2d(pe.in_channels, pe.out_channels, 1, bias=pe.bias is not None)
        with torch.no_grad():
            self.lisse.weight.copy_(pe.weight.detach()[:, :, :, None])
            if pe.bias is not None:
                self.lisse.bias.copy_(pe.bias.detach())
        assert isinstance(e.embed[1], nn.SiLU), e.embed[1]
        self.fc1, self.fc2 = conv_de(e.embed[0]), conv_de(e.embed[2])
        self.eval()

    def forward(self, a):
        return self.fc2(F.silu(self.fc1(self.lisse(a))))


def completer(x, lot, t):
    """(B, T, A) complété par des zéros jusqu'à (lot, t). Chaque jeton se calcule seul : sans effet sur les autres."""
    B, T, _ = x.shape
    assert B <= lot and T <= t, (B, T, lot, t)
    return F.pad(x, (0, 0, 0, t - T, 0, lot - B)) if (B, T) != (lot, t) else x


class ActionsANE(nn.Module):
    """Même interface que model.action_encoder(x) : (B, T, 10) vers (B, T, D).

    forme_fixe : si donnée ((300, 5) pour le modèle Core ML), x est complété par des zéros puis la sortie recoupée."""

    def __init__(self, model, forme_fixe=None):
        super().__init__()
        self.coeur = CoeurActionsANE(model)
        self.forme_fixe = forme_fixe

    def forward(self, x):
        x = x.float()  # comme Embedder.forward
        B, T, _ = x.shape
        if self.forme_fixe is not None:
            x = completer(x, *self.forme_fixe)
        y = depuis_disposition(self.coeur(vers_disposition(x, "1ctb")), "1ctb")
        return y[:B, :T].contiguous()


def _sortie_unique(ml, entrees):
    sortie = ml.predict(entrees)
    assert len(sortie) == 1, list(sortie)
    return np.asarray(next(iter(sortie.values())), dtype=np.float32)


def encode_coreml(model, ml, nom="pixels"):
    """Fonction encode(info) qui appelle le modèle Core ML du coeur (entrée (1, 3, 224, 224), sortie (1, D, 1, 1) ou
    (1, D)) image par image ; même interface que model.encode. act_emb par l'encodeur d'actions du modèle."""

    def encode(info):
        pixels = info["pixels"]
        b = pixels.size(0)
        px = pixels.flatten(0, 1).detach().to("cpu", torch.float32).numpy()
        emb = np.concatenate([_sortie_unique(ml, {nom: np.ascontiguousarray(px[i:i + 1])}).reshape(1, -1)
                              for i in range(px.shape[0])])
        emb = torch.from_numpy(emb).to(pixels.device)
        info["emb"] = emb.view(b, -1, emb.shape[-1])
        if "action" in info:
            info["act_emb"] = model.action_encoder(info["action"])
        return info

    return encode


def actions_coreml(ml, lot=LOT_ACTIONS, t=T_ACTIONS, nom="act"):
    """Fonction x (B, T, 10) -> (B, T, D) qui appelle le modèle Core ML de CoeurActionsANE (entrée (1, 10, t, lot)),
    x complété par des zéros jusqu'à (lot, t) puis sortie recoupée ; même interface que model.action_encoder."""

    def action_encoder(x):
        B, T, _ = x.shape
        a = completer(x.detach().to("cpu", torch.float32), lot, t).numpy()
        y = depuis_disposition(_sortie_unique(ml, {nom: np.ascontiguousarray(vers_disposition(a, "1ctb"))}), "1ctb")
        return torch.from_numpy(np.ascontiguousarray(y[:B, :T])).to(x.device)

    return action_encoder
