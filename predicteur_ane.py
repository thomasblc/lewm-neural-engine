"""Prédicteur de LeWM réécrit pour le Neural Engine (Core ML), construit à partir des poids du modèle chargé.

Même calcul que model.predict(emb, act_emb) : le prédicteur (6 blocs conditionnés façon AdaLN, attention causale,
plongement de position) puis pred_proj (MLP avec BatchNorm). Poids copiés, aucun réentraînement, dropout absent
(le modèle s'utilise en mode évaluation).

Règles d'Apple (« Deploying Transformers on the Apple Neural Engine », K-019) appliquées :
- tenseurs en 4 dimensions, canaux en deuxième dimension ;
- couches linéaires en convolutions 1 x 1 ;
- LayerNorm calculée sur l'axe des canaux ;
- ni reshape ni transpose dans le graphe : les changements de disposition se font hors du graphe (wrapper).

Deux dispositions :
- "bc1s" : (B, C, 1, T), celle d'Apple, attention découpée par tête avec einsum (code de ml-ane-transformers) ;
- "1ctb" : (1, C, T, B), le lot en dernier axe. Apple veut un dernier axe long : il est aligné sur 64 octets.
  Avec T = 3, le dernier axe de "bc1s" fait 6 octets en float16 ; avec le lot de 300, celui de "1ctb" en fait 600.
  L'attention se calcule par produits terme à terme et sommes par tête (convolutions 1 x 1 de poids 0 ou 1).

Usage :
    pred = PredicteurANE(model, "1ctb")
    sortie = pred(emb, act_emb)          # (B, T, D), comme model.predict(emb, act_emb)
    model.predict = pred.predict         # pour le brancher dans le planificateur
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

DISPOSITIONS = ("bc1s", "1ctb")
MASQUE = -1e4  # valeur ajoutée aux scores masqués ; exp(-1e4) vaut 0 en float32 comme en float16


def conv_de(lin, facteur=None):
    """Convolution 1 x 1 aux poids d'un nn.Linear (facteur optionnel par ligne de sortie)."""
    w = lin.weight.detach().clone()
    if facteur is not None:
        w = w * facteur[:, None]
    conv = nn.Conv2d(lin.in_features, lin.out_features, 1, bias=lin.bias is not None)
    with torch.no_grad():
        conv.weight.copy_(w[:, :, None, None])
        if lin.bias is not None:
            conv.bias.copy_(lin.bias.detach())
    return conv


class NormCanaux(nn.Module):
    """LayerNorm sur l'axe des canaux (dim 1), même formule que nn.LayerNorm."""

    def __init__(self, ln):
        super().__init__()
        assert isinstance(ln, nn.LayerNorm) and len(ln.normalized_shape) == 1, ln
        self.eps = ln.eps
        self.affine = ln.weight is not None
        if self.affine:
            self.register_buffer("w", ln.weight.detach().clone().view(1, -1, 1, 1))
            b = ln.bias if ln.bias is not None else torch.zeros_like(ln.weight)
            self.register_buffer("b", b.detach().clone().view(1, -1, 1, 1))

    def forward(self, x):
        m = x.mean(dim=1, keepdim=True)
        xc = x - m
        v = (xc * xc).mean(dim=1, keepdim=True)
        y = xc * torch.rsqrt(v + self.eps)
        return y * self.w + self.b if self.affine else y


class NormLot(nn.Module):
    """BatchNorm1d en mode évaluation, appliquée sur l'axe des canaux d'un tenseur 4D."""

    def __init__(self, bn):
        super().__init__()
        assert isinstance(bn, nn.BatchNorm1d) and bn.track_running_stats, bn
        self.eps = bn.eps
        for nom in ("running_mean", "running_var"):
            self.register_buffer(nom, getattr(bn, nom).detach().clone())
        self.register_buffer("w", bn.weight.detach().clone())
        self.register_buffer("b", bn.bias.detach().clone())

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.w, self.b, False, 0.0, self.eps)


def proj_canaux(p):
    """nn.Identity ou nn.Linear (projections d'entrée et de sortie du Transformer) en version 4D."""
    if isinstance(p, nn.Identity):
        return nn.Identity()
    assert isinstance(p, nn.Linear), p
    return conv_de(p)


class BlocANE(nn.Module):
    """ConditionalBlock : modulation AdaLN, attention causale, MLP, chacun avec sa porte."""

    def __init__(self, bloc, disposition, t_max):
        super().__init__()
        a, mlp = bloc.attn, bloc.mlp.net
        assert isinstance(a.to_out, nn.Sequential), "projection de sortie de l'attention attendue"
        assert isinstance(mlp[2], nn.GELU) and mlp[2].approximate == "none", mlp[2]
        assert isinstance(bloc.adaLN_modulation[0], nn.SiLU)
        self.disposition = disposition
        self.h = a.heads
        inner = a.to_qkv.out_features // 3
        self.d = inner // self.h
        self.mod = conv_de(bloc.adaLN_modulation[1])
        self.norm1, self.norm2 = NormCanaux(bloc.norm1), NormCanaux(bloc.norm2)
        self.norm_attn = NormCanaux(a.norm)
        # échelle 1/sqrt(d) de l'attention rangée dans les poids de q : 1/8 est une puissance de 2, produit exact
        facteur = torch.ones(3 * inner)
        facteur[:inner] = a.scale
        self.exacte = math.log2(a.scale).is_integer()
        self.qkv = conv_de(a.to_qkv, facteur)
        self.sortie = conv_de(a.to_out[0])
        self.norm_mlp = NormCanaux(mlp[0])
        self.fc1, self.fc2 = conv_de(mlp[1]), conv_de(mlp[4])
        # masque causal : la requête t ne voit pas les clés s > t
        s = torch.arange(t_max)
        masque = (s[:, None] > s[None, :]).float() * MASQUE  # [s, t]
        if disposition == "bc1s":
            self.register_buffer("masque", masque.view(1, t_max, 1, t_max))  # (1, clés, 1, requêtes)
        else:
            self.register_buffer("masque", masque.view(t_max, 1, 1, t_max, 1))  # par clé : (1, 1, requêtes, 1)
            un = torch.zeros(self.h, inner)
            for i in range(self.h):
                un[i, i * self.d:(i + 1) * self.d] = 1.0
            self.register_buffer("somme_tetes", un[:, :, None, None].clone())  # (H, HD, 1, 1)
            self.register_buffer("etale_tetes", un.t()[:, :, None, None].clone())  # (HD, H, 1, 1)

    def attention_bc1s(self, q, k, v):
        """q, k, v : (B, HD, 1, T). Code d'Apple (ml-ane-transformers), une tête à la fois."""
        T = q.shape[3]
        masque = self.masque[:, :T, :, :T]
        sorties = []
        for qi, ki, vi in zip(q.split(self.d, dim=1), k.transpose(1, 3).split(self.d, dim=3), v.split(self.d, dim=1)):
            w = torch.einsum("bchq,bkhc->bkhq", qi, ki) + masque  # (B, clés, 1, requêtes)
            w = w.softmax(dim=1)
            sorties.append(torch.einsum("bkhq,bchk->bchq", w, vi))  # (B, d, 1, requêtes)
        return torch.cat(sorties, dim=1)

    def attention_1ctb(self, q, k, v):
        """q, k, v : (1, HD, T, B). Toutes les têtes ensemble, une clé à la fois (T vaut au plus 3)."""
        T = q.shape[2]
        scores = []
        for s in range(T):
            sc = F.conv2d(q * k[:, :, s:s + 1, :], self.somme_tetes)  # (1, H, T, B), somme par tête
            if s > 0:
                sc = sc + self.masque[s][:, :, :T]
            scores.append(sc)
        m = scores[0]
        for sc in scores[1:]:
            m = torch.maximum(m, sc)
        e = [torch.exp(sc - m) for sc in scores]
        den = e[0]
        for es in e[1:]:
            den = den + es
        out = None
        for s in range(T):
            w = F.conv2d(e[s] / den, self.etale_tetes)  # (1, HD, T, B), poids de chaque tête sur ses d canaux
            terme = w * v[:, :, s:s + 1, :]
            out = terme if out is None else out + terme
        return out

    def forward(self, x, c):
        sh1, sc1, g1, sh2, sc2, g2 = self.mod(F.silu(c)).chunk(6, dim=1)
        h = self.norm_attn(self.norm1(x) * (1 + sc1) + sh1)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        att = self.attention_bc1s(q, k, v) if self.disposition == "bc1s" else self.attention_1ctb(q, k, v)
        x = x + g1 * self.sortie(att)
        h = self.norm_mlp(self.norm2(x) * (1 + sc2) + sh2)
        x = x + g2 * self.fc2(F.gelu(self.fc1(h)))
        return x


class CoeurANE(nn.Module):
    """predict en disposition du Neural Engine : entrées et sortie en (B, C, 1, T) ou (1, C, T, B).
    C'est ce module qu'on convertit en Core ML."""

    def __init__(self, model, disposition="1ctb"):
        super().__init__()
        assert disposition in DISPOSITIONS, disposition
        p, tr = model.predictor, model.predictor.transformer
        assert not model.training, "modèle attendu en mode évaluation"
        assert all(type(b).__name__ == "ConditionalBlock" for b in tr.layers)
        self.disposition = disposition
        self.t_max = p.pos_embedding.shape[1]
        pos = p.pos_embedding.detach().clone()[0].t()  # (D, T)
        self.register_buffer("pos", pos.view(1, -1, 1, self.t_max) if disposition == "bc1s"
                             else pos.view(1, -1, self.t_max, 1))
        self.entree, self.cond = proj_canaux(tr.input_proj), proj_canaux(tr.cond_proj)
        self.blocs = nn.ModuleList(BlocANE(b, disposition, self.t_max) for b in tr.layers)
        self.norm = NormCanaux(tr.norm)
        self.sortie = proj_canaux(tr.output_proj)
        pp = model.pred_proj
        if isinstance(pp, nn.Identity):
            self.proj = nn.Identity()
        else:
            net = pp.net
            assert isinstance(net[2], nn.GELU) and net[2].approximate == "none", net[2]
            norme = NormLot(net[1]) if isinstance(net[1], nn.BatchNorm1d) else (
                NormCanaux(net[1]) if isinstance(net[1], nn.LayerNorm) else None)
            assert norme is not None or isinstance(net[1], nn.Identity), net[1]
            self.proj = nn.Sequential(conv_de(net[0]), norme or nn.Identity(), nn.GELU(), conv_de(net[3]))
        self.exacte = all(b.exacte for b in self.blocs)
        self.eval()

    def forward(self, x, c):
        T = x.shape[3] if self.disposition == "bc1s" else x.shape[2]
        x = x + (self.pos[..., :T] if self.disposition == "bc1s" else self.pos[:, :, :T])
        x = self.entree(x)
        c = self.cond(c)
        for b in self.blocs:
            x = b(x, c)
        return self.proj(self.sortie(self.norm(x)))


def vers_disposition(x, disposition):
    """(B, T, D) vers la disposition du Neural Engine (marche pour torch et numpy)."""
    if disposition == "bc1s":
        return x.transpose(0, 2, 1)[:, :, None, :] if not torch.is_tensor(x) else x.permute(0, 2, 1).unsqueeze(2)
    return x.transpose(2, 1, 0)[None] if not torch.is_tensor(x) else x.permute(2, 1, 0).unsqueeze(0)


def depuis_disposition(y, disposition):
    """Disposition du Neural Engine vers (B, T, D)."""
    if disposition == "bc1s":
        return y[:, :, 0, :].transpose(0, 2, 1) if not torch.is_tensor(y) else y[:, :, 0, :].permute(0, 2, 1)
    return y[0].transpose(2, 1, 0) if not torch.is_tensor(y) else y[0].permute(2, 1, 0)


class PredicteurANE(nn.Module):
    """Même interface que model.predict(emb, act_emb) -> (B, T, D).

    t_fixe : si donné (3 pour un modèle Core ML à formes fixes), les séquences plus courtes sont complétées par des
    zéros à la fin, puis la sortie est recoupée. L'attention étant causale et le reste calculé jeton par jeton, les
    T premiers jetons ne dépendent pas des jetons ajoutés : le résultat est le même."""

    def __init__(self, model, disposition="1ctb", t_fixe=None):
        super().__init__()
        self.coeur = CoeurANE(model, disposition)
        self.disposition = disposition
        self.t_fixe = t_fixe

    def forward(self, emb, act_emb):
        B, T, _ = emb.shape
        if self.t_fixe is not None and T < self.t_fixe:
            emb = F.pad(emb, (0, 0, 0, self.t_fixe - T))
            act_emb = F.pad(act_emb, (0, 0, 0, self.t_fixe - T))
        y = self.coeur(vers_disposition(emb, self.disposition), vers_disposition(act_emb, self.disposition))
        return depuis_disposition(y, self.disposition)[:, :T].contiguous()

    def predict(self, emb, act_emb):
        return self(emb, act_emb)
