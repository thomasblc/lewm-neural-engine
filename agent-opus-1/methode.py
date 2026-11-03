"""Compression ternaire en forme close, couche par couche (essai-opus-1, EXP-020).

Pour chaque couche k, dans l'ordre :
 1. X~ = [entrées vues par la couche dans le réseau DÉJÀ compressé, 1] ; cible Y = pré-activation du float32
    (couche de sortie : soit la sortie du float32, soit le vrai déplacement (s2 - s) / VMAX du simulateur).
 2. W* = moindres carrés ridge de X~ vers Y (ridge faible vers les poids float32) : corrige l'erreur amont.
 3. Poids gardés : critère OBS/SpQR sur W* (budget de la couche fixé à la main).
 4. GPTQ de W* avec H = X~^T X~ (colonne de biais en dernier, toujours libre : elle absorbe les compensations),
    puis réajustement exact par ligne (échelle, biais, valeurs gardées) par moindres carrés, codes fixés.
    Deux passes fixes : la seconde refait GPTQ avec les échelles réajustées.
Aucune descente de gradient, aucune boucle de recherche : tout est algèbre linéaire en float64.
"""
import sys, math, torch
CODE = "/Users/thomasblanc/Documents/PRO/these-ia-thomas/papier-corl-2026/code-exp020"
sys.path.insert(0, CODE)
import banc, gptq

C96 = banc.RESULTATS / "c96"
SCORES = []          # scores des poids gardés de la dernière compression, couche par couche
torch.set_default_dtype(torch.float32)


def donnees(n_trans=200_000, n_traj=1000, pas_traj=40, graine=4242, n_pousse=0):
    """Transitions du simulateur : tirage de banc.transitions (graine à part) + trajectoires depuis des départs
    d'épisodes (graine à part, actions au hasard lissées), pour couvrir les états vus en planification."""
    s1, a1, s21 = banc.transitions(n_trans, graine)
    g = torch.Generator().manual_seed(graine + 1)
    s, but = banc.episodes(n_traj, graine + 2)
    S, A, S2 = [], [], []
    a = 2 * torch.rand(n_traj, 2, generator=g) - 1
    for t in range(pas_traj):
        a = (0.7 * a + 0.3 * (2 * torch.rand(n_traj, 2, generator=g) - 1) * 1.5).clamp(-1, 1)
        # moitié des épisodes : l'agent est attiré vers le bloc pour provoquer des poussées
        vers = (s[:, 2:] - s[:, :2]); vers = vers / vers.norm(dim=1, keepdim=True).clamp_min(1e-6)
        a_ = torch.where((torch.arange(n_traj) % 2 == 0)[:, None], (0.5 * a + vers).clamp(-1, 1), a)
        s2 = banc.pas(s, a_)
        S.append(s); A.append(a_); S2.append(s2)
        s = s2
    if n_pousse:
        # trajectoires du pousseur scripté (graine à part de celle de validation, 31337)
        P, PA = trajectoires_poussee(n=n_pousse, graine=graine + 3)
        S.append(P[:, :-1].reshape(-1, 4)); A.append(PA.reshape(-1, 2)); S2.append(P[:, 1:].reshape(-1, 4))
    return torch.cat([s1, *S]), torch.cat([a1, *A]), torch.cat([s21, *S2])


def gptq_lignes(W, H, s, garder, qmax=1, amort=0.01, ordre=None):
    """GPTQ sans blocs (d <= 97). W (o, d) float64, H (d, d), s (o,) échelles, garder (o, d) booléen
    (poids laissés libres : erreur nulle, ils reçoivent les compensations). ordre : permutation des colonnes."""
    o, d = W.shape
    if ordre is None:
        ordre = torch.arange(d)
    W = W[:, ordre].clone(); H = H[ordre][:, ordre].clone(); G = garder[:, ordre]
    mort = torch.diag(H) == 0
    H[mort, mort] = 1; W[:, mort] = 0
    H = H + amort * torch.mean(torch.diag(H)) * torch.eye(d, dtype=H.dtype)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    S = s.reshape(-1)
    for i in range(d):
        col = W[:, i]
        q = (col / S).round().clamp(-qmax, qmax) * S
        q = torch.where(G[:, i], col, q)
        Q[:, i] = q
        err = (col - q) / U[i, i]
        W[:, i:] -= err[:, None] * U[i, i:][None, :]
    inv = torch.argsort(ordre)
    return Q[:, inv]


def reajuster(codes, garder, H, R, ridge=1e-9):
    """Par ligne j : minimise E||X~ w_j - y_j||^2 sur (échelle s_j, valeurs libres), codes fixés.
    codes (o, d) entiers (0 aux positions libres), garder (o, d) booléen (biais compris), H = E[x~ x~^T],
    R = E[x~ y^T] (d, o). Renvoie s (o,), W (o, d) (poids complets, biais en dernière colonne)."""
    o, d = codes.shape
    s = torch.zeros(o, dtype=torch.float64); W = torch.zeros(o, d, dtype=torch.float64)
    for j in range(o):
        K = garder[j].nonzero().squeeze(1)
        u = codes[j].double().clone(); u[K] = 0
        cols = [u] + [torch.nn.functional.one_hot(k, d).double() for k in K]
        A = torch.stack(cols, 1)                                   # (d, 1 + |K|)
        M = A.T @ H @ A + ridge * torch.eye(A.shape[1], dtype=torch.float64) * torch.diag(A.T @ H @ A).mean()
        th = torch.linalg.solve(M, A.T @ R[:, j])
        if u.abs().sum() == 0:
            th[0] = 0
        s[j] = th[0]
        W[j] = A @ th
    return s, W


def couche(Xq, Y, Wfp, bfp, n_garde, qmax=1, ridge=1e-4, passes=2, act_order=True, poids=None, imp=None):
    """Compresse une couche. Xq (n, d) entrées compressées, Y (n, o) cibles. Renvoie le dict compact de la couche."""
    n, d = Xq.shape
    X = torch.cat([Xq, torch.ones(n, 1, dtype=torch.float64)], 1)
    if poids is not None:
        Xw = X * poids[:, None]
        H = Xw.T @ X / poids.sum(); R = Xw.T @ Y / poids.sum()
    else:
        H = X.T @ X / n; R = X.T @ Y / n
    W0 = torch.cat([Wfp, bfp[:, None]], 1)                         # (o, d+1)
    lam = ridge * torch.diag(H).mean()
    Wst = torch.linalg.solve(H + lam * torch.eye(d + 1, dtype=torch.float64), R + lam * W0.T).T   # (o, d+1)
    o = Wst.shape[0]
    # échelle initiale : facteur réduit d'EXP-011 (celui du banc), calculé sur W* et H sans la colonne de biais
    f = gptq.facteur_reduit(Wst[:, :d], H[:d, :d], 2)
    s0 = Wst[:, :d].abs().amax(1).clamp_min(1e-12) / qmax * f
    # poids gardés : critère OBS/SpQR sur W*, colonne de biais exclue
    Hd = H.clone(); Hd += 0.01 * torch.diag(Hd).mean() * torch.eye(d + 1, dtype=torch.float64)
    hinv = torch.diag(torch.cholesky_inverse(torch.linalg.cholesky(Hd)))[:d]
    q = (Wst[:, :d] / s0[:, None]).round().clamp(-qmax, qmax) * s0[:, None]
    score = (Wst[:, :d] - q) ** 2 / hinv[None, :]
    if imp is not None:
        score = score * imp[:, None]
    SCORES.append(score)
    G = torch.zeros(o, d + 1, dtype=torch.bool); G[:, d] = True
    if n_garde > 0:
        idx = torch.argsort(score.flatten(), descending=True, stable=True)[:n_garde]
        Gf = G[:, :d].flatten().clone(); Gf[idx] = True; G[:, :d] = Gf.reshape(o, d)
    ordre = torch.cat([torch.argsort(torch.diag(H)[:d], descending=True), torch.tensor([d])]) if act_order \
        else torch.arange(d + 1)
    s = s0
    for p in range(passes):
        Q = gptq_lignes(Wst, H, s, G, qmax, ordre=ordre)
        codes = torch.where(G, torch.zeros_like(Q), (Q / s[:, None]).round().clamp(-qmax, qmax))
        s_new, Wf = reajuster(codes, G, H, R)
        neg = s_new < 0
        codes[neg] = -codes[neg]; s_new = s_new.abs()
        s = torch.where(s_new > 0, s_new, s0)
    idx = G[:, :d].flatten().nonzero().squeeze(1)
    return {"codes": codes[:, :d].to(torch.int8), "echelle": s.to(torch.float16), "garde_idx": idx,
            "garde_val": Wf[:, :d].flatten()[idx].to(torch.float16), "biais": Wf[:, d].to(torch.float16)}


def couche_relu(Xq, Y, Wfp, bfp, n_garde, prelim, qmax=1, ridge=1e-4, passes=2, poids=None, imp=None):
    """Variante « ReLU » d'une couche cachée : la ligne j pèse chaque échantillon par 1 si la pré-activation cible
    est positive, sinon par exp(-(y/sigma_j)^2 / 2), sigma_j l'écart quadratique de la ligne après une première
    compression non pondérée (`prelim`) : une pré-activation très négative reste coupée par la ReLU, son erreur ne
    compte pas. Même suite d'étapes que `couche`, mais H et R propres à chaque ligne."""
    n, d = Xq.shape
    X = torch.cat([Xq, torch.ones(n, 1, dtype=torch.float64)], 1)
    Zp = avant_couche(prelim, Xq, Wfp.shape)
    sig = (Zp - Y).pow(2).mean(0).sqrt().clamp_min(1e-6)
    Pw = torch.where(Y > 0, torch.ones_like(Y), torch.exp(-0.5 * (Y / sig) ** 2))
    if poids is not None:
        Pw = Pw * poids[:, None]
    o = Y.shape[1]
    W0 = torch.cat([Wfp, bfp[:, None]], 1)
    Hs, Rs, Wst = [], [], torch.zeros(o, d + 1, dtype=torch.float64)
    for j in range(o):
        w = Pw[:, j]
        if not w.sum() > 0:          # ligne morte (cible toujours très négative) : poids uniformes
            w = torch.ones_like(w)
        Xw = X * w[:, None]
        H = Xw.T @ X / w.sum(); R = Xw.T @ Y[:, j] / w.sum()
        lam = ridge * torch.diag(H).mean()
        Wst[j] = torch.linalg.solve(H + lam * torch.eye(d + 1, dtype=torch.float64), R + lam * W0[j])
        Hs.append(H); Rs.append(R)
    # poids gardés : même critère OBS, avec la H de chaque ligne
    s0 = prelim["echelle"].double()
    score = torch.zeros(o, d, dtype=torch.float64)
    for j in range(o):
        Hd = Hs[j] + 0.01 * torch.diag(Hs[j]).mean() * torch.eye(d + 1, dtype=torch.float64)
        hinv = torch.diag(torch.cholesky_inverse(torch.linalg.cholesky(Hd)))[:d]
        q = (Wst[j, :d] / s0[j]).round().clamp(-qmax, qmax) * s0[j]
        score[j] = (Wst[j, :d] - q) ** 2 / hinv
    if imp is not None:
        score = score * imp[:, None]
    SCORES.append(score)
    G = torch.zeros(o, d + 1, dtype=torch.bool); G[:, d] = True
    if n_garde > 0:
        idx = torch.argsort(score.flatten(), descending=True, stable=True)[:n_garde]
        Gf = G[:, :d].flatten().clone(); Gf[idx] = True; G[:, :d] = Gf.reshape(o, d)
    s = s0.clone()
    codes = torch.zeros(o, d + 1, dtype=torch.float64); Wf = torch.zeros(o, d + 1, dtype=torch.float64)
    for j in range(o):
        ordre = torch.cat([torch.argsort(torch.diag(Hs[j])[:d], descending=True), torch.tensor([d])])
        sj = s[j:j + 1]
        for p in range(passes):
            Q = gptq_lignes(Wst[j:j + 1], Hs[j], sj, G[j:j + 1], qmax, ordre=ordre)
            cj = torch.where(G[j:j + 1], torch.zeros_like(Q), (Q / sj[:, None]).round().clamp(-qmax, qmax))
            s_new, Wj = reajuster(cj, G[j:j + 1], Hs[j], Rs[j][:, None])
            if s_new[0] < 0:
                cj = -cj; s_new = -s_new
            sj = torch.where(s_new > 0, s_new, sj)
        codes[j] = cj[0]; Wf[j] = Wj[0]; s[j] = sj[0]
    idx = G[:, :d].flatten().nonzero().squeeze(1)
    return {"codes": codes[:, :d].to(torch.int8), "echelle": s.to(torch.float16), "garde_idx": idx,
            "garde_val": Wf[:, :d].flatten()[idx].to(torch.float16), "biais": Wf[:, d].to(torch.float16)}


def ls_ridge(Xq, Y, W0, ridge=1e-4):
    n, d = Xq.shape
    X = torch.cat([Xq, torch.ones(n, 1, dtype=torch.float64)], 1)
    H = X.T @ X / n; R = X.T @ Y / n
    lam = ridge * torch.diag(H).mean()
    return torch.linalg.solve(H + lam * torch.eye(d + 1, dtype=torch.float64), R + lam * W0.T).T


def echelles_colonnes(Wst):
    """Équilibrage : W ~ diag(s) C diag(t). t_i = moyenne quadratique de la colonne i après normalisation des lignes.
    La ReLU étant positivement homogène, t est absorbé sans coût dans l'échelle, le biais et les poids gardés de la
    ligne i de la couche précédente."""
    W = Wst[:, :-1]
    rms_l = W.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
    t = (W / rms_l).pow(2).mean(0).sqrt()
    t = t / t.log().mean().exp()
    return t.clamp(1e-2, 1e2)


def avant_couche(c, X, forme):
    Wq = banc.depuis_couche(c, forme).double(); bq = c["biais"].double()
    return X @ Wq.T + bq


def importance(f32, s, a, c_sortie=(0.25, 0.25, 1.0, 1.0), n=20_000):
    """Sensibilité de la sortie à la pré-activation de chaque ligne, pour le float32 (jacobienne calculée à la main,
    produit de matrices et de masques ReLU, sans autograd) : imp_k[j] = E[sum_o c_o J_k[o, j]^2]. c_o : poids du
    coût de planification (1 pour le bloc, 0,5 pour l'agent, au carré)."""
    x = banc.Modele.entrees(s[:n], a[:n]).double()
    Ws = [l.weight.detach().double() for l in f32.couches]; bs = [l.bias.detach().double() for l in f32.couches]
    masques = []
    for k in range(3):
        z = x @ Ws[k].T + bs[k]; masques.append((z > 0).double()); x = torch.relu(z)
    c = torch.tensor(c_sortie, dtype=torch.float64)
    imps = [None] * 4
    J = Ws[3].unsqueeze(0).expand(x.shape[0], -1, -1)           # d sortie / d h2, (n, 4, 96)
    for k in (2, 1, 0):
        Jz = J * masques[k][:, None, :]                           # d sortie / d z_k
        imps[k] = (Jz ** 2 * c[None, :, None]).sum(1).mean(0)
        J = Jz @ Ws[k]                                            # d sortie / d h_{k-1}
    imps[3] = c.clone()
    return imps


def compresser(gardes=(6, 92, 92, 4), cible_sortie="vraie", ridge=1e-4, passes=2, act_order=True,
               equilibrer=False, relu=False, contact=0.0, n_trans=200_000, n_traj=1000, graine=4242, imp=False,
               n_pousse=0):
    f32, _ = banc.charger(C96 / "fp32.pt")
    s, a, s2 = donnees(n_trans, n_traj, graine=graine, n_pousse=n_pousse)
    SCORES.clear()
    imps = importance(f32, s, a) if imp else [None] * 4
    Xf = banc.Modele.entrees(s, a).double()
    Xin = Xf.clone()                       # entrées de la couche précédente compressée (avant elle)
    # couches cachées : poids 1 + contact pour les transitions où le bloc bouge (la physique à apprendre)
    bouge = ((s2[:, 2:] - s[:, 2:]).norm(dim=1) > 1e-6).double()
    pc = (1 + contact * bouge) if contact > 0 else None
    Xq = Xf.clone()
    out = {"bits": 2, "couches": []}
    L = len(f32.couches)
    for k, lin in enumerate(f32.couches):
        Wfp = lin.weight.detach().double(); bfp = lin.bias.detach().double()
        Yf = Xf @ Wfp.T + bfp
        Y = ((s2 - s) / banc.VMAX).double() if (k == L - 1 and cible_sortie == "vraie") else Yf
        if equilibrer and k > 0:
            Wst = ls_ridge(Xq, Y, torch.cat([Wfp, bfp[:, None]], 1), ridge)
            t = echelles_colonnes(Wst)
            p = out["couches"][k - 1]
            o_, i_ = p["codes"].shape
            rang = p["garde_idx"] // i_
            p["echelle"] = (p["echelle"].double() * t).to(torch.float16)
            p["biais"] = (p["biais"].double() * t).to(torch.float16)
            p["garde_val"] = (p["garde_val"].double() * t[rang]).to(torch.float16)
            Xq = torch.relu(avant_couche(p, Xin, (o_, i_)))
            Wfp = Wfp / t[None, :]
        pk = pc if k < L - 1 else None
        c = couche(Xq, Y, Wfp, bfp, gardes[k], ridge=ridge, passes=passes, act_order=act_order, poids=pk, imp=imps[k])
        if relu and k < L - 1:
            c = couche_relu(Xq, Y, Wfp, bfp, gardes[k], c, ridge=ridge, passes=passes, poids=pk, imp=imps[k])
        out["couches"].append(c)
        Xin = Xq
        Xq = avant_couche(c, Xq, Wfp.shape)
        Xf = Yf
        if k < L - 1:
            Xq = torch.relu(Xq); Xf = torch.relu(Xf)
    banc.valider(out, bits=2, budget=banc.budget_gardes())
    return out


def repartition_globale(budget=None, **kw):
    """Passe préliminaire sans poids gardés ; les scores OBS pondérés par l'importance (même unité : erreur de
    sortie) sont classés ensemble ; la part de chaque couche dans les `budget` meilleurs fixe sa répartition."""
    budget = budget or banc.budget_gardes()
    compresser(gardes=(0, 0, 0, 0), imp=True, **kw)
    sc = SCORES[-4:] if not kw.get("relu") else [SCORES[i] for i in (1, 3, 5, 6)]
    tout = torch.cat([x.flatten() for x in sc])
    seuil = torch.sort(tout, descending=True).values[budget - 1]
    n = [int((x >= seuil).sum()) for x in sc]
    n[-1] -= sum(n) - budget
    return tuple(n)


@torch.no_grad()
def valider_maison(compact, n=8192, graine=900):
    """Mes propres erreurs (graines distinctes de celles de banc.erreurs), en pas de VMAX : 1 pas (tout, bloc,
    bloc quand il bouge, agent), 8 pas en boucle ouverte (tout, bloc)."""
    m = banc.depuis_compact(compact) if isinstance(compact, dict) and "couches" in compact else compact
    s, a, s2 = banc.transitions(n, graine)
    p = m(s, a)
    e = lambda x, y: (x - y).norm(dim=1) / banc.VMAX
    bouge = (s2[:, 2:] - s[:, 2:]).norm(dim=1) > 1e-6
    g = torch.Generator().manual_seed(graine + 1)
    vrai = banc.etats_aleatoires(n, g); pred = vrai.clone()
    for _ in range(8):
        act = 2 * torch.rand(n, 2, generator=g) - 1
        vrai, pred = banc.pas(vrai, act), m(pred, act)
    r = {"un_pas": e(p, s2).mean(), "bloc_1": e(p[:, 2:], s2[:, 2:]).mean(), "bloc_bouge": e(p[bouge, 2:], s2[bouge, 2:]).mean(),
         "agent_1": e(p[:, :2], s2[:, :2]).mean(), "8_pas": e(pred, vrai).mean(), "bloc_8": e(pred[:, 2:], vrai[:, 2:]).mean(),
         "agent_8": e(pred[:, :2], vrai[:, :2]).mean()}
    return {k: round(v.item(), 4) for k, v in r.items()}


def trajectoires_poussee(n=2000, pas_=40, graine=31337, bruit=0.3):
    """Trajectoires proches de la planification, avec le vrai simulateur : départs d'épisodes (graine à part), un
    pousseur scripté (aller au point de poussée derrière le bloc, puis pousser vers le but), actions bruitées."""
    g = torch.Generator().manual_seed(graine)
    s, but = banc.episodes(n, graine)
    S, A = [s], []
    for _ in range(pas_):
        pa, pb = s[:, :2], s[:, 2:]
        v = but - pb; v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-6)
        point = pb - (banc.RA + banc.RB) * v
        vers_point = point - pa
        loin = vers_point.norm(dim=1, keepdim=True) > 0.02
        a = torch.where(loin, vers_point / banc.VMAX, v * ((but - pb).norm(dim=1, keepdim=True) / banc.VMAX).clamp(max=1))
        arrive = (but - pb).norm(dim=1, keepdim=True) < banc.TOL
        a = torch.where(arrive, torch.zeros_like(a), a)
        a = (a + bruit * torch.randn(n, 2, generator=g)).clamp(-1, 1)
        s = banc.pas(s, a)
        A.append(a); S.append(s)
    return torch.stack(S, 1), torch.stack(A, 1)        # (n, pas+1, 4), (n, pas, 2)


@torch.no_grad()
def valider_poussee(compact, h=8, graine=31337):
    """Erreur en boucle ouverte sur h pas le long des trajectoires de poussée, depuis chaque départ t multiple de h :
    bloc et agent, en pas de VMAX, moyennée sur les h pas (comme le coût de la CEM, cumulé sur l'horizon)."""
    m = banc.depuis_compact(compact) if isinstance(compact, dict) and "couches" in compact else compact
    S, A = trajectoires_poussee(graine=graine)
    eb, ea = [], []
    for t0 in range(0, A.shape[1] - h + 1, h):
        x = S[:, t0]
        for k in range(h):
            x = m(x, A[:, t0 + k])
            vrai = S[:, t0 + k + 1]
            eb.append((x[:, 2:] - vrai[:, 2:]).norm(dim=1) / banc.VMAX)
            ea.append((x[:, :2] - vrai[:, :2]).norm(dim=1) / banc.VMAX)
    eb, ea = torch.stack(eb), torch.stack(ea)
    return {"p_bloc": round(eb.mean().item(), 4), "p_agent": round(ea.mean().item(), 4)}


def critere(compact, graine_m=2000, graine_p=4444):
    """Critère de choix entre instances, fixé avant de regarder : erreurs bloc + 0,5 x agent (pondération du coût de
    la CEM), sur 8 pas en boucle ouverte (actions au hasard) et le long des trajectoires de poussée ; graines de
    validation neuves (2000, 4444), distinctes de celles utilisées pendant la mise au point."""
    v = valider_maison(compact, graine=graine_m); p = valider_poussee(compact, graine=graine_p)
    return round(v["bloc_8"] + 0.5 * v["agent_8"] + p["p_bloc"] + 0.5 * p["p_agent"], 4), v, p
