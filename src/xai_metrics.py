"""
xai_metrics.py — métricas de explicabilidade para o eixo "decomposição × Grad-CAM".

Três famílias, conforme decidido:

1. DERIVA (base × decomposto) — quanto a explicação se move. Sem ground truth.
   - spearman: correlação de rank entre os mapas achatados (robusta a escala)
   - iou_top10 / iou_top20: IoU dos k% pixels mais quentes (captura migração
     de pico, que a correlação dilui)
   Correlação alta + IoU baixa = ranking global preservado, pico deslocado.

2. FIDELIDADE — a explicação corresponde ao que o modelo usa.
   ROAD (Rong et al., ICML 2022) em vez de deletion/insertion clássico: apagar
   região gera imagem fora da distribuição e parte da queda de confiança vem do
   artefato, não da importância. ROAD imputa por vizinhança e corrige isso.
   Crítico em histopatologia, onde a textura é o sinal.

3. LOCALIZAÇÃO (opcional, só onde há máscara — PCam, não OED).
   - pointing_game, mask_iou

CUSTO: ROAD faz len(percentis) × 2 forwards por imagem. Com 9 percentis são 18
forwards/imagem. Meça em ~50 imagens antes de soltar no conjunto inteiro.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

__all__ = [
    "GradCAM", "explanation_drift", "road_scores", "pointing_game", "mask_iou",
    "evaluate_xai", "compare_models_xai", "MIN_CAM_SIDE",
]

# Abaixo disso, IoU e pointing game viram ruído (ex.: 3×3 = 9 células no PCam 96px).
MIN_CAM_SIDE = 7


# --------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM sem dependências externas.

    Guarda a ativação pelo forward hook e obtém o gradiente por autograd.grad
    (mais previsível que backward hooks, que mudam de semântica entre versões
    do PyTorch e quebram em módulos reescritos pela decomposição).

    O `target_layer` deve ser resolvido DEPOIS de decompor o modelo — use
    architectures.cam_target_layer(model).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activation = None
        self._handle = target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self._activation = output

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    @torch.enable_grad()
    def __call__(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None,
                 upsample_to: Optional[int] = None) -> torch.Tensor:
        """
        x: (B,3,H,W) já normalizado.
        targets: (B,) índices de classe; None = classe predita.
        Retorna (B,h,w) normalizado em [0,1] por amostra.
        """
        was_training = self.model.training
        self.model.eval()

        logits = self.model(x)
        if targets is None:
            targets = logits.argmax(dim=1)
        score = logits.gather(1, targets.view(-1, 1)).sum()

        act = self._activation
        if act is None:
            raise RuntimeError("hook não capturou ativação — target_layer não foi executado")
        grads = torch.autograd.grad(score, act, retain_graph=False)[0]

        weights = grads.mean(dim=(2, 3), keepdim=True)          # GAP dos gradientes
        cam = F.relu((weights * act).sum(dim=1))                 # (B,h,w)

        # normalização por amostra
        b = cam.shape[0]
        flat = cam.view(b, -1)
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
        cam = ((flat - lo) / (hi - lo + 1e-8)).view_as(cam)

        if upsample_to is not None:
            cam = F.interpolate(cam.unsqueeze(1), size=(upsample_to, upsample_to),
                                mode="bilinear", align_corners=False).squeeze(1)

        if was_training:
            self.model.train()
        return cam.detach()


# --------------------------------------------------------------------------
# 1. Deriva da explicação
# --------------------------------------------------------------------------

def _topk_mask(cam: np.ndarray, frac: float) -> np.ndarray:
    k = max(1, int(round(frac * cam.size)))
    thr = np.partition(cam.ravel(), -k)[-k]
    return cam >= thr


def explanation_drift(cam_a: torch.Tensor, cam_b: torch.Tensor,
                      top_fracs: Sequence[float] = (0.10, 0.20)) -> dict:
    """
    Deriva entre dois conjuntos de mapas (mesmo lote, mesma resolução).
    cam_a, cam_b: (B,h,w). Retorna médias por lote.
    """
    if cam_a.shape != cam_b.shape:
        raise ValueError(f"shapes diferentes: {tuple(cam_a.shape)} vs {tuple(cam_b.shape)}")
    if min(cam_a.shape[-2:]) < MIN_CAM_SIDE:
        warnings.warn(
            f"CAM {tuple(cam_a.shape[-2:])} abaixo de {MIN_CAM_SIDE}×{MIN_CAM_SIDE}: "
            "IoU top-k perde sentido. Ancore num bloco intermediário ou aumente a entrada.",
            stacklevel=2,
        )

    a = cam_a.detach().float().cpu().numpy()
    b = cam_b.detach().float().cpu().numpy()

    rho, ious = [], {f"iou_top{int(f*100)}": [] for f in top_fracs}
    for i in range(a.shape[0]):
        ai, bi = a[i].ravel(), b[i].ravel()
        if ai.std() < 1e-8 or bi.std() < 1e-8:
            rho.append(np.nan)                    # mapa degenerado (CAM toda zero)
        else:
            rho.append(spearmanr(ai, bi).statistic)
        for f in top_fracs:
            ma, mb = _topk_mask(a[i], f), _topk_mask(b[i], f)
            union = np.logical_or(ma, mb).sum()
            ious[f"iou_top{int(f*100)}"].append(
                np.logical_and(ma, mb).sum() / union if union else np.nan)

    out = {"spearman": float(np.nanmean(rho))}
    out.update({k: float(np.nanmean(v)) for k, v in ious.items()})
    out["n_degenerate"] = int(np.isnan(rho).sum())
    return out


# --------------------------------------------------------------------------
# 2. Fidelidade — ROAD
# --------------------------------------------------------------------------

_NLI_KERNEL = torch.tensor(
    [[1 / np.sqrt(2), 1.0, 1 / np.sqrt(2)],
     [1.0, 0.0, 1.0],
     [1 / np.sqrt(2), 1.0, 1 / np.sqrt(2)]], dtype=torch.float32
)


def noisy_linear_imputation(x: torch.Tensor, keep: torch.Tensor,
                            iters: int = 30, noise: float = 0.01) -> torch.Tensor:
    """
    Imputação linear ruidosa do ROAD: pixel removido = média ponderada dos 8
    vizinhos (diagonais com peso 1/√2). O sistema linear exato é resolvido por
    Jacobi vetorizado em GPU — aproximação documentada do solver esparso do
    paper original, suficientemente convergida em ~30 iterações.

    x: (B,C,H,W) normalizado. keep: (B,1,H,W) booleano, True = pixel mantido.
    """
    kernel = _NLI_KERNEL.to(x.device).view(1, 1, 3, 3)
    keep_f = keep.float()
    c = x.shape[1]
    k_rep = kernel.repeat(c, 1, 1, 1)

    # denominador: soma dos pesos dos vizinhos existentes (borda inclusa)
    denom = F.conv2d(torch.ones_like(keep_f), kernel, padding=1).clamp_min(1e-6)

    out = x * keep_f  # pixels removidos começam em 0
    for _ in range(iters):
        neigh = F.conv2d(out, k_rep, padding=1, groups=c) / denom
        out = torch.where(keep.expand_as(x), x, neigh)

    if noise > 0:
        out = torch.where(keep.expand_as(x), out, out + noise * torch.randn_like(out))
    return out


@torch.no_grad()
def road_scores(model: nn.Module, x: torch.Tensor, cams: torch.Tensor,
                targets: torch.Tensor,
                percentiles: Sequence[float] = (10, 20, 30, 40, 50, 60, 70, 80, 90),
                imputation_iters: int = 30) -> dict:
    """
    ROAD MoRF e LeRF.

    MoRF: remove primeiro o mais importante -> a probabilidade deve CAIR rápido.
    LeRF: remove primeiro o menos importante -> deve cair devagar.
    Score combinado = AUC(LeRF) - AUC(MoRF); maior = explicação mais fiel.

    cams: (B,h,w); reamostrado para o tamanho de x automaticamente.
    Retorna probabilidades médias da classe alvo por percentil e as AUCs.
    """
    model.eval()
    b, _, h, w = x.shape
    if cams.shape[-2:] != (h, w):
        cams = F.interpolate(cams.unsqueeze(1).float(), size=(h, w),
                             mode="bilinear", align_corners=False).squeeze(1)
    flat = cams.reshape(b, -1)
    order_desc = flat.argsort(dim=1, descending=True)   # mais importante primeiro
    n_pix = flat.shape[1]

    curves = {"morf": [], "lerf": []}
    for p in percentiles:
        n_rm = int(round(n_pix * p / 100.0))
        for direction in ("morf", "lerf"):
            idx = order_desc[:, :n_rm] if direction == "morf" else order_desc[:, -n_rm:]
            keep = torch.ones(b, n_pix, dtype=torch.bool, device=x.device)
            keep.scatter_(1, idx, False)
            keep = keep.view(b, 1, h, w)
            x_imp = noisy_linear_imputation(x, keep, iters=imputation_iters)
            probs = F.softmax(model(x_imp), dim=1).gather(1, targets.view(-1, 1)).squeeze(1)
            curves[direction].append(probs.mean().item())

    pct = np.asarray(percentiles, dtype=float) / 100.0
    auc_morf = float(np.trapezoid(curves["morf"], pct) / (pct[-1] - pct[0]))
    auc_lerf = float(np.trapezoid(curves["lerf"], pct) / (pct[-1] - pct[0]))
    return {
        "road_morf_auc": auc_morf,
        "road_lerf_auc": auc_lerf,
        "road_combined": auc_lerf - auc_morf,
        "road_morf_curve": curves["morf"],
        "road_lerf_curve": curves["lerf"],
        "road_percentiles": list(percentiles),
    }


# --------------------------------------------------------------------------
# 3. Localização (só com máscara — PCam)
# --------------------------------------------------------------------------

def pointing_game(cams: torch.Tensor, masks: torch.Tensor) -> float:
    """Fração de amostras em que o pico da CAM cai dentro da máscara."""
    b = cams.shape[0]
    if cams.shape[-2:] != masks.shape[-2:]:
        cams = F.interpolate(cams.unsqueeze(1).float(), size=masks.shape[-2:],
                             mode="bilinear", align_corners=False).squeeze(1)
    peak = cams.view(b, -1).argmax(dim=1)
    hit = masks.reshape(b, -1).bool().gather(1, peak.view(-1, 1)).squeeze(1)
    return hit.float().mean().item()


def mask_iou(cams: torch.Tensor, masks: torch.Tensor, frac: float = 0.20) -> float:
    """IoU entre os top-frac% da CAM e a máscara de referência."""
    if cams.shape[-2:] != masks.shape[-2:]:
        cams = F.interpolate(cams.unsqueeze(1).float(), size=masks.shape[-2:],
                             mode="bilinear", align_corners=False).squeeze(1)
    c = cams.detach().cpu().numpy()
    m = masks.detach().cpu().numpy().astype(bool)
    vals = []
    for i in range(c.shape[0]):
        pred = _topk_mask(c[i], frac)
        union = np.logical_or(pred, m[i]).sum()
        vals.append(np.logical_and(pred, m[i]).sum() / union if union else np.nan)
    return float(np.nanmean(vals))


# --------------------------------------------------------------------------
# Orquestração
# --------------------------------------------------------------------------

def evaluate_xai(model: nn.Module, loader: Iterable, target_layer: nn.Module,
                 device: str = "cuda", max_batches: Optional[int] = None,
                 with_road: bool = True, road_percentiles=(10, 30, 50, 70, 90),
                 use_true_label: bool = True) -> dict:
    """
    Métricas de um único modelo (fidelidade + acurácia). Use quando não há par
    base/decomposto — ex.: comparar regimes de treino entre si.
    """
    model = model.to(device).eval()
    cam_fn = GradCAM(model, target_layer)
    acc, n, road_acc = 0, 0, []

    try:
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                pred = model(x).argmax(1)
            acc += (pred == y).sum().item()
            n += y.numel()

            targets = y if use_true_label else pred
            cams = cam_fn(x, targets)
            if with_road:
                road_acc.append(road_scores(model, x, cams, targets,
                                            percentiles=road_percentiles))
    finally:
        cam_fn.remove()

    out = {"accuracy": acc / max(n, 1), "n_samples": n}
    if road_acc:
        for k in ("road_morf_auc", "road_lerf_auc", "road_combined"):
            out[k] = float(np.mean([r[k] for r in road_acc]))
        out["road_percentiles"] = list(road_percentiles)
    return out


def compare_models_xai(model_base: nn.Module, model_other: nn.Module,
                       loader: Iterable, layer_base: nn.Module, layer_other: nn.Module,
                       device: str = "cuda", max_batches: Optional[int] = None,
                       with_road: bool = True, road_percentiles=(10, 30, 50, 70, 90),
                       top_fracs: Sequence[float] = (0.10, 0.20)) -> dict:
    """
    Comparação pareada base × decomposto — a chamada principal do eixo 2.

    IMPORTANTE: as camadas devem ser resolvidas separadamente em cada modelo
    (cam_target_layer(model_base) e cam_target_layer(model_other)), nunca
    reaproveitadas, porque a decomposição reescreve os módulos.

    O par (delta_accuracy, deriva) é o que deve ir para a tabela: deriva
    isolada não distingue "a decomposição mudou a explicação" de "o modelo
    ficou pior e por isso explica diferente".
    """
    model_base, model_other = model_base.to(device).eval(), model_other.to(device).eval()
    cam_a, cam_b = GradCAM(model_base, layer_base), GradCAM(model_other, layer_other)

    drift_acc, road_a, road_b = [], [], []
    acc_a = acc_b = n = 0
    try:
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                acc_a += (model_base(x).argmax(1) == y).sum().item()
                acc_b += (model_other(x).argmax(1) == y).sum().item()
            n += y.numel()

            ca, cb = cam_a(x, y), cam_b(x, y)
            if ca.shape[-2:] != cb.shape[-2:]:  # resoluções divergiram: iguala p/ comparar
                cb = F.interpolate(cb.unsqueeze(1).float(), size=ca.shape[-2:],
                                   mode="bilinear", align_corners=False).squeeze(1)
            drift_acc.append(explanation_drift(ca, cb, top_fracs))

            if with_road:
                road_a.append(road_scores(model_base, x, ca, y, road_percentiles))
                road_b.append(road_scores(model_other, x, cb, y, road_percentiles))
    finally:
        cam_a.remove()
        cam_b.remove()

    out = {"n_samples": n,
           "acc_base": acc_a / max(n, 1),
           "acc_other": acc_b / max(n, 1)}
    out["delta_accuracy"] = out["acc_other"] - out["acc_base"]
    for k in drift_acc[0]:
        out[f"drift_{k}"] = float(np.mean([d[k] for d in drift_acc]))
    if road_a:
        for k in ("road_morf_auc", "road_lerf_auc", "road_combined"):
            out[f"base_{k}"] = float(np.mean([r[k] for r in road_a]))
            out[f"other_{k}"] = float(np.mean([r[k] for r in road_b]))
        out["delta_road_combined"] = out["other_road_combined"] - out["base_road_combined"]
    return out
