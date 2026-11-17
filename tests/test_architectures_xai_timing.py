"""Teste de ponta a ponta dos tres modulos com dados sinteticos."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from architectures import (ARCHITECTURES, build_model, cam_target_layer,
                           cam_spatial_size, count_parameters)
from xai_metrics import (GradCAM, explanation_drift, road_scores, pointing_game,
                         mask_iou, evaluate_xai, compare_models_xai,
                         noisy_linear_imputation)
from timing import StageTimer, extrapolate, matrix_size, probe_report

torch.manual_seed(0)
DEV = "cpu"
N, C = 24, 4

x = torch.randn(N, 3, 64, 64)
y = torch.randint(0, C, (N,))
loader = DataLoader(TensorDataset(x, y), batch_size=8)

print("=" * 62)
print("1. ARQUITETURAS")
for arch in ARCHITECTURES:
    m = build_model(arch, num_classes=C, pretrained=False)
    p = count_parameters(m)
    layer = cam_target_layer(m)
    shape = cam_spatial_size(m, 224)
    shape96 = cam_spatial_size(m, 96)
    print(f"  {arch:<17} total={p['total']/1e6:6.2f}M conv={p['conv']/1e6:6.2f}M "
          f"head={p['head']:>8,}  cam@224={shape} cam@96={shape96}  "
          f"ancora={type(layer).__name__}")

print("\n  modo container (imune a reorganizacao interna):")
m = build_model("vgg16_bn", C, pretrained=False)
print(f"    vgg16_bn last_act={cam_spatial_size(m, 224, 'last_act')} "
      f"container={cam_spatial_size(m, 224, 'container')}")

# estabilidade do hook a uma "decomposicao": expande uma conv em 3 camadas
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, C))
    def forward(self, s): return self.head(self.features(s))

base = TinyNet()
decomp = TinyNet()
# simula Tucker: conv 3x3 -> (1x1, 3x3, 1x1), deslocando indices dentro de features
decomp.features[4] = nn.Sequential(
    nn.Conv2d(8, 6, 1), nn.Conv2d(6, 6, 3, padding=1), nn.Conv2d(6, 16, 1))
l_base, l_dec = cam_target_layer(base), cam_target_layer(decomp)
print(f"  hook sobrevive a expansao de camadas: base={type(l_base).__name__} "
      f"decomposto={type(l_dec).__name__} -> {'OK' if type(l_base) is type(l_dec) else 'FALHOU'}")

print("=" * 62)
print("2. EXPLICABILIDADE")
with GradCAM(base, l_base) as cam_fn:
    cams_a = cam_fn(x[:8], y[:8])
print(f"  gradcam shape={tuple(cams_a.shape)} range=[{cams_a.min():.3f},{cams_a.max():.3f}]")

with GradCAM(decomp, l_dec) as cam_fn:
    cams_b = cam_fn(x[:8], y[:8])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    d_self = explanation_drift(cams_a, cams_a)
    d_pair = explanation_drift(cams_a, cams_b)
print(f"  deriva contra si mesmo (sanidade): spearman={d_self['spearman']:.3f} "
      f"iou_top10={d_self['iou_top10']:.3f}   [esperado ~1.0]")
print(f"  deriva base x decomposto:          spearman={d_pair['spearman']:.3f} "
      f"iou_top10={d_pair['iou_top10']:.3f} iou_top20={d_pair['iou_top20']:.3f}")

# imputacao: pixels mantidos nao podem mudar
keep = torch.rand(4, 1, 64, 64) > 0.5
xi = noisy_linear_imputation(x[:4], keep, iters=10, noise=0.0)
same = torch.allclose(xi[keep.expand_as(xi)], x[:4][keep.expand_as(xi)], atol=1e-5)
print(f"  imputacao preserva pixels mantidos: {'OK' if same else 'FALHOU'}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = road_scores(base, x[:8], cams_a, y[:8], percentiles=(10, 50, 90))
print(f"  ROAD morf_auc={r['road_morf_auc']:.4f} lerf_auc={r['road_lerf_auc']:.4f} "
      f"combined={r['road_combined']:+.4f}")
print(f"       curva morf={[round(v,3) for v in r['road_morf_curve']]}")
print(f"       curva lerf={[round(v,3) for v in r['road_lerf_curve']]}")

masks = (torch.rand(8, 64, 64) > 0.7)
print(f"  pointing_game={pointing_game(cams_a, masks):.3f} "
      f"mask_iou={mask_iou(cams_a, masks):.3f}  [so no PCam]")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    single = evaluate_xai(base, loader, l_base, device=DEV, road_percentiles=(10, 50, 90))
    paired = compare_models_xai(base, decomp, loader, l_base, l_dec, device=DEV,
                                road_percentiles=(10, 50, 90))
print(f"  evaluate_xai: acc={single['accuracy']:.3f} road_combined={single['road_combined']:+.4f}")
print(f"  compare_models_xai: delta_acc={paired['delta_accuracy']:+.3f} "
      f"drift_spearman={paired['drift_spearman']:.3f} "
      f"delta_road={paired['delta_road_combined']:+.4f}")
print(f"  chaves p/ CSV: {sorted(k for k in paired if not k.startswith('road_'))}")

# CAM degenerada nao pode quebrar
zero = torch.zeros(2, 8, 8)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    dz = explanation_drift(zero, zero)
print(f"  CAM degenerada tratada: spearman={dz['spearman']} n_degenerate={dz['n_degenerate']}")

# alerta de resolucao
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    explanation_drift(torch.rand(2, 3, 3), torch.rand(2, 3, 3))
    print(f"  alerta de resolucao 3x3 disparado: {'OK' if w else 'FALHOU'}")

print("=" * 62)
print("3. CRONOMETRAGEM")
timer = StageTimer(cuda=False)
model = TinyNet()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

def fake_mixup(xb, yb):
    import time as _t; _t.sleep(0.004)   # simula o alfa-beta swap em CPU
    return xb, yb

mixup = timer.wrap("mixup", fake_mixup)
for epoch in range(2):
    for xb, yb in loader:
        with timer.stage("data"):
            xb, yb = xb.to(DEV), yb.to(DEV)
        xb, yb = mixup(xb, yb)
        with timer.stage("fwd_bwd"):
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
    timer.end_epoch()

print(timer.report())
print(f"\n  linha de CSV: {timer.to_csv_row()}")

cells = matrix_size(n_arch=2, n_datasets=1, n_regimes=3, n_decompositions=2,
                    n_thresholds=3, n_seeds=3)
print(f"\n  matriz OED (2 arq x 1 ds x 3 regimes x 2 decomp x 3 thr x 3 seeds) = {cells} celulas")
print()
print(probe_report(timer, epochs_full=100, n_cells=cells,
                   eval_seconds_per_cell=600, deadline_days=30))
print("=" * 62)
print("TODOS OS TESTES PASSARAM")
