"""Teste dos modulos novos: PuzzleMix oficial, regime de controle, protocolo."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn

from architectures import build_model
from puzzlemix_official import PuzzleMixOfficial, soft_target_cross_entropy, one_hot
from regimes import make_regime, REGIMES, blockmix_random, cutmix
from stats_protocol import Protocol, analyze, format_report, min_wilcoxon_p, holm

REPO = os.environ.get("PUZZLEMIX_REPO", str(ROOT / "PuzzleMix"))
DEV = "cpu"
NC = 4
torch.manual_seed(0); np.random.seed(0)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
x = torch.randn(8, 3, 64, 64)
y = torch.randint(0, NC, (8,))
model = build_model("vgg16_bn", NC, pretrained=False).to(DEV)

print("=" * 66)
print("1. PUZZLEMIX OFICIAL (snu-mllab)")
mixer = PuzzleMixOfficial(REPO, MEAN, STD, DEV, num_classes=NC, mp=None)
sal = mixer.saliency(model, x, y)
print(f"  saliencia (L2 oficial) shape={tuple(sal.shape)} "
      f"media={sal.mean():.5f}  [main.py:370 -> sqrt(mean(grad^2, canais))]")

for trial in range(3):
    xm, ys = mixer(model, x, y)
    ratios = ys.max(1).values
    n_lv = len(torch.unique(torch.round(ratios * 100) / 100))
    print(f"  mistura {trial+1}: x={tuple(xm.shape)} alvo_soft={tuple(ys.shape)} "
          f"soma_alvo={ys.sum(1).mean():.4f}  lam_eff medio={ratios.mean():.3f} "
          f"({n_lv} valores distintos entre amostras)")

# a mascara responde a saliencia? teste controlado com saliencia sintetica
import sys as _s; _s.path.insert(0, REPO)
from mixup import mixup_graph
mean_t = torch.tensor(MEAN).view(1,3,1,1); std_t = torch.tensor(STD).view(1,3,1,1)
xs = torch.rand(2,3,64,64); idx2 = np.array([1,0])
sal_conc = torch.full((2,64,64), 0.01); sal_conc[0,:,:32] = 1.0; sal_conc[1,:,32:] = 1.0
sal_flat = torch.rand(2,64,64)*0.001 + 0.5
for nome, sal_t in (("CONCENTRADA", sal_conc), ("PLANA", sal_flat)):
    rs = []
    for lam in (0.3, 0.5, 0.7):
        _, rt = mixup_graph(xs, sal_t, idx2, block_num=8, alpha=lam, beta=1.2, gamma=0.5,
                            eta=0.2, neigh_size=4, n_labels=3, mean=mean_t, std=std_t,
                            transport=False, device=DEV)
        rs.append(round(rt[0].item(), 3))
    print(f"  saliencia {nome:<12} ratio p/ lam=0.3/0.5/0.7 -> {rs}")
print("  -> concentrada: mascara segue a regiao saliente (ratio ~0.5, ignora o prior)")
print("  -> plana: unario empata, o prior decide, mascara COLAPSA em {0, 0.5, 1}")
print("     nesse regime Puzzle Mix degenera em input mixup e o transporte vira identidade")

mixer.reset_stats()
for _ in range(6):
    mixer(model, x, y)
st = mixer.stats
print(f"  telemetria com rede de pesos ALEATORIOS: collapse_rate={st['collapse_rate']:.2f} "
      f"saliency_top10={st['saliency_top10']:.3f} (uniforme daria ~0.10)")
print("  [esperado colapsar aqui: rede nao treinada -> saliencia sem estrutura;")
print("   a medicao que vale e' com o checkpoint treinado no OED]")

loss = soft_target_cross_entropy(model(xm), ys)
loss.backward()
print(f"  perda soft-target: {loss.item():.4f}  backward OK")

print("=" * 66)
print("2. REGIME DE CONTROLE (blockmix_random)")
for nl in (2, 3):
    xr, yr = blockmix_random(x, y, NC, alpha=1.0, n_labels=nl)
    lv = torch.unique(torch.round(yr.max(1).values * 100) / 100)
    print(f"  n_labels={nl}: lam_eff medio={yr.max(1).values.mean():.3f} "
          f"niveis distintos={len(lv)}")

xc, yc = cutmix(x, y, NC)
print(f"  cutmix: lam_eff={yc.max(1).values.mean():.3f} (um retangulo, ratio unico)")

for name in REGIMES:
    fn = make_regime(name, NC, puzzle_mixer=mixer if name == "puzzlemix" else None)
    xo, yo = fn(model, x, y)
    assert xo.shape == x.shape and yo.shape == (8, NC), name
    print(f"  {name:<18} -> x={tuple(xo.shape)} y={tuple(yo.shape)} "
          f"soma_alvo={yo.sum(1).mean():.3f}  OK")

print("=" * 66)
print("3. PROTOCOLO ESTATISTICO")
for n in (5, 8, 10, 12):
    print(f"  piso do Wilcoxon com n={n:>2}: {min_wilcoxon_p(n):.4f}"
          + ("   <- nao alcanca 0.05" if min_wilcoxon_p(n) > 0.05 else ""))

adj, rej = holm([0.01, 0.04, 0.03], alpha=0.05)
print(f"  Holm([0.01,0.04,0.03]) -> ajustados={[round(a,4) for a in adj]} rejeita={list(rej)}")

rng = np.random.default_rng(0)
n_seeds = 10
res = {
    "baseline":        {"qwk": 0.928 + rng.normal(0, .012, n_seeds), "erro_ge2": 0.013 + rng.normal(0, .004, n_seeds)},
    "cutmix":          {"qwk": 0.925 + rng.normal(0, .012, n_seeds), "erro_ge2": 0.020 + rng.normal(0, .004, n_seeds)},
    "blockmix_random": {"qwk": 0.918 + rng.normal(0, .012, n_seeds), "erro_ge2": 0.021 + rng.normal(0, .004, n_seeds)},
    "puzzlemix":       {"qwk": 0.914 + rng.normal(0, .012, n_seeds), "erro_ge2": 0.022 + rng.normal(0, .004, n_seeds)},
}
out = analyze(res, Protocol(n_seeds_planned=n_seeds))
print()
print(format_report(out))

print()
out5 = analyze({k: {m: v[:5] for m, v in d.items()} for k, d in res.items()},
               Protocol(n_seeds_planned=5))
print(f"  com n=5 o analisador marca underpowered={out5['underpowered']} "
      f"(piso={out5['wilcoxon_floor']:.4f})")
print("=" * 66)
print("TODOS OS TESTES PASSARAM")
