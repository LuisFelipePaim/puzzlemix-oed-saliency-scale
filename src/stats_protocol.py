"""
stats_protocol.py — protocolo estatístico pré-registrado.

Dois problemas do relatório anterior, ambos resolvidos aqui:

1. n=5 é insuficiente. O Wilcoxon pareado bilateral tem p mínimo de 0,0625 com
   n=5 — nem no caso extremo alcança 0,05. `min_wilcoxon_p()` calcula esse piso
   para você conferir antes de rodar. Com n=10 o piso cai para ~0,002.

2. Cinco métricas × três regimes sem correção é onde revisor puxa o fio. Aqui a
   família CONFIRMATÓRIA é declarada antes (métrica primária + comparações
   pré-especificadas) e leva correção de Holm; todo o resto é reportado como
   exploratório, sem p corrigido e sem alegação de conclusão.

Para graduação ordinal de displasia, a primária defensável é QWK; Erro>=2
(erro de dois ou mais graus) entra como secundária clínica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon

__all__ = ["Protocol", "min_wilcoxon_p", "holm", "analyze", "format_report"]

# métricas em que menor é melhor
LOWER_IS_BETTER = {"erro_ge2", "erro>=2", "error_ge2"}


def min_wilcoxon_p(n: int) -> float:
    """
    Menor p bilateral alcançável pelo Wilcoxon pareado com n pares — o piso que
    decide se o teste pode ou não dar significância antes de você rodar.
    """
    if n < 1:
        return 1.0
    x = np.arange(1, n + 1, dtype=float)
    try:
        return float(wilcoxon(x, np.zeros(n), zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def holm(pvals: Sequence[float], alpha: float = 0.05):
    """Holm-Bonferroni. Retorna (p_ajustados, rejeita) na ordem de entrada."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj, adj <= alpha


@dataclass
class Protocol:
    """
    Declare ANTES de rodar. Serve tanto para guiar a análise quanto para ir
    literalmente no texto do artigo como pré-registro.
    """
    primary_metric: str = "qwk"
    secondary_metrics: List[str] = field(default_factory=lambda: ["erro_ge2"])
    confirmatory_pairs: List[tuple] = field(
        default_factory=lambda: [("puzzlemix", "cutmix"),
                                 ("puzzlemix", "blockmix_random"),
                                 ("puzzlemix", "baseline")])
    alpha: float = 0.05
    n_seeds_planned: int = 10
    test: str = "wilcoxon"          # "wilcoxon" (não-paramétrico) ou "ttest"

    def describe(self) -> str:
        floor = min_wilcoxon_p(self.n_seeds_planned)
        return (
            f"Protocolo pré-registrado\n"
            f"  primária: {self.primary_metric}\n"
            f"  secundária(s): {', '.join(self.secondary_metrics)}\n"
            f"  família confirmatória: {len(self.confirmatory_pairs)} comparações "
            f"na primária, correção de Holm em alpha={self.alpha}\n"
            f"  seeds planejadas: {self.n_seeds_planned} "
            f"(piso do Wilcoxon = {floor:.4f})\n"
            f"  demais métricas e pares: EXPLORATÓRIOS, sem correção"
        )


def _paired(a: np.ndarray, b: np.ndarray, test: str) -> float:
    if np.allclose(a, b):
        return 1.0
    if test == "ttest":
        return float(ttest_rel(a, b).pvalue)
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def analyze(results: Dict[str, Dict[str, Sequence[float]]],
            protocol: Optional[Protocol] = None) -> dict:
    """
    results[regime][métrica] = lista de valores, um por seed, PAREADOS por seed
    (mesma ordem de seeds em todos os regimes).
    """
    protocol = protocol or Protocol()
    regimes = list(results)
    metrics = sorted({m for r in results.values() for m in r})
    n_seeds = min(len(v) for r in results.values() for v in r.values())
    out = {"n_seeds": n_seeds,
           "wilcoxon_floor": min_wilcoxon_p(n_seeds),
           "underpowered": min_wilcoxon_p(n_seeds) > protocol.alpha,
           "protocol": protocol,
           "descriptives": {}, "confirmatory": [], "exploratory": [], "omnibus": {}}

    for m in metrics:
        out["descriptives"][m] = {
            r: {"mean": float(np.mean(results[r][m][:n_seeds])),
                "sd": float(np.std(results[r][m][:n_seeds], ddof=1)) if n_seeds > 1 else 0.0}
            for r in regimes if m in results[r]
        }
        arrays = [np.asarray(results[r][m][:n_seeds], float) for r in regimes if m in results[r]]
        if len(arrays) >= 3 and n_seeds >= 3:
            out["omnibus"][m] = float(friedmanchisquare(*arrays).pvalue)

    # família confirmatória: só a primária, só os pares pré-declarados
    pm = protocol.primary_metric
    raw = []
    for a, b in protocol.confirmatory_pairs:
        if a in results and b in results and pm in results[a] and pm in results[b]:
            va = np.asarray(results[a][pm][:n_seeds], float)
            vb = np.asarray(results[b][pm][:n_seeds], float)
            raw.append({"pair": (a, b), "metric": pm,
                        "delta": float(np.mean(va - vb)),
                        "p_raw": _paired(va, vb, protocol.test)})
    if raw:
        adj, rej = holm([r["p_raw"] for r in raw], protocol.alpha)
        for r, pa, rj in zip(raw, adj, rej):
            r["p_holm"], r["significant"] = float(pa), bool(rj)
        out["confirmatory"] = raw

    # exploratório: todo o resto, sem correção e sem alegação
    for m in metrics:
        for a, b in combinations(regimes, 2):
            if m == pm and ((a, b) in protocol.confirmatory_pairs
                            or (b, a) in protocol.confirmatory_pairs):
                continue
            if m in results[a] and m in results[b]:
                va = np.asarray(results[a][m][:n_seeds], float)
                vb = np.asarray(results[b][m][:n_seeds], float)
                out["exploratory"].append(
                    {"pair": (a, b), "metric": m, "delta": float(np.mean(va - vb)),
                     "p_raw": _paired(va, vb, protocol.test)})
    return out


def format_report(res: dict) -> str:
    p: Protocol = res["protocol"]
    L = [p.describe(), ""]
    L.append(f"seeds efetivas: {res['n_seeds']}  |  piso do Wilcoxon: {res['wilcoxon_floor']:.4f}")
    if res["underpowered"]:
        L.append("  *** SUBDIMENSIONADO: nem no caso extremo o teste alcança alpha. "
                 "Trate TUDO como exploratório. ***")
    L.append("")

    L.append("Descritivas (media +/- dp)")
    for m, d in res["descriptives"].items():
        arrow = "menor melhor" if m.lower() in LOWER_IS_BETTER else "maior melhor"
        L.append(f"  [{m}] ({arrow})")
        for r, s in d.items():
            L.append(f"    {r:<20}{s['mean']:.4f} +/- {s['sd']:.4f}")
    L.append("")

    if res["omnibus"]:
        L.append("Friedman (omnibus, exploratorio)")
        for m, pv in res["omnibus"].items():
            L.append(f"  {m:<20}p={pv:.4f}")
        L.append("")

    L.append(f"CONFIRMATORIO — metrica primaria '{p.primary_metric}', Holm")
    for c in res["confirmatory"]:
        star = "SIM" if c["significant"] else "nao"
        L.append(f"  {c['pair'][0]:<18}vs {c['pair'][1]:<18}"
                 f"d={c['delta']:+.4f}  p={c['p_raw']:.4f}  p_holm={c['p_holm']:.4f}  "
                 f"rejeita H0: {star}")
    L.append("")
    L.append(f"EXPLORATORIO ({len(res['exploratory'])} comparacoes, SEM correcao — "
             "reportar como geracao de hipotese, nunca como conclusao)")
    for e in sorted(res["exploratory"], key=lambda z: z["p_raw"])[:10]:
        L.append(f"  [{e['metric']:<10}] {e['pair'][0]:<18}vs {e['pair'][1]:<18}"
                 f"d={e['delta']:+.4f}  p={e['p_raw']:.4f}")
    return "\n".join(L)
