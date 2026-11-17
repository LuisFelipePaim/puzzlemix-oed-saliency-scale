"""
timing.py — cronometragem por estágio, para a sonda de custo.

O ponto não é medir "tempo por época", e sim SEPARAR os estágios. O gargalo
esperado do Puzzle Mix não é a GPU: o forward+backward extra de saliência
(input_saliency) e o α-β swap (corte mínimo binário, PyMaxflow — não pyGCO,
ver nota abaixo) rodam em CPU, exemplo a exemplo. Se `mixup` domina o
relógio enquanto `fwd_bwd` está baixo, o problema é CPU e a solução é
paralelizar/reduzir a grade da máscara — não trocar de GPU.

NOTA (corrigido em 2026-08-13, ver puzzlemix_sanity_check/RELATORIO.md):
até aqui este docstring assumia pyGCO para o α-β swap, mas a implementação
real (pipeline_sibgrapi.ipynb, Seção 3) usa PyMaxflow. Motivo: pyGCO /
gco-wrapper não tem wheel pré-compilada para Windows e exige MSVC Build
Tools para compilar do fonte (falhou neste ambiente). Isso não muda a
suposição de custo abaixo: para K=2 rótulos (bloco vem de A ou de B), o
algoritmo geral de troca alpha-beta se reduz a UM ÚNICO corte mínimo exato
— PyMaxflow resolve exatamente o mesmo problema que pyGCO resolveria aqui,
só que sem a dependência C++. Também vale notar que `StageTimer` não está
importado em nenhuma célula do notebook hoje — esta é uma sonda avulsa,
não uma instrumentação do treino real; rode-a manualmente se quiser medir
o custo do α-β swap antes de orçar a matriz completa.

Uso:

    timer = StageTimer(cuda=torch.cuda.is_available())
    for epoch in range(n_epochs):
        for batch in loader:                      # o próprio laço mede o data load
            with timer.stage("data"):
                x, y = batch[0].to(dev), batch[1].to(dev)
            with timer.stage("mixup"):
                x, ya, yb, lam = mixup_fn(x, y)
            with timer.stage("fwd_bwd"):
                ...
        timer.end_epoch()
    print(timer.report())
    timer.to_csv_row()        # -> dict para juntar na linha do CSV da célula
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional

try:
    import torch
except ImportError:  # permite usar o módulo sem torch
    torch = None


class StageTimer:
    """
    Acumula wall-clock por estágio nomeado.

    `cuda=True` sincroniza antes de parar o cronômetro. Sem isso, chamadas CUDA
    são assíncronas e o tempo medido cai no estágio errado — tipicamente todo o
    custo de GPU aparece colado no `.item()` seguinte, o que faria a sonda
    concluir que o mixup é barato quando não é.
    """

    def __init__(self, cuda: bool = False):
        self.cuda = cuda and torch is not None and torch.cuda.is_available()
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.epochs = []
        self._epoch_start = time.perf_counter()
        self._t0 = self._epoch_start

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    @contextmanager
    def stage(self, name: str):
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            dt = time.perf_counter() - t0
            self.totals[name] += dt
            self.counts[name] += 1

    def wrap(self, name: str, fn):
        """Instrumenta uma função sem tocar no corpo dela (ex.: a fn de mixup)."""
        def wrapped(*args, **kwargs):
            with self.stage(name):
                return fn(*args, **kwargs)
        return wrapped

    def end_epoch(self):
        now = time.perf_counter()
        self.epochs.append(now - self._epoch_start)
        self._epoch_start = now

    # -- saída ------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def seconds_per_epoch(self) -> float:
        return sum(self.epochs) / len(self.epochs) if self.epochs else float("nan")

    def summary(self) -> dict:
        total = sum(self.totals.values()) or 1.0
        return {
            name: {
                "seconds": round(t, 3),
                "calls": self.counts[name],
                "ms_per_call": round(1000 * t / max(self.counts[name], 1), 2),
                "share": round(100 * t / total, 1),
            }
            for name, t in sorted(self.totals.items(), key=lambda kv: -kv[1])
        }

    def to_csv_row(self, prefix: str = "t_") -> dict:
        """Colunas planas para anexar à linha da célula no CSV."""
        row = {f"{prefix}{k}_s": round(v, 3) for k, v in self.totals.items()}
        row[f"{prefix}epoch_mean_s"] = round(self.seconds_per_epoch(), 3)
        row[f"{prefix}epochs_measured"] = len(self.epochs)
        return row

    def report(self) -> str:
        lines = [f"{'estagio':<14}{'seg':>10}{'chamadas':>10}{'ms/chamada':>13}{'%':>8}"]
        for name, s in self.summary().items():
            lines.append(f"{name:<14}{s['seconds']:>10.2f}{s['calls']:>10}"
                         f"{s['ms_per_call']:>13.2f}{s['share']:>7.1f}%")
        lines.append(f"\nepocas medidas: {len(self.epochs)}  "
                     f"media: {self.seconds_per_epoch():.2f}s/epoca")
        return "\n".join(lines)


def extrapolate(seconds_per_epoch: float, epochs: int, n_cells: int = 1,
                eval_seconds_per_cell: float = 0.0) -> dict:
    """
    Extrapola a sonda para a matriz inteira. O custo por época é praticamente
    constante, então a extrapolação linear erra pouco — desde que a sonda tenha
    rodado ao menos 2 épocas (a primeira inclui warm-up de cuDNN e cache).
    """
    train = seconds_per_epoch * epochs * n_cells
    total = train + eval_seconds_per_cell * n_cells
    return {
        "cells": n_cells,
        "train_hours": round(train / 3600, 2),
        "eval_hours": round(eval_seconds_per_cell * n_cells / 3600, 2),
        "total_hours": round(total / 3600, 2),
        "total_days": round(total / 86400, 2),
    }


def matrix_size(n_arch: int, n_datasets: int, n_regimes: int,
                n_decompositions: int = 1, n_thresholds: int = 1,
                n_seeds: int = 1) -> int:
    """Contagem de células. Multiplique por seeds antes de olhar o total."""
    return n_arch * n_datasets * n_regimes * n_decompositions * n_thresholds * n_seeds


def probe_report(timer: StageTimer, epochs_full: int, n_cells: int,
                 eval_seconds_per_cell: float = 0.0,
                 deadline_days: Optional[float] = None) -> str:
    est = extrapolate(timer.seconds_per_epoch(), epochs_full, n_cells,
                      eval_seconds_per_cell)
    lines = [timer.report(), "",
             f"matriz: {n_cells} celulas x {epochs_full} epocas",
             f"treino estimado: {est['train_hours']}h",
             f"avaliacao estimada: {est['eval_hours']}h",
             f"TOTAL: {est['total_hours']}h ({est['total_days']} dias de GPU continua)"]
    if deadline_days is not None:
        folga = deadline_days - est["total_days"]
        lines.append(f"prazo: {deadline_days} dias -> folga de {folga:.1f} dias"
                     + ("  [NAO CABE]" if folga < 0 else ""))
    return "\n".join(lines)
