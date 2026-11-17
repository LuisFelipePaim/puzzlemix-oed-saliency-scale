"""
regimes.py — regimes de mistura, incluindo o CONTROLE que faltava no desenho.

O problema do desenho anterior: comparar "blocos guiados por saliência" contra
"um retângulo do CutMix" muda DUAS coisas ao mesmo tempo — a granularidade da
máscara e a guia por saliência. Quando o guiado perde, não dá para saber qual
das duas causou.

`blockmix_random` fecha esse buraco: mesma grade, mesma distribuição de λ, mesma
estatística de máscara — só que os blocos são escolhidos ao acaso. A diferença
entre ele e o Puzzle Mix isola exatamente a contribuição da saliência, que é a
hipótese do artigo.

Todos os regimes devolvem (x_misturado, alvo_soft) e são treinados com
soft_target_cross_entropy, para que a perda não seja um confundidor entre eles.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from puzzlemix_official import one_hot, soft_target_cross_entropy  # noqa: F401

__all__ = ["baseline", "input_mixup", "cutmix", "blockmix_random", "make_regime",
           "soft_target_cross_entropy"]


def _lam(alpha: float) -> float:
    return float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0


def baseline(x, y, num_classes, **kw):
    return x, one_hot(y, num_classes)


def input_mixup(x, y, num_classes, alpha: float = 1.0, **kw):
    lam = _lam(alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    y1 = one_hot(y, num_classes)
    return lam * x + (1 - lam) * x[idx], lam * y1 + (1 - lam) * y1[idx]


def cutmix(x, y, num_classes, alpha: float = 1.0, **kw):
    """CutMix padrão: um retângulo. ratio efetivo medido, não o λ sorteado."""
    lam = _lam(alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, h, w = x.shape
    rh, rw = int(h * np.sqrt(1 - lam)), int(w * np.sqrt(1 - lam))
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1_, y2_ = np.clip(cy - rh // 2, 0, h), np.clip(cy + rh // 2, 0, h)
    x1_, x2_ = np.clip(cx - rw // 2, 0, w), np.clip(cx + rw // 2, 0, w)

    out = x.clone()
    out[:, :, y1_:y2_, x1_:x2_] = x[idx][:, :, y1_:y2_, x1_:x2_]
    ratio = 1 - ((y2_ - y1_) * (x2_ - x1_) / (h * w))
    y1 = one_hot(y, num_classes)
    return out, ratio * y1 + (1 - ratio) * y1[idx]


def blockmix_random(x, y, num_classes, alpha: float = 1.0,
                    block_nums: Sequence[int] = (2, 4, 8, 16),
                    n_labels: int = 3, **kw):
    """
    CONTROLE. Máscara de blocos aleatória com a mesma grade e o mesmo espaço de
    rótulos do Puzzle Mix oficial — sem saliência, sem suavidade, sem transporte.

    n_labels=3 reproduz z ∈ {0, ½, 1} do oficial; n_labels=2 dá máscara binária.
    A máscara é sorteada de forma independente POR AMOSTRA, como no oficial (que
    resolve um graph-cut por exemplo), e não uma máscara única por batch.
    """
    b, _, h, w = x.shape
    lam = _lam(alpha)
    idx = torch.randperm(b, device=x.device)
    block_num = int(np.random.choice(list(block_nums)))

    levels = torch.linspace(0, 1, n_labels, device=x.device)
    # sorteia nível por bloco com E[z] = lam (mesmo alvo do termo de prior η)
    probs = torch.full((b, block_num * block_num), lam, device=x.device)
    draws = torch.bernoulli(probs.unsqueeze(-1).expand(-1, -1, n_labels - 1)).sum(-1)
    mask = levels[draws.long().clamp(max=n_labels - 1)].view(b, 1, block_num, block_num)

    mask = F.interpolate(mask, size=(h, w), mode="nearest")
    out = mask * x + (1 - mask) * x[idx]
    ratio = mask.reshape(b, -1).mean(-1).view(-1, 1)

    y1 = one_hot(y, num_classes)
    return out, ratio * y1 + (1 - ratio) * y1[idx]


def make_regime(name: str, num_classes: int, alpha: float = 1.0,
                puzzle_mixer=None, **kw):
    """
    Devolve fn(model, x, y) -> (x_mix, y_soft), uniforme entre regimes.
    `puzzle_mixer` é uma instância de PuzzleMixOfficial (só para 'puzzlemix').
    """
    name = name.lower()
    if name == "puzzlemix":
        if puzzle_mixer is None:
            raise ValueError("regime 'puzzlemix' exige puzzle_mixer=PuzzleMixOfficial(...)")
        return lambda model, x, y: puzzle_mixer(model, x, y)

    fns = {"baseline": baseline, "input": input_mixup,
           "cutmix": cutmix, "blockmix_random": blockmix_random}
    if name not in fns:
        raise ValueError(f"regime desconhecido: {name!r}. Use {list(fns) + ['puzzlemix']}")
    fn = fns[name]
    return lambda model, x, y: fn(x, y, num_classes, alpha=alpha, **kw)


REGIMES = ["baseline", "cutmix", "blockmix_random", "puzzlemix"]
