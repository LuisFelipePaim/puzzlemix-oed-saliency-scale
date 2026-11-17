"""
puzzlemix_official.py — adaptador para a implementação oficial (snu-mllab/PuzzleMix).

Motivo: um resultado negativo é sempre atacado por "vocês implementaram errado".
A auditoria de 12/08 mostrou que o ataque teria procedido. Usando o código dos
autores, esse flanco fecha.

Pré-requisitos:

    git clone https://github.com/snu-mllab/PuzzleMix.git
    pip install gco-wrapper          # o α-β swap (pyGCO)

Este módulo NÃO reimplementa nada: importa `mixup_graph` do repositório oficial e
só replica o call site do `main.py` deles (linhas ~356-392), que é onde a
saliência é calculada e passada como termo unário.

Diferenças em relação ao `puzzlemix_batch()` antigo do notebook, todas por
construção do código oficial:
  - saliência = sqrt(mean(grad², canais)), ou seja ‖·‖₂ (a versão antiga somava
    |grad|, uma L1)
  - máscara multi-label (n_labels=3 -> z ∈ {0, ½, 1}), não binária
  - suavidade espacial via graph-cut com β (rótulo) e γ (dados)
  - transporte que de fato move conteúdo entre posições
  - block_num sorteado por mini-batch em {2, 4, 8, 16}

Rótulos: o oficial treina com alvo SOFT e `bce_loss(softmax(out), alvo)`.
`soft_target_cross_entropy` abaixo reproduz isso; usar CrossEntropyLoss com
rótulo duro sobre saída misturada NÃO é equivalente.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _load_official(repo_path: str):
    p = str(Path(repo_path).expanduser().resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    try:
        from mixup import mixup_graph  # noqa: E402
    except ImportError as e:
        raise ImportError(
            f"não achei mixup.py em {p}. Clone o repositório oficial:\n"
            "  git clone https://github.com/snu-mllab/PuzzleMix.git"
        ) from e
    return mixup_graph


def one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(y, num_classes).float()


def soft_target_cross_entropy(logits: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
    """Perda do treino oficial: alvo soft, não rótulo duro."""
    return -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


class PuzzleMixOfficial:
    """
    Envelopa `mixup_graph` oficial. Defaults = hiperparâmetros do paper para
    CIFAR-100 (β=1.2, γ=0.5, η=0.2, ξ=t_eps=0.8, n_labels=3).

    CORRIGIDO (2026-08-13, achado rodando de ponta a ponta no WSL com gco
    instalado - ver ablacao_block_nums_integracao/): `t_size` do paper e'
    16, calibrado para CIFAR (32x32). O passo de transporte oficial
    (mixup.py:mask_transport) usa uma grade PROPRIA pro transporte,
    `t_block_num = width // t_size` quando `t_size < block_size`, e so'
    tem custo pre-computado (`cost_matrix_dict`) pra t_block_num em
    {2,4,8,16}. Pra CIFAR (32px), width//t_size cai nesse conjunto pros
    t_size do paper; pra imagens 224x224 (OED), width//t_size=14, que NAO
    esta' no dict -> KeyError('14') em TODO batch, sempre, independente de
    qual block_num foi sorteado pra mascara. Defalt trocado pra t_size=-1:
    isso faz `t_block_num = block_num` diretamente (ver mixup.py:311-313),
    ou seja, o transporte roda na MESMA grade da mascara - sempre em
    {2,4,8,16} (as unicas chaves do dict), pra qualquer tamanho de imagem.
    Se algum dia quiser bater exatamente os hiperparametros CIFAR do
    artigo, passe t_size=16 explicitamente (so' funciona pra imagens onde
    width for multiplo de 16 landing em {2,4,8,16} - nao e' o caso do OED).

    Mesma causa-raiz, segundo parametro: `neigh_size` do paper e' 4.
    mixup.py:232 faz `block_size = width // block_num` e depois
    `k = block_size // neigh_size` pra montar a penalidade de vizinhanca
    (neigh_penalty) - precisa de block_size MULTIPLO de neigh_size, senao
    o pooling produz uma resolucao espacial que nao bate com o tensor
    pw_x/pw_y esperado (RuntimeError de shape, ex.: "[16, 18, 18]" vs
    "[16, 15, 16]"). Pra CIFAR (32px): block_size em {16,8,4,2} - todos
    multiplos de 4. Pra 224px (OED): block_size em {112,56,28,14} - 14
    (do block_num=16, presente nos bracos 'full'/'fine') NAO e' multiplo
    de 4. Default trocado pra neigh_size=2, que divide {112,56,28,14}
    todos exatamente, pra qualquer block_num em {2,4,8,16}.

    Uso no loop de treino:

        mixer = PuzzleMixOfficial(REPO, mean, std, device, num_classes=4)
        ...
        x_mix, y_soft = mixer(model, x, y)
        loss = soft_target_cross_entropy(model(x_mix), y_soft)
    """

    def __init__(self, repo_path: str, mean: Sequence[float], std: Sequence[float],
                 device: str, num_classes: int,
                 alpha: float = 1.0, beta: float = 1.2, gamma: float = 0.5,
                 eta: float = 0.2, neigh_size: int = 2, n_labels: int = 3,
                 transport: bool = True, t_eps: float = 0.8, t_size: int = -1,
                 block_nums: Sequence[int] = (2, 4, 8, 16),
                 mp: Optional[int] = None, clean_lam: float = 0.0):
        self.mixup_graph = _load_official(repo_path)
        self.device = device
        self.num_classes = num_classes
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.alpha, self.beta, self.gamma, self.eta = alpha, beta, gamma, eta
        self.neigh_size, self.n_labels = neigh_size, n_labels
        self.transport, self.t_eps, self.t_size = transport, t_eps, t_size
        self.block_nums = list(block_nums)
        self.mp, self.clean_lam = mp, clean_lam
        self.reset_stats()

    def saliency(self, model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Termo unário, idêntico ao main.py oficial (linha 370):
            unary = sqrt(mean(grad², dim=canais))
        model.eval() durante o cálculo para não mexer nas estatísticas de BN.
        """
        was_training = model.training
        model.eval()
        xv = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(xv), y, reduction="none")
        scale = 1.0 if self.clean_lam == 0 else 2 * self.clean_lam / self.num_classes
        (scale * loss.mean()).backward()
        unary = torch.sqrt(torch.mean(xv.grad ** 2, dim=1))
        if was_training:
            model.train()
        model.zero_grad(set_to_none=True)
        return unary.detach()

    def __call__(self, model, x: torch.Tensor, y: torch.Tensor):
        """Retorna (x_misturado, alvo_soft). y pode ser índice de classe."""
        unary = self.saliency(model, x, y)
        indices = np.random.permutation(x.size(0))
        block_num = int(np.random.choice(self.block_nums))
        lam = float(np.random.beta(self.alpha, self.alpha)) if self.alpha > 0 else 1.0

        out, ratio = self.mixup_graph(
            x, unary, indices, block_num=block_num, alpha=lam,
            beta=self.beta, gamma=self.gamma, eta=self.eta,
            neigh_size=self.neigh_size, n_labels=self.n_labels,
            mean=self.mean, std=self.std,
            transport=self.transport, t_eps=self.t_eps, t_size=self.t_size,
            device=self.device, mp=self.mp,
        )

        self._record(unary, ratio, block_num)

        y1 = one_hot(y, self.num_classes)
        r = ratio.view(-1, 1).to(y1.device)
        y_soft = y1 * r + y1[indices] * (1 - r)
        return out, y_soft

    # -- telemetria ---------------------------------------------------------

    def _record(self, unary: torch.Tensor, ratio: torch.Tensor, block_num: int):
        """
        Registra colapso de máscara e concentração de saliência.

        POR QUE ISSO IMPORTA: com saliência plana o termo unário empata e a
        máscara é decidida só pelo prior η, saindo CONSTANTE — todos os blocos
        no mesmo nível. Nesse regime o Puzzle Mix degenera em input mixup com λ
        quantizado em {0, ½, 1}, e o transporte vira identidade. Verificado com
        saliência sintética: concentrada -> máscara segue a região saliente;
        plana -> ratio colapsa em 0, ½ ou 1 conforme λ.

        Uma taxa de colapso alta durante o treino é, portanto, evidência direta
        de que a premissa do método não vale no domínio — que é exatamente a
        hipótese do artigo. Reporte `collapse_rate` junto dos resultados.
        """
        r = ratio.detach().float().cpu()
        levels = torch.linspace(0, 1, self.n_labels)
        collapsed = torch.min((r.view(-1, 1) - levels.view(1, -1)).abs(), dim=1).values < 1e-6
        self._n += r.numel()
        self._n_collapsed += int(collapsed.sum())

        # concentração da saliência por bloco: fração da massa nos 10% maiores
        pooled = F.avg_pool2d(unary.unsqueeze(1), unary.shape[-1] // block_num).flatten(1)
        pooled = pooled / pooled.sum(1, keepdim=True).clamp_min(1e-12)
        k = max(1, int(round(0.1 * pooled.shape[1])))
        self._top10.append(pooled.topk(k, dim=1).values.sum(1).mean().item())
        self._top10_base.append(k / pooled.shape[1])   # piso uniforme desta grade

    def reset_stats(self):
        self._n = 0
        self._n_collapsed = 0
        self._top10: list = []
        self._top10_base: list = []

    @property
    def stats(self) -> dict:
        """
        collapse_rate: fração de amostras com máscara constante (Puzzle Mix
                       degenerado em input mixup). Perto de 1 = premissa falhou.
        saliency_top10: massa média de saliência nos k=round(0.1·n) blocos mais
                        quentes. ATENÇÃO: a referência uniforme é k/n, NÃO 0.10
                        — o arredondamento de k só dá 0.10 em grades finas.
                        Em 2×2 o piso é 0.25; em 4×4, 0.125; em 8×8, 0.094.
                        Compare sempre com `saliency_top10_uniform`, não com 0.10.
        """
        base = float(np.mean(self._top10_base)) if self._top10_base else float("nan")
        val = float(np.mean(self._top10)) if self._top10 else float("nan")
        return {
            "collapse_rate": self._n_collapsed / self._n if self._n else float("nan"),
            "saliency_top10": val,
            "saliency_top10_uniform": base,
            "saliency_concentration": val / base if base == base and base > 0 else float("nan"),
            "n_samples": self._n,
        }