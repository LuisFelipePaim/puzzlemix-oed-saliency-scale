"""
architectures.py — registro de arquiteturas para a matriz do SIBGRAPI.

Substitui o `build_model()` do notebook. Mantém a decisão anterior para a
VGG16-BN (cabeça trocada por GAP + 1 FC, porque as duas FC de 4096 originais
somam ~120M parâmetros e estouram 6GB de VRAM com AdamW) e acrescenta as
candidatas a segunda arquitetura.

Também resolve o ponto de hook do Grad-CAM de forma ESTÁVEL À DECOMPOSIÇÃO:
Tucker/TT trocam uma conv k×k por três camadas (1×1 -> k×k -> 1×1), o que
desloca índices dentro de `model.features`. Por isso nunca se referencia
`features[i]` por índice; resolve-se por travessia ou pelo container inteiro.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

# Nomes aceitos em ARCH_NAME / loop da matriz.
ARCHITECTURES = ["vgg16_bn", "densenet121", "efficientnet_b0"]

# Ativações usadas como âncora de CAM (EfficientNet usa SiLU, não ReLU).
_ACTIVATIONS = (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Hardswish, nn.GELU)


class _GAPHead(nn.Module):
    """Global average pooling + 1 FC. ~2k parâmetros no lugar de ~120M."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))


class VGGGap(nn.Module):
    """VGG16-BN com cabeça GAP. `features` preservado (alvo da decomposição)."""

    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        weights = tvm.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        base = tvm.vgg16_bn(weights=weights)
        self.features = base.features
        self.head = _GAPHead(512, num_classes, dropout)

    def forward(self, x):
        return self.head(self.features(x))


def build_model(
    arch_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """Constrói a arquitetura pelo nome. Todas expõem `.features`."""
    arch_name = arch_name.lower()

    if arch_name == "vgg16_bn":
        return VGGGap(num_classes, pretrained, dropout)

    if arch_name == "densenet121":
        weights = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if arch_name == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        if dropout > 0:
            model.classifier[0] = nn.Dropout(dropout)
        return model

    raise ValueError(f"arquitetura desconhecida: {arch_name!r}. Use um de {ARCHITECTURES}")


# --------------------------------------------------------------------------
# Ponto de hook do Grad-CAM
# --------------------------------------------------------------------------

def feature_extractor(model: nn.Module) -> nn.Module:
    """Container convolucional da arquitetura (alvo da decomposição)."""
    if hasattr(model, "features"):
        return model.features
    raise AttributeError("modelo sem atributo `features`; informe o alvo manualmente")


def cam_target_layer(model: nn.Module, mode: str = "last_act") -> nn.Module:
    """
    Resolve o módulo cuja saída ancora o Grad-CAM.

    mode="last_act"  -> última ativação dentro de `features`.
                        Preserva resolução (14×14 em 224px na VGG16, contra 7×7
                        depois do maxpool final) e é reencontrada por travessia
                        a cada chamada, então sobrevive à decomposição.
    mode="container" -> saída de `features` inteiro. Mais grosseiro, mas
                        imune a qualquer reorganização interna.

    Sempre resolver DEPOIS de decompor; nunca guardar o módulo entre modelos.
    """
    feats = feature_extractor(model)
    if mode == "container":
        return feats
    if mode != "last_act":
        raise ValueError("mode deve ser 'last_act' ou 'container'")

    # Percorrer feats.modules() e pegar a última ativação NÃO funciona: em
    # DenseNet a última ReLU registrada está dentro do último denselayer e sua
    # saída são só os 32 canais de crescimento, não a saída concatenada do bloco.
    # Regra correta: último filho de primeiro nível que não seja pooling; dentro
    # dele, a última ativação, se houver.
    skip = (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.Dropout, nn.Identity)
    block = None
    for child in reversed(list(feats.children())):
        if not isinstance(child, skip):
            block = child
            break
    if block is None:
        raise RuntimeError("nenhuma camada âncora encontrada em `features`")

    if isinstance(block, _ACTIVATIONS):
        return block
    inner = [m for m in block.modules() if isinstance(m, _ACTIVATIONS)]
    return inner[-1] if inner else block


def cam_spatial_size(model: nn.Module, input_size: int, mode: str = "last_act",
                     device: str = "cpu") -> tuple:
    """Resolução do mapa de CAM. Use para checar o alerta de 3×3 no PCam."""
    model = model.to(device).eval()
    layer = cam_target_layer(model, mode)
    out = {}
    h = layer.register_forward_hook(lambda m, i, o: out.update(shape=tuple(o.shape[1:])))
    with torch.no_grad():
        model(torch.zeros(1, 3, input_size, input_size, device=device))
    h.remove()
    return out["shape"]  # (C, H, W) — confira C: deve ser a largura do bloco final


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    conv = sum(p.numel() for m in model.modules() if isinstance(m, nn.Conv2d)
               for p in m.parameters())
    return {"total": total, "conv": conv, "head": total - conv}
