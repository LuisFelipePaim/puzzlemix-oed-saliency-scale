# Puzzle Mix na gradação de displasia epitelial oral: escala da máscara × escala da saliência

Código e resultados do trabalho apresentado no Workshop of Undergraduate Works (WUW) do SIBGRAPI 2026.
O estudo caracteriza **quando** o Puzzle Mix se aplica à gradação de displasia epitelial oral (OED) —
em especial o descasamento entre a escala da grade de máscara e a escala em que a saliência tem estrutura —
em vez de propor um ganho de desempenho.

Pesquisa de iniciação científica na FACOM/UFU.

## Configuração experimental

- **Dados:** OralEpitheliumDB, 4 classes ordinais (`healthy`, `mild`, `moderate`, `severe`). Não incluído.
- **Modelo:** VGG16-BN pré-treinada no ImageNet, com cabeça trocada por *global average pooling* + 1 FC (cabe em 6 GB de VRAM).
- **Treino:** AdamW, CosineAnnealingLR, 50 épocas, batch efetivo 32 (16 × 2 de acumulação), AMP.
- **Avaliação:** holdout de teste de 20%, validação cruzada de 5 folds no restante e ensemble dos 5 modelos no teste; checkpoint pelo critério clínico composto.
- **Métrica principal:** QWK (kappa ponderado quadrático), além de AUC, F1, acurácia e erros perigosos (≥ 2 graus).
- **Hardware:** NVIDIA RTX 3050 Laptop (6 GB).

## Resultados

Tabelas geradas a partir dos CSVs em `results/` (média ± desvio-padrão entre seeds).

### Primeira matriz — `results/sibgrapi_logs.csv`

Rótulos duros. O regime `puzzlemix` aqui é a implementação **simplificada** da Seção 3 do notebook, não o Puzzle Mix oficial.

| Regime | Seeds | QWK (teste) | AUC | F1 | Acurácia |
|---|---|---|---|---|---|
| `baseline` | 5 | 0.9281 ± 0.0220 | 0.9706 | 0.8596 | 0.8587 |
| `cutmix` | 5 | 0.9252 ± 0.0117 | 0.9751 | 0.8719 | 0.8717 |
| `puzzlemix` | 5 | 0.9136 ± 0.0207 | 0.9685 | 0.8501 | 0.8500 |

### Puzzle Mix oficial × controles aleatórios por grade — `results/sibgrapi_ablacao_block_nums.csv`

Rótulos suaves. `puzzlemix_*` usa o `mixup_graph` do repositório oficial; `random_*` mistura blocos sorteados
nas mesmas grades, sem saliência, para isolar a contribuição da saliência.

| Braço | Grades (`block_num`) | Seeds | QWK (teste) | AUC | Taxa de colapso da máscara |
|---|---|---|---|---|---|
| `random_coarse` | [2, 4] | 5 | 0.9169 ± 0.0213 | 0.9682 | — |
| `random_full` | [2, 4, 8, 16] | 5 | 0.9114 ± 0.0166 | 0.9674 | — |
| `random_fine` | [8, 16] | 5 | 0.9048 ± 0.0268 | 0.9655 | — |
| `puzzlemix_full` | [2, 4, 8, 16] | 5 | 0.8922 ± 0.0399 | 0.9522 | 0.334 |
| `puzzlemix_fine` | [8, 16] | 2 | 0.9102 ± 0.0450 | 0.9517 | 0.187 |

Nenhuma comparação pareada atinge p < 0,05 com 5 seeds; os testes são exploratórios (ver `src/stats_protocol.py`).

## Estrutura

```
src/
  architectures.py       # build_model, camada-alvo do Grad-CAM
  puzzlemix_official.py  # adaptador do snu-mllab/PuzzleMix + telemetria de colapso da máscara
  regimes.py             # baseline, cutmix, puzzlemix e controle blockmix_random (alvo suave)
  stats_protocol.py      # métrica primária pré-registrada, Wilcoxon/Friedman, correção de Holm
  xai_metrics.py         # Grad-CAM, deriva de explicação, ROAD
  timing.py              # cronometragem por estágio (sonda de custo)
notebooks/
  pipeline_sibgrapi.ipynb  # matriz completa: treino, CV + ensemble, resumo estatístico, ablação de block_num
scripts/
  run_ablacao_wsl.py       # roda a ablação de block_num fora do Jupyter (Linux/WSL, onde o gco compila)
  plot_saliency_figure.py  # figura saliência real × permutada
tests/                     # testes com dados sintéticos (não precisam do dataset)
results/                   # CSVs das execuções
```

## Como reproduzir

```bash
git clone https://github.com/<usuario>/puzzlemix-oed-saliency-scale.git
cd puzzlemix-oed-saliency-scale
pip install -r requirements.txt

# Repositório oficial do Puzzle Mix (commit usado: e2dbf3a), na raiz do projeto
git clone https://github.com/snu-mllab/PuzzleMix.git
git -C PuzzleMix checkout e2dbf3a
pip install gco-wrapper   # necessário só para os braços puzzlemix_*; em Windows, use WSL
```

Coloque as imagens do OralEpitheliumDB em `Originais/<classe>/` (uma pasta por classe) e abra
`notebooks/pipeline_sibgrapi.ipynb`. A primeira célula ajusta os caminhos para a raiz do repositório;
pesos e telemetria são gravados em `sibgrapi_ckpt_clinical_composite/` (ignorado pelo Git).

Para a ablação de `block_num` em Linux/WSL:

```bash
python scripts/run_ablacao_wsl.py --smoke   # teste rápido, saída descartável
python scripts/run_ablacao_wsl.py           # execução completa, com retomada pelo CSV
```

Testes sem dataset:

```bash
python tests/test_architectures_xai_timing.py
PUZZLEMIX_REPO=./PuzzleMix python tests/test_puzzlemix_regimes_stats.py
```

## Referências

- Kim, Choo e Song. *Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup.* ICML 2020. Código: [snu-mllab/PuzzleMix](https://github.com/snu-mllab/PuzzleMix) (MIT).
- Yun et al. *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.* ICCV 2019.
