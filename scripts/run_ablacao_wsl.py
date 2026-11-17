"""
run_ablacao_wsl.py — roda main_ablacao() (com os bracos puzzlemix_* de
verdade, gco disponivel) a partir do venv Linux do WSL.

Extrai as MESMAS celulas do pipeline_sibgrapi.ipynb (nao reescreve nada),
so' que executadas pelo Python do WSL em vez do kernel Jupyter Windows.
Escreve no MESMO CSV que o notebook usa (sibgrapi_ablacao_block_nums.csv)
- o resume por (Architecture, Arm, Seed) funciona igual dos dois lados.

Uso:
    /opt/sibgrapi-venv/bin/python scripts/run_ablacao_wsl.py         # roda tudo (default)
    /opt/sibgrapi-venv/bin/python scripts/run_ablacao_wsl.py --smoke # smoke test rapido/descartavel
"""
import argparse
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz do repositorio
NB_PATH = os.path.join(PROJECT_DIR, "notebooks", "pipeline_sibgrapi.ipynb")

CELL_ORDER = [
    "f06cecf8",  # 1 imports
    "671a129e",  # 2 config
    "ee5267b6",  # 3 device
    "535d3033",  # 4 seed_everything
    "d61bb79b",  # 5 loaders/build_model/scheduler/ordinal_ranks/gradcam
    "4171bead",  # 6 clinical_metrics/checkpoint_selection_score
    "0741e477",  # 7 plot_fold_telemetry
    "28079a18",  # 16 ablacao_block_nums.py (build_arms corrigido, collapse_report, ARMS)
    "2dc8613e",  # 17 run_config_ablacao / main_ablacao
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: seed descartavel 999, 1 epoca, 2 folds, saida temporaria")
    args = ap.parse_args()

    os.chdir(PROJECT_DIR)
    sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

    with open(NB_PATH, encoding="utf-8") as f:
        nb = json.load(f)
    cells_by_id = {c["id"]: "".join(c["source"]) for c in nb["cells"]}

    ns = {"__name__": "__wsl_ablacao__"}
    for cid in CELL_ORDER:
        exec(compile(cells_by_id[cid], cid, "exec"), ns)

    print(f"GCO_AVAILABLE={ns['GCO_AVAILABLE']}  POT_AVAILABLE={ns['POT_AVAILABLE']}  "
          f"device={ns['device']}")

    if args.smoke:
        out_dir = "/tmp/ablacao_wsl_smoke"
        os.makedirs(out_dir, exist_ok=True)
        ns["main_ablacao"](
            arch_name="vgg16_bn", seeds=[999], epochs=1, n_folds=2,
            csv_path=os.path.join(out_dir, "smoke.csv"), output_dir=out_dir,
        )
    else:
        ns["main_ablacao"]()  # usa EPOCHS/N_FOLDS/SEEDS_BY_ARCH globais, CSV real


if __name__ == "__main__":
    main()
