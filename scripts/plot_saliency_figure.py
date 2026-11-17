"""Generate a real-vs-permuted saliency figure for the first validation fold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from architectures import build_model
from puzzlemix_official import PuzzleMixOfficial

IMAGE_SIZE = 224
GRID_SIZE = 4
NUM_FOLDS = 5
TEST_SIZE = 0.2
DISEASE_GRADES = {"mild", "moderate", "severe"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Training split/checkpoint seed")
    parser.add_argument("--permutation-seed", type=int, default=314159)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "Originais")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "sibgrapi_ckpt_clinical_composite",
    )
    parser.add_argument("--check-only", action="store_true", help="Load the checkpoint and test one ROI")
    return parser.parse_args()


def deterministic_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    crop = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
    ])
    model_input = transforms.Compose([
        crop,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return crop, model_input


def validation_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    cv_indices, _ = train_test_split(
        np.arange(len(labels)),
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=seed,
    )
    fold_splitter = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
    for fold_number, (_train_positions, val_positions) in enumerate(
        fold_splitter.split(np.zeros(len(cv_indices)), labels[cv_indices]), start=1
    ):
        if fold_number == 1:
            return cv_indices[val_positions]
    raise RuntimeError("The first validation fold was not generated")


def load_baseline(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {checkpoint_path}. No model was trained."
        )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    state = {key.replace("classifier.", "head.fc."): value for key, value in state.items()}

    model = build_model("vgg16_bn", num_classes=4, pretrained=False).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def compute_saliency(
    model: torch.nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
) -> torch.Tensor:
    probe = SimpleNamespace(clean_lam=0.0, num_classes=4)
    return PuzzleMixOfficial.saliency(probe, model, image, label)[0]


def top_mass(
    saliency: torch.Tensor, permutation_seed: int
) -> tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    pooled = F.avg_pool2d(
        saliency[None, None], kernel_size=IMAGE_SIZE // GRID_SIZE, stride=IMAGE_SIZE // GRID_SIZE
    )[0, 0]
    k = max(1, round(0.1 * GRID_SIZE**2))
    real_mass = pooled.flatten()
    real_mass = real_mass / real_mass.sum().clamp_min(1e-12)

    rng = np.random.default_rng(permutation_seed)
    shuffled_indices = rng.permutation(saliency.numel())
    shuffled = saliency.flatten()[torch.as_tensor(shuffled_indices, device=saliency.device)]
    shuffled = shuffled.reshape_as(saliency)
    permuted_pool = F.avg_pool2d(
        shuffled[None, None],
        kernel_size=IMAGE_SIZE // GRID_SIZE,
        stride=IMAGE_SIZE // GRID_SIZE,
    )[0, 0]
    permuted_mass = permuted_pool.flatten()
    permuted_mass = permuted_mass / permuted_mass.sum().clamp_min(1e-12)

    real_top = torch.topk(real_mass, k)
    permuted_top = torch.topk(permuted_mass, k)
    return (
        real_top.values.sum().item(),
        permuted_top.values.sum().item(),
        real_top.indices,
        permuted_top.indices,
        shuffled,
    )


def analyze_fold(
    dataset: datasets.ImageFolder,
    indices: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    permutation_seed: int,
    limit: int | None = None,
) -> list[dict]:
    results = []
    selected_indices = indices if limit is None else indices[:limit]
    for position, dataset_index in enumerate(selected_indices):
        path, label = dataset.samples[int(dataset_index)]
        image, _ = dataset[int(dataset_index)]
        image = image.unsqueeze(0).to(device)
        target = torch.tensor([label], dtype=torch.long, device=device)
        saliency = compute_saliency(model, image, target)
        real, permuted, real_top, permuted_top, shuffled = top_mass(
            saliency, permutation_seed + position
        )
        results.append({
            "path": Path(path),
            "grade": Path(path).parent.name,
            "label": label,
            "image": image[0].detach().cpu(),
            "saliency": saliency.detach().cpu(),
            "real": real,
            "permuted": permuted,
            "ratio": real / permuted if permuted else float("inf"),
            "real_top": real_top.detach().cpu().numpy(),
            "permuted_top": permuted_top.detach().cpu().numpy(),
            "permuted_saliency": shuffled.detach().cpu().numpy(),
        })
    return results


def draw_grid(ax: plt.Axes, top_indices: np.ndarray) -> None:
    block_size = IMAGE_SIZE // GRID_SIZE
    for boundary in range(1, GRID_SIZE):
        coordinate = boundary * block_size
        ax.axhline(coordinate, color="white", linewidth=1.5, alpha=0.95)
        ax.axvline(coordinate, color="white", linewidth=1.5, alpha=0.95)
    for index in top_indices:
        row, column = divmod(int(index), GRID_SIZE)
        ax.add_patch(patches.Rectangle(
            (column * block_size, row * block_size),
            block_size,
            block_size,
            fill=False,
            edgecolor="#2C752C",
            linewidth=3.0,
        ))
    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(IMAGE_SIZE, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def save_figure(candidate: dict, crop_transform: transforms.Compose) -> None:
    pil_image = crop_transform(Image.open(candidate["path"]).convert("RGB"))
    rgb_image = np.asarray(pil_image)
    saliency = candidate["saliency"].numpy()
    max_value = max(float(saliency.max()), 1e-12)
    k = max(1, round(0.1 * GRID_SIZE**2))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 15})
    figure, axes = plt.subplots(1, 3, figsize=(14.17, 5.05), facecolor="white")
    axes[0].imshow(rgb_image)
    axes[0].set_title(f"(a) H&E | {candidate['grade']}", fontsize=17, pad=12)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(saliency, cmap="magma", vmin=0, vmax=max_value, interpolation="nearest")
    axes[1].set_title(f"(b) Real | C top-{k} = {candidate['real']:.3f}", fontsize=17, pad=12)
    draw_grid(axes[1], candidate["real_top"])

    axes[2].imshow(
        candidate["permuted_saliency"],
        cmap="magma",
        vmin=0,
        vmax=max_value,
        interpolation="nearest",
    )
    axes[2].set_title(
        f"(c) Permutada | C top-{k} = {candidate['permuted']:.3f}",
        fontsize=17,
        pad=12,
    )
    draw_grid(axes[2], candidate["permuted_top"])

    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.86, wspace=0.10)
    output_dir = PROJECT_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(
            output_dir / f"saliencia_real_vs_permutada.{extension}",
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    crop_transform, model_transform = deterministic_transforms()
    dataset = datasets.ImageFolder(args.data_dir, transform=model_transform)
    labels = np.asarray(dataset.targets)
    val_indices = validation_indices(labels, args.seed)
    checkpoint = args.checkpoint_dir / f"vgg16_bn_baseline_seed{args.seed}_fold1_best.pth"

    model = load_baseline(checkpoint, device)
    results = analyze_fold(
        dataset,
        val_indices,
        model,
        device,
        args.permutation_seed,
        limit=1 if args.check_only else None,
    )
    if args.check_only:
        print(
            f"Checkpoint loaded; first fold-1 ROI saliency OK: {results[0]['path']} "
            f"(real={results[0]['real']:.3f}, permuted={results[0]['permuted']:.3f})"
        )
        return

    mean_real = float(np.mean([item["real"] for item in results]))
    mean_permuted = float(np.mean([item["permuted"] for item in results]))
    mean_ratio = mean_real / mean_permuted if mean_permuted else float("inf")
    preferred = [item for item in results if item["grade"].lower() in DISEASE_GRADES]
    ranked = sorted(preferred or results, key=lambda item: item["ratio"], reverse=True)
    selected = ranked[0]

    print(f"seed={args.seed}, fold=1, validation ROIs={len(results)}")
    print(
        f"Fold mean at 4x4: real={mean_real:.3f}, permuted={mean_permuted:.3f}, "
        f"ratio={mean_ratio:.3f} (Table II reference: 0.209 / 0.128)"
    )
    print("Top 3 candidates (dysplasia grades preferred):")
    for rank, item in enumerate(ranked[:3], start=1):
        print(
            f"{rank}. {item['path'].relative_to(PROJECT_ROOT)} | {item['grade']} | "
            f"real={item['real']:.3f} | permuted={item['permuted']:.3f} | "
            f"ratio={item['ratio']:.3f}"
        )

    save_figure(selected, crop_transform)
    print(
        f"Selected: {selected['path'].relative_to(PROJECT_ROOT)} | {selected['grade']} | "
        f"real={selected['real']:.3f} | permuted={selected['permuted']:.3f} | "
        f"ratio={selected['ratio']:.3f}"
    )
    print("Saved figures/saliencia_real_vs_permutada.png and .svg")
    print(
        "Re-generate: python scripts/plot_saliency_figure.py "
        f"--seed {args.seed} --permutation-seed {args.permutation_seed}"
    )


if __name__ == "__main__":
    main()
