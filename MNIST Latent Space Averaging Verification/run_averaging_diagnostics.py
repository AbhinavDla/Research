"""Generate per-space UMAPs and voting diagnostics for the MNIST ensembles."""

import argparse
import csv
import importlib.util
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from umap import UMAP


ROOT = Path(__file__).resolve().parent
METHODS = {
    "autoencoder": ("Unsupervised Autoencoder", "unsupervised_latent_ensemble.py"),
    "simclr": ("Semi-Supervised SimCLR", "semi_supervised_latent.py"),
}


def load_module(folder, filename):
    path = ROOT / folder / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load model script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_space(path, features, labels, predictions, indices, seed, title):
    coords = UMAP(n_neighbors=min(15, len(indices) - 1), random_state=seed,
                  n_jobs=1).fit_transform(features[indices])
    correct = predictions[indices] == labels[indices]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for digit in range(10):
        mask = labels[indices] == digit
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.55,
                        color=colors(digit), label=f"{digit} ({mask.sum()})", rasterized=True)
    axes[0].legend(title="True digit", ncol=2, markerscale=2)
    for mask, label, color in [(correct, "Correct", "#a9a9a9"),
                               (~correct, "Incorrect", "#d62728")]:
        axes[1].scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6,
                        color=color, label=f"{label} ({mask.sum()})", rasterized=True)
    axes[1].legend(markerscale=2)
    axes[0].set_title("True digit")
    axes[1].set_title("Individual space: kNN predictions")
    for axis in axes:
        axis.set(xlabel="UMAP 1", ylabel="UMAP 2")
    fig.suptitle(f"{title} | {len(indices)} held-out test images")
    fig.savefig(path / "umap.png", dpi=150)
    plt.close(fig)
    write_rows(path / "umap.csv", [
        {"test_index": int(index), "digit": int(labels[index]),
         "prediction": int(predictions[index]), "umap_1": float(point[0]),
         "umap_2": float(point[1])}
        for index, point in zip(indices, coords)
    ])


def run(method, args):
    folder, filename = METHODS[method]
    model_module = load_module(folder, filename)
    setattr(model_module, "NUM_MODELS", args.spaces)
    setattr(model_module, "LATENT_DIM", args.dimensions)
    if args.epochs is not None:
        setattr(model_module, "EPOCHS", args.epochs)
    if args.spaces < 1 or args.dimensions < 1 or model_module.EPOCHS < 1:
        raise ValueError("spaces, dimensions, and epochs must be positive")

    spaces_folder = f"SimCLR {args.spaces} Spaces" if method == "simclr" else f"{args.spaces} Spaces"
    base = args.output_root if args.output_root else ROOT / folder / "Iterations"
    output = (base / spaces_folder /
              f"{args.dimensions} Latent Dimensions" / "Diagnostics" /
              datetime.now().strftime("run_%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True)
    started = time.time()
    data_root = ROOT.parent / "data"
    train = datasets.MNIST(data_root, train=True, download=False)
    test = datasets.MNIST(data_root, train=False, download=False)
    train_x = train.data.float().div(255).unsqueeze(1)
    test_x = test.data.float().div(255).unsqueeze(1)[:args.test_limit]
    test_y = test.targets[:len(test_x)].numpy()
    if args.umap_sample < 3 or len(test_x) < 3:
        raise ValueError("At least three test images and UMAP samples are required")
    indices = np.sort(np.random.RandomState(42).choice(
        len(test_x), min(args.umap_sample, len(test_x)), replace=False))
    config = {"method": method, "spaces": args.spaces, "latent_dimensions": args.dimensions,
              "epochs": model_module.EPOCHS, "batch_size": model_module.BATCH_SIZE,
              "k": model_module.K_NEIGHBORS, "train_images": len(train_x),
              "test_images": len(test_x), "umap_points_per_space": len(indices),
              "seeds": list(range(1000, 1000 + args.spaces)),
              "device": str(model_module.DEVICE), "status": "running"}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\n{method}: output {output} | device {model_module.DEVICE}", flush=True)

    train_loader = reference_loader = test_loader = None
    reference_indices = None
    if method == "autoencoder":
        train_data = TensorDataset(train_x, train.targets)
        test_loader = DataLoader(TensorDataset(test_x, test.targets[:len(test_x)]),
                                 batch_size=model_module.BATCH_SIZE, shuffle=False)
        train_loader = DataLoader(train_data, batch_size=model_module.BATCH_SIZE, shuffle=True)
        reference_loader = DataLoader(train_data, batch_size=model_module.BATCH_SIZE, shuffle=False)
    else:
        reference_indices = model_module.choose_labeled_indices(train.targets, 50)

    all_predictions = []
    metrics = []
    for number in range(args.spaces):
        seed = 1000 + number
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"\n{method}: space {number + 1}/{args.spaces}, seed {seed}", flush=True)
        if method == "autoencoder":
            if train_loader is None or reference_loader is None or test_loader is None:
                raise RuntimeError("Autoencoder data loaders were not initialized")
            model = model_module.Autoencoder(args.dimensions).to(model_module.DEVICE)
            model_module.train_autoencoder(model, train_loader)
            references, ref_labels = model_module.extract_latent_features(model, reference_loader)
            features, _ = model_module.extract_latent_features(model, test_loader)
            predictions = model_module.batched_knn_predictions(
                features, references, ref_labels, model_module.K_NEIGHBORS)
        else:
            if reference_indices is None:
                raise RuntimeError("SimCLR reference indices were not initialized")
            subset = torch.randperm(len(train_x))[:model_module.TRAIN_SUBSET]
            model = model_module.SimCLRModel(args.dimensions).to(model_module.DEVICE)
            model_module.train_simclr(model, train_x[subset])
            references = model_module.encode(model, train_x[reference_indices])
            features_tensor = model_module.encode(model, test_x)
            predictions = model_module.knn_predict(
                features_tensor, references, train.targets[reference_indices],
                model_module.K_NEIGHBORS).numpy()
            features = features_tensor.cpu().numpy()
        all_predictions.append(predictions)
        accuracy = float(np.mean(predictions == test_y))
        metrics.append({"space": number + 1, "seed": seed, "accuracy": accuracy})
        space_dir = output / f"space_{number + 1:02d}"
        space_dir.mkdir()
        plot_space(space_dir, features, test_y, predictions, indices, 42,
                   f"{method} | space {number + 1}, seed {seed}")
        print(f"  accuracy {accuracy:.4%}; elapsed {(time.time() - started) / 60:.1f} min",
              flush=True)

    stacked = np.stack(all_predictions, axis=1)
    if method == "autoencoder":
        ensemble = np.array([Counter(row).most_common(1)[0][0] for row in stacked])
    else:
        ensemble = np.array([np.bincount(row, minlength=10).argmax() for row in stacked])
    counts = np.stack([(stacked == digit).sum(axis=1) for digit in range(10)], axis=1)
    winning_votes = counts.max(axis=1)
    tied = (counts == winning_votes[:, None]).sum(axis=1) > 1
    write_rows(output / "spaces.csv", metrics)
    write_rows(output / "predictions.csv", [
        {"test_index": i, "digit": int(label), "ensemble": int(ensemble[i]),
         "winning_votes": int(winning_votes[i]), "tie": bool(tied[i]),
         **{f"space_{j + 1:02d}": int(value) for j, value in enumerate(stacked[i])}}
        for i, label in enumerate(test_y)
    ])
    per_digit = []
    for digit in range(10):
        mask = test_y == digit
        if mask.any():
            per_digit.append({"digit": digit, "test_images": int(mask.sum()),
                              "ensemble_accuracy": float(np.mean(ensemble[mask] == digit)),
                              "mean_space_accuracy": float(np.mean(stacked[mask] == digit))})
    write_rows(output / "digits.csv", per_digit)
    mean = float(np.mean([row["accuracy"] for row in metrics]))
    std = float(np.std([row["accuracy"] for row in metrics]))
    ens = float(np.mean(ensemble == test_y))
    correct_by_space = stacked == test_y[:, None]
    all_correct = int(correct_by_space.all(axis=1).sum())
    all_wrong = int((~correct_by_space).all(axis=1).sum())
    ensemble_correct = ensemble == test_y
    first_correct = correct_by_space[:, 0]
    ensemble_only = int((ensemble_correct & ~first_correct).sum())
    first_only = int((~ensemble_correct & first_correct).sum())
    reference = ("all 60,000 labeled training images" if method == "autoencoder"
                 else "50 labeled training images per digit (500 total)")
    tie_rule = ("the earliest space's tied prediction" if method == "autoencoder"
                else "the smallest tied digit")
    summary = [
        f"# {method} latent-space voting diagnostics",
        "",
        f"{args.spaces} independently trained spaces, {args.dimensions} dimensions each; "
        f"{model_module.EPOCHS} training epochs per space. Seeds: "
        f"{', '.join(map(str, config['seeds']))}.",
        f"Training: {len(train_x):,} MNIST images. Evaluation: {len(test_y):,} held-out test images. "
        f"kNN reference: {reference}; k={model_module.K_NEIGHBORS}.",
        "",
        "## What the experiment means",
        "",
        "A latent space is the collection of numerical representations learned by one neural network. "
        "The autoencoder learns to reconstruct images without digit labels; SimCLR learns by bringing "
        "two randomly augmented views of an image together. Labels are used afterward for the reference "
        "set and evaluation, not for training either encoder.",
        "For each test image, k-nearest neighbors (kNN) finds the closest labeled reference images in "
        "that model's latent space and votes on their digits. The ensemble then votes across the spaces' "
        f"predicted digits. Tied ensemble votes select {tie_rule}. This is prediction voting, not an "
        "arithmetic average of latent coordinates: independently trained spaces are not aligned.",
        "A model is a set of learned numerical weights; a seed initializes its random training choices. "
        "An epoch is one pass through the training images. The autoencoder's encoder compresses an image "
        "into a vector of numbers, and its decoder attempts to reconstruct the image from that vector. "
        "The training objective is reconstruction error, not digit classification.",
        "The labeled reference set supplies example answers for kNN after training. The test set is "
        "separate from that reference set and is used to measure performance. The nearest neighbors "
        "are selected in the full latent space using Euclidean distance; the most common neighbor "
        "digit is the prediction. Neighbor-vote ties select the smallest digit.",
        "",
        "## Reading the maps",
        "",
        f"Each space has a UMAP plot of the same {len(indices):,} held-out images, with true digits "
        "on the left and that space's correct/incorrect predictions on the right. UMAP compresses "
        f"{args.dimensions}-dimensional representations into two dimensions for visualization; the actual kNN uses "
        "the original representations. Each UMAP is fitted separately, so position, rotation and "
        "distances across different plots are not comparable. Clusters are suggestive, not proof.",
        "",
        "## Results",
        "",
        f"- Mean individual accuracy: {mean:.2%}; population standard deviation: {std:.2%}.",
        f"- Ensemble accuracy: {ens:.2%}; difference from mean individual: {(ens - mean) * 100:+.2f} percentage points.",
        f"- Images with tied highest ensemble vote: {int(tied.sum()):,} of {len(test_y):,}.",
        f"- All spaces were correct on {all_correct:,} images, all were wrong on {all_wrong:,}, "
        f"and they had mixed correctness on {len(test_y) - all_correct - all_wrong:,}.",
        f"- Compared with space 1 specifically, the ensemble corrected {ensemble_only:,} images "
        f"that space 1 missed, but missed {first_only:,} images that space 1 got right.",
        "- Accuracy is the fraction of test images classified correctly. Standard deviation describes "
        "variation between these spaces, not a confidence interval.",
        "- A percentage-point gain is a direct subtraction of percentages. It does not imply the "
        "same gain on a different dataset, and the ten spaces are not ten independent test sets.",
        "",
        "| Space | Seed | Accuracy |",
        "|---:|---:|---:|",
    ]
    summary.extend(f"| {r['space']} | {r['seed']} | {r['accuracy']:.2%} |" for r in metrics)
    summary.extend(["", "## Per-digit results", "", "| Digit | Test images | Mean space accuracy | Ensemble accuracy |",
                    "|---:|---:|---:|---:|"])
    summary.extend(f"| {r['digit']} | {r['test_images']} | {r['mean_space_accuracy']:.2%} | "
                   f"{r['ensemble_accuracy']:.2%} |" for r in per_digit)
    summary.extend(["", "## Files", "",
                    "- `space_NN/umap.png` and `umap.csv`: map and coordinates for that space.",
                    "- `spaces.csv`: individual accuracies and seeds.",
                    "- `predictions.csv`: every test image's individual and ensemble predictions, "
                    "winning vote count, and tie flag.",
                    "- `digits.csv`: per-digit accuracy; `config.json`: run settings.",
                    ""])
    if method == "autoencoder":
        (output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    config.update(status="completed", elapsed_seconds=round(time.time() - started, 2))
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Completed {method}: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("autoencoder", "simclr", "both"), default="both")
    parser.add_argument("--spaces", type=int, default=10)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--umap-sample", type=int, default=10000)
    parser.add_argument("--test-limit", type=int, default=10000)
    parser.add_argument("--output-root", type=Path, help="Override output location (e.g. for test runs)")
    args = parser.parse_args()
    for method in METHODS if args.method == "both" else (args.method,):
        run(method, args)


if __name__ == "__main__":
    main()
