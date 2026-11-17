<<<<<<< HEAD
<<<<<<< HEAD
"""Run five-space cleaning diagnostics for MNIST digits 0-6 with 7-9 injected."""

import argparse
import csv
import importlib
=======
=======
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
"""Run five-space cleaning diagnostics for MNIST digits 0-6 with 7-8 injected."""

import argparse
import csv
<<<<<<< HEAD
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
=======
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
import json
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import datasets


HERE = Path(__file__).resolve().parent
AUTOENCODER = HERE.parents[5]
NOTEBOOK = (AUTOENCODER / "Versions" / "KNN With Anomalies" /
            "anomaly_cleaning_train_0-6_inject_7-9_before_evaluation_fix.ipynb")
NORMAL_DIGITS = list(range(7))
<<<<<<< HEAD
<<<<<<< HEAD
ANOMALY_DIGITS = [7, 8, 9]
=======
ANOMALY_DIGITS = [7, 8]
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
=======
ANOMALY_DIGITS = [7, 8]
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e


def load_original_functions(namespace):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    markers = ("class ConvAutoencoder", "def encode(", "def evaluate(")
    for marker in markers:
        matches = ["".join(cell.get("source", [])) for cell in notebook["cells"]
                   if cell.get("cell_type") == "code" and marker in "".join(cell.get("source", []))]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one notebook cell containing {marker!r}; found {len(matches)}")
        exec(compile(matches[0], str(NOTEBOOK), "exec"), namespace)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_umap(latents, labels, indices, removed_mask, output_dir, title, sample_size, seed):
<<<<<<< HEAD
<<<<<<< HEAD
=======
    from umap.umap_ import UMAP

>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
=======
    from umap.umap_ import UMAP

>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
    rng = np.random.RandomState(seed)
    injected = np.isin(labels, ANOMALY_DIGITS)
    required = np.flatnonzero(removed_mask | injected)
    optional = np.flatnonzero(~(removed_mask | injected))
    optional_count = len(optional) if sample_size == 0 else min(len(optional), max(0, sample_size - len(required)))
    chosen = np.sort(np.concatenate([required, rng.choice(optional, optional_count, replace=False)]))
    if len(chosen) < 3:
        raise ValueError("UMAP needs at least three sampled training images")
<<<<<<< HEAD
<<<<<<< HEAD
    umap = importlib.import_module("umap")
    coordinates = np.asarray(umap.UMAP(n_neighbors=min(15, len(chosen) - 1), random_state=seed,
                                       n_jobs=1).fit_transform(latents[chosen].numpy()))
=======
    coordinates = np.asarray(UMAP(n_neighbors=min(15, len(chosen) - 1), random_state=seed,
                                  n_jobs=1).fit_transform(latents[chosen].numpy()))
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
=======
    coordinates = np.asarray(UMAP(n_neighbors=min(15, len(chosen) - 1), random_state=seed,
                                  n_jobs=1).fit_transform(latents[chosen].numpy()))
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
    rows = [{"mnist_train_index": int(indices[position]), "digit": int(labels[position]),
             "injected": bool(injected[position]),
             "removed_this_round": bool(removed_mask[position]),
             "umap_1": float(point[0]), "umap_2": float(point[1])}
            for position, point in zip(chosen, coordinates)]
    write_csv(output_dir.with_suffix(".csv"), rows)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for digit in NORMAL_DIGITS + ANOMALY_DIGITS:
        mask = labels[chosen] == digit
        axes[0].scatter(coordinates[mask, 0], coordinates[mask, 1], s=12 if digit in ANOMALY_DIGITS else 5,
                        alpha=0.8 if digit in ANOMALY_DIGITS else 0.45,
                        marker="x" if digit in ANOMALY_DIGITS else "o",
                        color=colors(digit), label=f"{digit} ({mask.sum()})", rasterized=True)
    axes[0].legend(title="True digit", markerscale=2, ncol=3, fontsize=8)
    selected_injected = injected[chosen]
    selected_removed = removed_mask[chosen]
    masks = (~selected_injected & ~selected_removed, ~selected_injected & selected_removed,
             selected_injected & ~selected_removed, selected_injected & selected_removed)
    for mask, label, color, size, marker in zip(
            masks, ("Retained normal", "Removed normal", "Retained injected", "Removed injected"),
            ("#b3b3b3", "#d62728", "#0072b2", "#cc79a7"), (5, 12, 20, 24), ("o", "o", "x", "x")):
        axes[1].scatter(coordinates[mask, 0], coordinates[mask, 1], s=size, alpha=0.8,
                        color=color, marker=marker, label=f"{label} ({mask.sum()})", rasterized=True)
    axes[1].legend(markerscale=2)
    axes[0].set_title("True digit (evaluation only)")
    axes[1].set_title("Ground truth and removal in this round")
    for axis in axes:
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
    fig.suptitle(title + f" | {len(chosen)} of {len(labels)} points")
    fig.savefig(output_dir.with_suffix(".png"), dpi=150)
    plt.close(fig)

<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True, constrained_layout=True)
    for axis, digit in zip(axes.flat, NORMAL_DIGITS + ANOMALY_DIGITS):
        mask = labels[chosen] == digit
        axis.scatter(coordinates[~mask, 0], coordinates[~mask, 1], s=2, color="#d9d9d9",
                     alpha=0.25, rasterized=True)
        axis.scatter(coordinates[mask, 0], coordinates[mask, 1], s=8, color=colors(digit),
                     alpha=0.8, rasterized=True)
        axis.set_title(f"Digit {digit}: {mask.sum()} shown" + (" (injected)" if digit in ANOMALY_DIGITS else ""))
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
    fig.suptitle(title + " | each digit highlighted in the same projection")
    fig.savefig(output_dir.with_name(output_dir.name + "_digits").with_suffix(".png"), dpi=150)
    plt.close(fig)


<<<<<<< HEAD
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
=======
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
def scalar_metrics(metrics):
    return {key: float(metrics[key]) for key in ("acc", "auc", "ap", "auc_latent", "auc_recon")}


def plot_removal_diagnostics(images, labels, injected, removed, knn, rec, scores, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for mask, label, color in [(~injected, "Normal", "#3979a3"),
                               (injected, "Injected", "#d55e00")]:
        axes[0].hist(scores[mask], bins=40, density=True, alpha=0.55, label=label, color=color)
        axes[1].scatter(knn[mask], rec[mask], s=3, alpha=0.25, label=label, color=color,
                        rasterized=True)
    axes[0].axvline(scores[removed].min(), color="black", linestyle="--", label="Removal cutoff")
    axes[0].set(xlabel="Combined rank score", ylabel="Density", title="Score distributions")
    axes[1].set(xlabel="Mean kNN distance", ylabel="Reconstruction error per ink",
                title="Cleaning signals")
    for axis in axes:
        axis.legend()
    fig.savefig(path / "score_diagnostics.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 8, figsize=(16, 5), squeeze=False, constrained_layout=True)
    for row, (mask, title) in enumerate([(removed & ~injected, "Removed normal"),
                                         (removed & injected, "Removed injected")]):
        positions = np.flatnonzero(mask)
        positions = positions[np.argsort(scores[positions])[::-1]][:8]
        for axis in axes[row]:
            axis.axis("off")
        for axis, position in zip(axes[row], positions):
            axis.imshow(images[position, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axis.set_title(f"Digit {labels[position]} | {scores[position]:.3f}", fontsize=8)
        axes[row, 0].set_ylabel(title)
        axes[row, 0].text(-0.15, 0.5, title, rotation=90, transform=axes[row, 0].transAxes,
                          va="center", ha="right", fontsize=9)
    fig.suptitle("Highest-scoring removed examples (score is a relative rank)")
    fig.savefig(path / "removed_examples.png", dpi=150)
    plt.close(fig)


def run_experiment(args, output_dir, namespace):
    namespace.update({"LATENT_DIM": 64, "EPOCHS": args.epochs, "BATCH_SIZE": 128,
                      "LEARNING_RATE": 1e-3, "SCORE_MODE": "combined"})
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    namespace.update({"DEVICE": device, "USE_AMP": device.type == "cuda"})
    load_original_functions(namespace)
    for base in (HERE, *HERE.parents):
        if (base / "data" / "MNIST" / "raw" / "train-images-idx3-ubyte").exists():
            data_root = base / "data"
            break
    else:
        raise FileNotFoundError("Could not find data/MNIST/raw in the parent directories")
    train_set = datasets.MNIST(data_root, train=True, download=False)
    test_set = datasets.MNIST(data_root, train=False, download=False)
    train_x = train_set.data.float().div(255).unsqueeze(1)
    train_y = train_set.targets.numpy()
    test_x = test_set.data.float().div(255).unsqueeze(1)
    test_y = test_set.targets.numpy()
    normal_train = np.flatnonzero(np.isin(train_y, NORMAL_DIGITS))
    anomaly_train = np.flatnonzero(np.isin(train_y, ANOMALY_DIGITS))
    normal_test = np.isin(test_y, NORMAL_DIGITS)
    anomaly_test = np.isin(test_y, ANOMALY_DIGITS)
    normal_test_tensor = torch.as_tensor(normal_test, dtype=torch.bool)
    anomaly_test_tensor = torch.as_tensor(anomaly_test, dtype=torch.bool)
    namespace["ctx"] = {"normal_x": test_x[normal_test_tensor], "normal_y": test_y[normal_test],
                        "det_x": torch.cat([test_x[normal_test_tensor], test_x[anomaly_test_tensor]]),
                        "det_is_anom": np.concatenate([np.zeros(normal_test.sum(), int),
                                                       np.ones(anomaly_test.sum(), int)])}
    pool_size = min(args.pool_size, len(normal_train))
    inject_count = int(pool_size * args.inject_fraction / (1 - args.inject_fraction))
    if not 0 < inject_count <= len(anomaly_train):
        raise ValueError("Injection count must be positive and no larger than available anomalies")
    train_model, encode = namespace["train_autoencoder"], namespace["encode"]
    set_seed, evaluate = namespace["set_seed"], namespace["evaluate"]
    distance, reconstruction, pick_score = (namespace["knn_distance"],
                                              namespace["recon_error_per_ink"], namespace["pick_score"])
    metrics_rows, round_rows, digit_rows = [], [], []
    for seed in range(args.seeds):
        rng = np.random.RandomState(seed)
        normal_indices = rng.choice(normal_train, pool_size, replace=False)
        anomaly_indices = rng.choice(anomaly_train, inject_count, replace=False)
        all_indices = np.concatenate([normal_indices, anomaly_indices])
        images = train_x[all_indices]
        labels = train_y[all_indices]
        injected = np.isin(labels, ANOMALY_DIGITS)
        removal_count = int(len(images) * args.remove_fraction)
        if (removal_count < 1 or pool_size < max(args.k_values) or
            len(images) - removal_count * args.rounds <= max(args.k_values)):
            raise ValueError("Removal settings leave too few samples or remove none")
        seed_dir = output_dir / f"seed_{seed + 1:02d}"
        seed_dir.mkdir()
        print(f"Seed {seed + 1}/{args.seeds}: {pool_size} normal, {inject_count} injected", flush=True)

        for k in args.k_values:
            namespace["K_NEIGHBORS"] = k
            k_dir = seed_dir / f"k_{k:02d}"
            k_dir.mkdir()
            current = np.arange(len(images))
            for round_index in range(args.rounds + 1):
                set_seed(seed * 100 + round_index)
                model = train_model(images[current])
                if round_index == 0:
                    initial_result = scalar_metrics(evaluate(model, images[current], labels[current]))
                    metrics_rows.append({"seed": seed + 1, "k": k, "method": "uncleaned",
                                         "reference_count": len(current),
                                         "injected_remaining": int(injected[current].sum()), **initial_result})
                if round_index == args.rounds:
                    result = scalar_metrics(evaluate(model, images[current], labels[current]))
                    metrics_rows.append({"seed": seed + 1, "k": k, "method": "iterative",
                                         "reference_count": len(current),
                                         "injected_remaining": int(injected[current].sum()), **result})
                    latents = encode(model, images[current])
                    make_umap(latents, labels[current], all_indices[current],
                              np.zeros(len(current), dtype=bool), k_dir / "final_umap",
                              f"Seed {seed + 1}, final retained set, k={k}",
                              args.umap_sample, seed)
                    print(f"  k={k}: accuracy={result['acc']:.4f}, AUROC={result['auc']:.4f}, AP={result['ap']:.4f}", flush=True)
                    break
                latents = encode(model, images[current])
                knn = distance(latents, latents, k, exclude_self=True)
                rec = reconstruction(model, images[current])
                scores = pick_score(knn, rec)
                selected = np.argsort(scores)[-removal_count:]
                removed_mask = np.zeros(len(current), dtype=bool)
                removed_mask[selected] = True
                hits = int(injected[current][removed_mask].sum())
                round_rows.append({"seed": seed + 1, "k": k, "round": round_index + 1,
                                   "before": len(current), "anomalies_before": int(injected[current].sum()),
                                   "removed": removal_count, "anomalies_removed": hits,
                                   "precision": hits / removal_count,
                                   "random_precision": float(injected[current].mean()),
                                   "recall_cumulative": (inject_count - int(injected[current].sum()) + hits) / inject_count})
                for digit in NORMAL_DIGITS + ANOMALY_DIGITS:
                    digit_mask = labels[current] == digit
                    digit_rows.append({"seed": seed + 1, "k": k, "round": round_index + 1,
                                       "digit": digit, "available": int(digit_mask.sum()),
                                       "removed": int((digit_mask & removed_mask).sum())})
                rows = [{"mnist_train_index": int(all_indices[index]), "digit": int(labels[index]),
                     "injected": bool(injected[index]), "removed": bool(removed_mask[position]),
                     "knn_distance": float(knn[position]), "reconstruction_error_per_ink": float(rec[position]),
                     "combined_rank_score": float(scores[position])}
                    for position, index in enumerate(current)]
                write_csv(k_dir / f"round_{round_index + 1:02d}_scores.csv", rows)
                make_umap(latents, labels[current], all_indices[current], removed_mask,
                      k_dir / f"round_{round_index + 1:02d}_umap",
                      f"Seed {seed + 1}, before removal round {round_index + 1}, k={k}",
                      args.umap_sample, seed + round_index)
                round_dir = k_dir / f"round_{round_index + 1:02d}"
                round_dir.mkdir()
                plot_removal_diagnostics(images[current], labels[current], injected[current],
                             removed_mask, knn, rec, scores, round_dir)
                print(f"  k={k} round={round_index + 1}: removed {hits}/{removal_count} injected", flush=True)
                current = current[~removed_mask]
        set_seed(seed * 100 + 92)
        ceiling_model = train_model(images[:pool_size])
        for k in args.k_values:
            namespace["K_NEIGHBORS"] = k
            result = scalar_metrics(evaluate(ceiling_model, images[:pool_size], labels[:pool_size]))
            metrics_rows.append({"seed": seed + 1, "k": k, "method": "ceiling",
                                 "reference_count": pool_size, "injected_remaining": 0, **result})
        write_csv(output_dir / "metrics.csv", metrics_rows)
        write_csv(output_dir / "rounds.csv", round_rows)
        write_csv(output_dir / "digits.csv", digit_rows)
<<<<<<< HEAD
<<<<<<< HEAD
    return metrics_rows, round_rows, digit_rows


def write_summary(output_dir, args, metrics, rounds, digits):
    lines = ["# Iterative cleaning diagnostics", "",
             "Normal digits: 0-6. Injected and test anomaly digits: 7-9.",
             f"Spaces: {args.seeds} independently sampled seeds; each model has 64 latent dimensions.",
             f"Each seed has {args.rounds} cleaning rounds and one final plot. K values: {args.k_values}.",
             "Each k has its own retrained cleaning trajectory. The same k is used for cleaning and evaluation.",
             "The uncleaned and iterative evaluations use their contaminated training sets as KNN references.",
             "The ceiling uses an independently trained normal-only model and reference. Labels are used for evaluation and visualizations, not for training or removal.",
             "The anomaly test prevalence is determined by the selected MNIST test digits, not the injection fraction.",
             "UMAP fits independently at every stage: coordinates and distances are not comparable across plots.",
             "By default all points are shown. If sampling is enabled, all removed and injected points are included and retained normal points are sampled.",
             "UMAP is a visualization, not the space used for KNN or scoring.", "",
             "## How to read the plots", "",
             f"For each seed, open `round_01_umap.png` through `round_{args.rounds:02d}_umap.png` in order, then `final_umap.png`. "
             "The round plots show the training set *before* that round's removal; the final plot shows what remains after all rounds and retraining. "
             "The left panel colors each digit, using crosses for the injected digits 7-9. "
             "The right panel distinguishes retained and removed normal images and retained and removed injected images. "
             "Labels reveal the true outcome for analysis; the cleaning algorithm does not use them. "
             "Counts in the legends are plotted counts, not necessarily full-dataset counts when sampling is enabled.", "",
             "Each plot is a fresh UMAP fit of that stage's 64-dimensional representations. "
             "UMAP places similar examples near each other in a two-dimensional view, but it can distort distances and group sizes. "
             "Orientation, position, and distance between separate plots have no common meaning. "
             "Apparent clusters or overlap are clues, not proof of separation or removal quality.", "",
             "## What the experiment does", "",
             "A seed controls the random selection of normal and injected MNIST training images and the model's training randomness. "
             "The seeds are separate experimental runs, not separate groups on one plot. "
             "An autoencoder learns to reconstruct its input image without using its digit label. "
             "Its encoder turns each image into a 64-number representation called a latent vector; "
             "together these vectors form that model's latent space. An epoch is one pass through the current training set.", "",
             "At each round a fresh autoencoder is trained on the current images. "
             "For each image, the method measures (1) the average distance to its k nearest other images in the latent space, "
             "and (2) reconstruction error, the squared pixel differences between the image and its reconstruction, "
             "divided by the image's total ink. Larger values can indicate unusual images. "
             "Each signal is converted to a relative rank from 0 to 1 among the current images; their average is the combined suspicion score. "
             "This is not a probability. The highest-scoring images are removed, without using their labels. "
             "Each round removes the same number, calculated from the original mixed pool, not from the shrinking pool. "
             "The final model is retrained after the last removal.", "",
             "## Meaning of the measurements", "",
             "- **Removal precision:** injected images removed divided by all images removed. "
             "The other removed images are normal images incorrectly discarded.",
             "- **Random precision:** proportion of injected images present before the round; the expected precision of random removal.",
             "- **Cumulative anomaly recall:** fraction of the original injected images removed through the current round.",
             "- **Accuracy:** fraction of normal test images correctly classified by a labeled k-nearest-neighbor vote, "
             "using the current training set as its reference. This is classification, not anomaly detection.",
             "- **AUROC:** how often an injected test image receives a higher anomaly score than a normal test image; "
             "0.5 is chance and 1 is perfect. Latent and reconstruction AUROC use each signal separately.",
             "- **Average precision (AP):** summarizes precision across detection thresholds; higher is better. "
             "Its random baseline is the fraction of anomalies in the test set, which is not the training injection fraction.",
             "- **Uncleaned:** model and reference before removal. **Iterative:** after all rounds. "
             "**Ceiling:** a separately trained model given only normal training images; a reference, not a cleaning method.",
             "- **Mean +/- population standard deviation:** average across seeds and how much the seed results vary. "
             "These are descriptive, not confidence intervals.", "",
=======
=======
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
    return metrics_rows, round_rows


def write_summary(output_dir, args, metrics, rounds):
    lines = ["# Iterative cleaning diagnostics", "",
             "Normal digits: 0-6. Injected and test anomaly digits: 7-8; digit 9 is excluded.",
             f"Five-space default: {args.seeds} independent seeds, 64 latent dimensions.",
             f"K values: {args.k_values}; UMAP and individual score files are saved for every k.",
             "Each k has its own retrained cleaning trajectory. The same k is used for cleaning and evaluation.",
             "The uncleaned and iterative evaluations use their contaminated training sets as KNN references.",
             "The ceiling uses an independently trained normal-only model and reference. Labels are only used for evaluation.",
             "The anomaly test prevalence is determined by the selected MNIST test digits, not the injection fraction.",
             "UMAP fits independently at every stage: coordinates and distances are not comparable across plots.",
             "By default all points are shown. If sampling is enabled, all removed and injected points are included and normal retained points are sampled.",
             "UMAP is a visualization, not the space used for KNN or scoring.", "",
<<<<<<< HEAD
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
=======
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
             "## Results", "", "Values are mean +/- population standard deviation across seeds.", "",
             "| k | method | accuracy | AUROC | average precision | latent AUROC | reconstruction AUROC |",
             "|---|---|---|---|---|---|---|"]
    for k in args.k_values:
        for method in ("uncleaned", "iterative", "ceiling"):
            selected = [row for row in metrics if row["k"] == k and row["method"] == method]
            values = [np.array([row[key] for row in selected]) for key in ("acc", "auc", "ap", "auc_latent", "auc_recon")]
            lines.append(f"| {k} | {method} | " + " | ".join(f"{value.mean():.3f} +/- {value.std():.3f}" for value in values) + " |")
    lines += ["", "## Removal", "", "| k | round | precision | random precision | cumulative anomaly recall |",
              "|---|---|---|---|---|"]
    for k in args.k_values:
        for round_index in range(1, args.rounds + 1):
            selected = [row for row in rounds if row["k"] == k and row["round"] == round_index]
            lines.append(f"| {k} | {round_index} | " + " | ".join(
                f"{np.mean([row[key] for row in selected]):.3f}" for key in
                ("precision", "random_precision", "recall_cumulative")) + " |")
<<<<<<< HEAD
<<<<<<< HEAD
    lines += ["", "## What these results mean", ""]
    for k in args.k_values:
        first = [row for row in rounds if row["k"] == k and row["round"] == 1]
        last = [row for row in rounds if row["k"] == k and row["round"] == args.rounds]
        selected = [row for row in rounds if row["k"] == k]
        initial_anomalies = np.mean([row["anomalies_before"] for row in first])
        removed_anomalies = sum(row["anomalies_removed"] for row in selected) / len(first)
        removed_normal = sum(row["removed"] - row["anomalies_removed"] for row in selected) / len(first)
        remaining_anomalies = np.mean([row["anomalies_before"] - row["anomalies_removed"] for row in last])
        remaining_total = np.mean([row["before"] - row["removed"] for row in last])
        by_method = {method: [row for row in metrics if row["k"] == k and row["method"] == method]
                     for method in ("uncleaned", "iterative", "ceiling")}
        def mean_metric(method, key):
            return np.mean([row[key] for row in by_method[method]])
        lines += [
            f"For k={k}, each seed started with an average of {initial_anomalies:.1f} injected images. "
            f"Across all rounds, the method removed an average of {removed_anomalies:.1f} injected images "
            f"and {removed_normal:.1f} normal images per seed. "
            f"An average of {remaining_anomalies:.1f} injected images remained among {remaining_total:.1f} "
            f"retained images ({remaining_anomalies / remaining_total:.1%} of the retained set).",
            f"Average cumulative anomaly recall was {np.mean([row['recall_cumulative'] for row in last]):.1%}: "
            "this is the share of initially injected images found, not the share of all removals that were correct.",
            f"Compared with the uncleaned model, the final model's mean test accuracy changed by "
            f"{(mean_metric('iterative', 'acc') - mean_metric('uncleaned', 'acc')) * 100:+.2f} percentage points "
            f"and its detection AUROC changed by "
            f"{(mean_metric('iterative', 'auc') - mean_metric('uncleaned', 'auc')) * 100:+.2f} points. "
            f"Its detection AUROC was {(mean_metric('iterative', 'auc') - mean_metric('ceiling', 'auc')) * 100:+.2f} "
            "points relative to the normal-only ceiling. These are descriptive comparisons, not evidence that "
            "cleaning caused the differences; models are retrained and can vary with randomness.",
            ""
        ]
    lines += ["", "## Every seed and round", "",
              "Counts refer to the full training set, even when plots are sampled.", "",
              "| Seed | k | Round | Before | Injected before | Removed | Injected removed | Normal removed | Injected left | Precision | Random precision | Cumulative recall |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rounds:
        lines.append(
            f"| {row['seed']} | {row['k']} | {row['round']} | {row['before']} | "
            f"{row['anomalies_before']} | {row['removed']} | {row['anomalies_removed']} | "
            f"{row['removed'] - row['anomalies_removed']} | {row['anomalies_before'] - row['anomalies_removed']} | "
            f"{row['precision']:.1%} | {row['random_precision']:.1%} | {row['recall_cumulative']:.1%} |"
        )
    lines += ["", "## Removed by digit", "",
              "Each entry is removed / available before that round. Digits 0-6 are normal; 7-9 are injected.", "",
              "| Seed | k | Round | " + " | ".join(str(d) for d in NORMAL_DIGITS + ANOMALY_DIGITS) + " |",
              "|---|---|---|" + "---|" * 10]
    for row in rounds:
        group = [d for d in digits if d["seed"] == row["seed"] and d["k"] == row["k"] and d["round"] == row["round"]]
        lines.append(f"| {row['seed']} | {row['k']} | {row['round']} | " +
                     " | ".join(f"{d['removed']} / {d['available']}" for d in group) + " |")
    lines += ["", "## Per-seed evaluation", "",
              "| Seed | k | Method | Reference size | Injected remaining | Accuracy | AUROC | AP |",
              "|---|---|---|---|---|---|---|---|"]
    for row in metrics:
        lines.append(f"| {row['seed']} | {row['k']} | {row['method']} | {row['reference_count']} | "
                     f"{row['injected_remaining']} | {row['acc']:.3f} | {row['auc']:.3f} | {row['ap']:.3f} |")
=======
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
=======
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
    lines += ["", "## Files", "", "- metrics.csv: per-seed evaluation metrics and retained anomaly counts.",
              "- rounds.csv: per-seed removal precision, current random baseline and cumulative recall.",
              "- digits.csv: available and removed counts for each digit, round, seed and k.",
              "- seed_NN/k_NN/round_NN_scores.csv: all current images, scores and removal decisions.",
              "- seed_NN/k_NN/*_umap.png and .csv: projections, true digits, injection and removal status.",
<<<<<<< HEAD
<<<<<<< HEAD
=======
              "- seed_NN/k_NN/*_umap_digits.png: each digit highlighted separately in the same projection.",
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
=======
              "- seed_NN/k_NN/*_umap_digits.png: each digit highlighted separately in the same projection.",
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
              "- seed_NN/k_NN/round_NN/: signal distributions and examples of removed normal and injected digits.",
              "- config.json: full settings and source notebook."]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
<<<<<<< HEAD
<<<<<<< HEAD
    parser.add_argument("--k-values", type=int, nargs="+", default=[5],
=======
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20],
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
=======
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20],
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
                        help="K values for independent cleaning trajectories and plots")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--pool-size", type=int, default=20000)
    parser.add_argument("--inject-fraction", type=float, default=0.05)
    parser.add_argument("--remove-fraction", type=float, default=0.02)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--umap-sample", type=int, default=0,
                        help="Maximum plotted points (0: all); removed and injected points are always included")
    args = parser.parse_args()
    if (args.seeds < 1 or args.epochs < 1 or args.pool_size < 1 or args.rounds < 1 or
            (args.umap_sample != 0 and args.umap_sample < 3) or not 0 < args.inject_fraction < 1 or
            not 0 < args.remove_fraction < 1 or any(k < 1 for k in args.k_values) or
            len(set(args.k_values)) != len(args.k_values)):
        parser.error("Counts and k values must be positive, k values unique, fractions between 0 and 1, and UMAP sample 0 or >= 3")
    try:
<<<<<<< HEAD
<<<<<<< HEAD
        importlib.import_module("umap")
    except ModuleNotFoundError as error:
        if error.name != "umap":
            raise
        parser.error("UMAP is required: install umap-learn in the Python environment running this script")
    output_dir = AUTOENCODER / "Diagnostics runs" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True)
=======
=======
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
        import umap  # noqa: F401
    except ImportError:
        parser.error("UMAP is required: install umap-learn in the Python environment running this script")
    output_dir = HERE / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output_dir.mkdir()
<<<<<<< HEAD
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
=======
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
    config = vars(args) | {"normal_digits": NORMAL_DIGITS, "anomaly_digits": ANOMALY_DIGITS,
                          "latent_dim": 64, "source_notebook": str(NOTEBOOK), "status": "running"}
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Saving diagnostics to {output_dir}", flush=True)
    started = time.monotonic()
    try:
<<<<<<< HEAD
<<<<<<< HEAD
        namespace = {"np": np, "torch": torch, "nn": nn, "F": F, "random": random,
                     "average_precision_score": average_precision_score, "roc_auc_score": roc_auc_score}
        metrics, rounds, digits = run_experiment(args, output_dir, namespace)
        write_summary(output_dir, args, metrics, rounds, digits)
=======
        metrics, rounds = run_experiment(args, output_dir, globals().copy())
        write_summary(output_dir, args, metrics, rounds)
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
=======
        metrics, rounds = run_experiment(args, output_dir, globals().copy())
        write_summary(output_dir, args, metrics, rounds)
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
    except Exception as error:
        config.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        config["status"] = "completed"
    finally:
        config["elapsed_seconds"] = round(time.monotonic() - started, 2)
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Finished: {output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()