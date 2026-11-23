"""Run five-space cleaning diagnostics for MNIST digits 0-6 with 7-9 injected.

The autoencoder and kNN cleaning run in the full latent space. UMAP projects
selected examples into two dimensions only to visualize each stage; the plotted
coordinates are not used to calculate scores or decide which images to remove.
"""

# argparse defines and validates command-line options.
import argparse
# csv writes per-image, per-round, and per-digit tables.
import csv
# importlib reads the UMAP package at runtime so a missing package has a clear error.
import importlib
# json reads the notebook and writes run configuration files.
import json
# random seeds the Python standard-library generator used by notebook functions.
import random
# time measures elapsed run duration.
import time
# datetime gives each run a unique timestamped output folder.
from datetime import datetime
# Path builds paths relative to this script and the project.
from pathlib import Path

# Matplotlib saves figures without requiring a graphical desktop session.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# NumPy handles sampling, masks, ranks/metrics, and CSV-compatible values.
import numpy as np
# PyTorch loads image tensors, selects a compute device, and trains the model.
import torch
# These modules are provided to notebook functions that build/train the autoencoder.
import torch.nn as nn
import torch.nn.functional as F
# These metrics evaluate the classifier and anomaly-detection scores.
from sklearn.metrics import average_precision_score, roc_auc_score
# torchvision loads the local MNIST training and test datasets.
from torchvision import datasets
from compare_diagnostics import write_results_text


# HERE is the directory containing this script.
HERE = Path(__file__).resolve().parent
# AUTOENCODER is the enclosing Autoencoder project folder.
AUTOENCODER = HERE.parents[5]
# The notebook is the source of the model, training, scoring, and evaluation functions.
NOTEBOOK = AUTOENCODER / "anomaly_cleaning_train_0-6_inject_7-9_before_evaluation_fix.ipynb"
# Digits 0 through 6 are treated as normal examples.
NORMAL_DIGITS = list(range(7))
# Digits 7 through 9 are injected into training and treated as test anomalies.
ANOMALY_DIGITS = [7, 8, 9]


def load_original_functions(namespace):
    """Execute the notebook cells defining the model and evaluation functions."""
    # Read the notebook's JSON structure from disk.
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    # These three definitions are the functions/classes needed by this script.
    markers = ("class ConvAutoencoder", "def encode(", "def evaluate(")
    # Find and execute exactly one code cell for each required definition.
    for marker in markers:
        # Notebook cell sources may be stored as a list of text fragments.
        matches = ["".join(cell.get("source", [])) for cell in notebook["cells"]
                   if cell.get("cell_type") == "code" and marker in "".join(cell.get("source", []))]
        # Fail explicitly if the notebook was changed so the expected cell is absent/duplicated.
        if len(matches) != 1:
            raise RuntimeError(f"Expected one notebook cell containing {marker!r}; found {len(matches)}")
        # Compile and execute the cell in the shared namespace so its definitions
        # can use the imports/settings supplied by run_experiment().
        exec(compile(matches[0], str(NOTEBOOK), "exec"), namespace)


def write_csv(path, rows):
    """Write a list of same-shaped dictionaries as a headered CSV file."""
    # Do not create an empty file with no header when there are no rows.
    if not rows:
        return
    # newline="" lets csv manage line endings consistently across platforms.
    with path.open("w", newline="", encoding="utf-8") as output:
        # Use the first row's keys as the output column order.
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        # Write one header row followed by all data dictionaries.
        writer.writeheader()
        writer.writerows(rows)


def make_umap(latents, labels, indices, removed_mask, output_dir, title, sample_size, seed):
    """Project one training-set stage to 2D and save plot plus point metadata.

    All removed and injected examples are retained in a sampled map. If sampling
    is enabled, only retained normal examples may be omitted. This keeps the
    important cleaning outcomes visible while limiting the plot size.
    """
    # Use a repeatable random sampler for optional retained-normal points.
    rng = np.random.RandomState(seed)
    # Mark ground-truth injected digits; labels are for visualization, not scoring.
    injected = np.isin(labels, ANOMALY_DIGITS)
    # Always include every point removed this round and every injected point.
    required = np.flatnonzero(removed_mask | injected)
    # Only retained normal examples are eligible to be sampled or left out.
    optional = np.flatnonzero(~(removed_mask | injected))
    # A sample size of zero means keep every optional point; otherwise fill up
    # the requested size after accounting for the required points.
    optional_count = len(optional) if sample_size == 0 else min(len(optional), max(0, sample_size - len(required)))
    # Combine required and randomly sampled optional positions, then sort them
    # so plotted rows retain the original order in the current training set.
    chosen = np.sort(np.concatenate([required, rng.choice(optional, optional_count, replace=False)]))
    # UMAP needs at least three examples to form a meaningful 2D projection.
    if len(chosen) < 3:
        raise ValueError("UMAP needs at least three sampled training images")
    # Import UMAP only when it is needed; main() checks this dependency up front.
    umap = importlib.import_module("umap")
    # Fit a fresh UMAP model for this stage. n_neighbors is capped at 15 and
    # below the sample count; random_state makes the fit repeatable, and n_jobs=1
    # avoids nondeterminism from parallel execution. Each stage is fitted alone.
    coordinates = np.asarray(umap.UMAP(n_neighbors=min(15, len(chosen) - 1), random_state=seed,
                                       n_jobs=1).fit_transform(latents[chosen].numpy()))
    # Build CSV rows joining each projected point back to its original MNIST
    # training index, true digit, injection status, and removal status.
    rows = [{"mnist_train_index": int(indices[position]), "digit": int(labels[position]),
             "injected": bool(injected[position]),
             "removed_this_round": bool(removed_mask[position]),
             "umap_1": float(point[0]), "umap_2": float(point[1])}
            # chosen contains positions in the current arrays; coordinates has
            # one matching projected row for each of those positions.
            for position, point in zip(chosen, coordinates)]
    # Save the coordinate table next to the plot using the same basename.
    write_csv(output_dir.with_suffix(".csv"), rows)


def make_overview(projection_dir, output_path, seed, k, score_mode):
    """Render the four saved projections without refitting UMAP."""
    from matplotlib.lines import Line2D

    markers = ("o", "s", "^", "v", "D", "P", "X", "<", ">", "*")
    colors = ("#a9a9a9",) * 7 + ("#0072b2", "#d55e00", "#cc79a7")
    fig, axes = plt.subplots(2, 2, figsize=(18, 15), constrained_layout=True)
    for axis, name, title in zip(axes.flat,
                                 ("round_01_umap", "round_02_umap", "round_03_umap", "final_umap"),
                                 ("Round 1: before removal", "Round 2: before removal",
                                  "Round 3: before removal", "Final: after retraining")):
        with (projection_dir / f"{name}.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        digits = np.array([int(row["digit"]) for row in rows])
        points = np.array([[float(row["umap_1"]), float(row["umap_2"])] for row in rows])
        removed = np.array([row["removed_this_round"].lower() == "true" for row in rows])
        for digit in range(10):
            mask = digits == digit
            axis.scatter(points[mask, 0], points[mask, 1], marker=markers[digit],
                         c=colors[digit], s=10 if digit < 7 else 20,
                         alpha=0.5 if digit < 7 else 0.9, linewidths=0, rasterized=True)
        axis.scatter(points[removed, 0], points[removed, 1], marker="o", s=38,
                     facecolors="none", edgecolors="black", linewidths=0.6,
                     alpha=0.8, rasterized=True)
        removed_injected = int((removed & (digits >= 7)).sum())
        axis.set_title(f"{title} | shown: {len(rows):,}\n"
                       f"Removed: {removed.sum() - removed_injected:,} normal, "
                       f"{removed_injected:,} injected | injected shown: {(digits >= 7).sum():,}")
        axis.set(xlabel="UMAP 1", ylabel="UMAP 2")
    handles = [Line2D([], [], linestyle="", marker=markers[digit], color=colors[digit],
                      markersize=8, label=f"{digit}" + (" (injected)" if digit >= 7 else ""))
               for digit in range(10)]
    handles.append(Line2D([], [], linestyle="", marker="o", markerfacecolor="none",
                          markeredgecolor="black", markersize=8, label="Removed this round"))
    fig.legend(handles=handles, loc="outside lower center", ncol=6, title="True digit (evaluation only)")
    fig.suptitle(f"{score_mode} scoring | seed {seed} | k={k}\n"
                 "Separate UMAP fits: positions and distances are not comparable between panels")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def scalar_metrics(metrics):
    """Convert the evaluation outputs needed in the report to plain floats."""
    # Select only the requested classification/detection metrics and normalize
    # their scalar types for consistent CSV/JSON serialization.
    return {key: float(metrics[key]) for key in ("acc", "auc", "ap", "auc_latent", "auc_recon")}


def plot_score_diagnostics(injected, removed, knn, rec, scores, path, score_mode):
    """Save score distributions and signal relationships."""
    # Create a score histogram and a scatter plot of the two cleaning signals.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    # Compare normal and injected examples separately in both views.
    for mask, label, color in [(~injected, "Normal", "#a9a9a9"),
                               (injected, "Injected", "#d55e00")]:
        # The histogram shows score distributions for the two ground-truth groups.
        axes[0].hist(scores[mask], bins=40, density=True, alpha=0.55, label=label, color=color)
        # Each point relates its kNN distance to its reconstruction error.
        axes[1].scatter(knn[mask], rec[mask], s=3, alpha=0.25, label=label, color=color,
                        rasterized=True)
    # Mark the lowest score among selected removals as the effective cutoff.
    axes[0].axvline(scores[removed].min(), color="black", linestyle="--", label="Removal cutoff")
    # Label the axes and titles for the two signal diagnostic plots.
    axes[0].set(xlabel=f"{score_mode} rank score", ylabel="Density", title="Score distributions")
    axes[1].set(xlabel="Mean kNN distance", ylabel="Reconstruction error per ink",
                title="Cleaning signals")
    # Show the normal/injected legend on each panel.
    for axis in axes:
        axis.legend()
    # Write the two-panel plot and close the figure.
    fig.savefig(path / "score_diagnostics.png", dpi=150)
    plt.close(fig)


def plot_removal_diagnostics(images, labels, injected, removed, knn, rec, scores, path, score_mode):
    """Save score distributions, signal relationships, and removed-image examples."""
    plot_score_diagnostics(injected, removed, knn, rec, scores, path, score_mode)

    # Reserve up to eight image thumbnails for each removal outcome.
    fig, axes = plt.subplots(2, 8, figsize=(16, 5), squeeze=False, constrained_layout=True)
    # Show removed normal examples in row one and removed injected examples in row two.
    for row, (mask, title) in enumerate([(removed & ~injected, "Removed normal"),
                                         (removed & injected, "Removed injected")]):
        # Find positions in this outcome group and sort from highest score downward.
        positions = np.flatnonzero(mask)
        positions = positions[np.argsort(scores[positions])[::-1]][:8]
        # Hide all axes initially so unused thumbnail slots are blank.
        for axis in axes[row]:
            axis.axis("off")
        # Draw the highest-scoring examples, if any, in the row.
        for axis, position in zip(axes[row], positions):
            # Convert the selected grayscale image tensor to an array for display.
            axis.imshow(images[position, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            # Display the true digit and its relative combined rank score.
            axis.set_title(f"Digit {labels[position]} | {scores[position]:.3f}", fontsize=8)
        # Label the row by outcome.
        axes[row, 0].set_ylabel(title)
        # Add a larger rotated row label alongside the thumbnails.
        axes[row, 0].text(-0.15, 0.5, title, rotation=90, transform=axes[row, 0].transAxes,
                          va="center", ha="right", fontsize=9)
    # Explain that this plot ranks removed examples by relative score.
    fig.suptitle("Highest-scoring removed examples (score is a relative rank)")
    # Save and close the image-example figure.
    fig.savefig(path / "removed_examples.png", dpi=150)
    plt.close(fig)


def run_experiment(args, output_dir, namespace):
    """Run seeded iterative cleaning, plus a normal-only ceiling comparison."""
    # Supply notebook globals for this diagnostic: fixed latent width, requested
    # epochs, batch size, learning rate, and the combined kNN/reconstruction score.
    namespace.update({"LATENT_DIM": 64, "EPOCHS": args.epochs, "BATCH_SIZE": 128,
                      "LEARNING_RATE": 1e-3, "SCORE_MODE": args.score_mode})
    # Prefer CUDA, then Apple Metal (MPS), and fall back to the CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # Enable automatic mixed precision only for CUDA in this experiment.
    namespace.update({"DEVICE": device, "USE_AMP": device.type == "cuda"})
    # Load the model, training, scoring, and evaluation definitions from the notebook.
    load_original_functions(namespace)
    # Search this directory and its ancestors for the repository MNIST files.
    for base in (HERE, *HERE.parents):
        # Use the project data folder whose raw training image file is present.
        if (base / "data" / "MNIST" / "raw" / "train-images-idx3-ubyte").exists():
            data_root = base / "data"
            break
    else:
        # Give an explicit setup error rather than letting torchvision download data.
        raise FileNotFoundError("Could not find data/MNIST/raw in the parent directories")
    # Load MNIST locally; download=False keeps this diagnostic offline.
    train_set = datasets.MNIST(data_root, train=True, download=False)
    test_set = datasets.MNIST(data_root, train=False, download=False)
    # Scale pixels from byte values to [0, 1] and add the channel dimension.
    train_x = train_set.data.float().div(255).unsqueeze(1)
    # Keep integer training labels for sampling groups and computing diagnostics.
    train_y = train_set.targets.numpy()
    # Apply the same image preprocessing to the held-out test images.
    test_x = test_set.data.float().div(255).unsqueeze(1)
    # Keep test labels for evaluation; they are not used in training or cleaning.
    test_y = test_set.targets.numpy()
    # Identify candidate normal and anomalous images by their ground-truth digits.
    normal_train = np.flatnonzero(np.isin(train_y, NORMAL_DIGITS))
    anomaly_train = np.flatnonzero(np.isin(train_y, ANOMALY_DIGITS))
    # Create corresponding ground-truth masks for the held-out test set.
    normal_test = np.isin(test_y, NORMAL_DIGITS)
    anomaly_test = np.isin(test_y, ANOMALY_DIGITS)
    # Convert test masks to torch boolean tensors for indexing test images.
    normal_test_tensor = torch.as_tensor(normal_test, dtype=torch.bool)
    anomaly_test_tensor = torch.as_tensor(anomaly_test, dtype=torch.bool)
    # Populate the notebook evaluation context. The detection examples contain
    # normal test images followed by anomalous test images, with binary labels
    # marking which are anomalous.
    namespace["ctx"] = {"normal_x": test_x[normal_test_tensor], "normal_y": test_y[normal_test],
                        "det_x": torch.cat([test_x[normal_test_tensor], test_x[anomaly_test_tensor]]),
                        "det_is_anom": np.concatenate([np.zeros(normal_test.sum(), int),
                                                       np.ones(anomaly_test.sum(), int)])}
    # Limit the normal reference pool to the requested maximum or available normals.
    pool_size = min(args.pool_size, len(normal_train))
    # Solve injected / (normal + injected) = inject_fraction for injected count.
    inject_count = int(pool_size * args.inject_fraction / (1 - args.inject_fraction))
    # Check that the requested injection count can be drawn from available anomalies.
    if not 0 < inject_count <= len(anomaly_train):
        raise ValueError("Injection count must be positive and no larger than available anomalies")
    # Cache notebook callables used repeatedly in the experiment.
    train_model, encode = namespace["train_autoencoder"], namespace["encode"]
    set_seed, evaluate = namespace["set_seed"], namespace["evaluate"]
    # Cleaning operates on latent kNN distance and per-ink reconstruction error,
    # combined by the notebook's rank-based score function.
    distance, reconstruction, pick_score = (namespace["knn_distance"],
                                              namespace["recon_error_per_ink"], namespace["pick_score"])
    # Accumulate evaluation, round, and per-digit records across all seeds and k values.
    metrics_rows, round_rows, digit_rows = [], [], []
    # Run an independent random sample and model trajectory for each seed.
    for seed in range(args.seeds):
        # Use this seed to draw a pool of normal and injected MNIST training indices.
        rng = np.random.RandomState(seed)
        normal_indices = rng.choice(normal_train, pool_size, replace=False)
        anomaly_indices = rng.choice(anomaly_train, inject_count, replace=False)
        # Keep normal indices first; this makes images[:pool_size] the normal-only
        # reference used later for the ceiling comparison.
        all_indices = np.concatenate([normal_indices, anomaly_indices])
        # Materialize the selected image tensors and their true digit labels.
        images = train_x[all_indices]
        labels = train_y[all_indices]
        # Mark injected examples from ground truth for analysis only.
        injected = np.isin(labels, ANOMALY_DIGITS)
        # Keep the number of images removed per round fixed relative to the original pool.
        removal_count = int(len(images) * args.remove_fraction)
        # Require at least one removal and enough current examples for kNN at every stage.
        if (removal_count < 1 or pool_size < max(args.k_values) or
            len(images) - removal_count * args.rounds <= max(args.k_values)):
            raise ValueError("Removal settings leave too few samples or remove none")
        # Create this seed's output folder.
        seed_dir = output_dir / "data" / f"seed_{seed + 1:02d}"
        seed_dir.mkdir(parents=True)
        # Report pool composition before beginning the k trajectories.
        print(f"Seed {seed + 1}/{args.seeds}: {pool_size} normal, {inject_count} injected", flush=True)

        # Compare each requested k independently; each gets its own fitted models
        # and cleaning trajectory rather than sharing removed examples.
        for k in args.k_values:
            # The notebook's kNN helpers read their neighbor count from this setting.
            namespace["K_NEIGHBORS"] = k
            # Place this k value's plots and per-round artifacts in its own folder.
            k_dir = seed_dir / f"k_{k:02d}"
            k_dir.mkdir()
            figures_dir = output_dir / "figures" / f"seed_{seed + 1:02d}" / f"k_{k:02d}"
            figures_dir.mkdir(parents=True)
            # current stores positions into images/labels, initially all examples.
            current = np.arange(len(images))
            # Train once before cleaning and once after each removal round.
            for round_index in range(args.rounds + 1):
                # Derive a repeatable training seed for each seed/round combination.
                set_seed(seed * 100 + round_index)
                # Fit a fresh autoencoder using only the currently retained images.
                model = train_model(images[current])
                if round_index == 0:
                    # Measure the starting contaminated pool before any cleaning.
                    initial_result = scalar_metrics(evaluate(model, images[current], labels[current]))
                    # Record the baseline metrics and starting reference composition.
                    metrics_rows.append({"seed": seed + 1, "k": k, "method": "uncleaned",
                                         "reference_count": len(current),
                                         "injected_remaining": int(injected[current].sum()), **initial_result})
                if round_index == args.rounds:
                    # This is the model trained after all removals; evaluate its final set.
                    result = scalar_metrics(evaluate(model, images[current], labels[current]))
                    # Record final iterative metrics and remaining injected count.
                    metrics_rows.append({"seed": seed + 1, "k": k, "method": "iterative",
                                         "reference_count": len(current),
                                         "injected_remaining": int(injected[current].sum()), **result})
                    # Encode the final retained set for its own UMAP plot.
                    latents = encode(model, images[current])
                    # Plot the final set, marking no examples as removed at this final stage.
                    make_umap(latents, labels[current], all_indices[current],
                              np.zeros(len(current), dtype=bool), k_dir / "final_umap",
                              f"Seed {seed + 1}, final retained set, k={k}",
                              args.umap_sample, seed)
                    make_overview(k_dir, figures_dir / "cleaning_overview.png",
                                  seed + 1, k, args.score_mode)
                    # Report final classification and detection metrics.
                    print(f"  k={k}: accuracy={result['acc']:.4f}, AUROC={result['auc']:.4f}, AP={result['ap']:.4f}", flush=True)
                    # No more cleaning is needed for this k trajectory.
                    break
                # Encode currently retained images in this round's newly trained model.
                latents = encode(model, images[current])
                # Compute latent-space mean distance to k neighbors, excluding self.
                knn = distance(latents, latents, k, exclude_self=True)
                # Compute each image's reconstruction error normalized by its ink.
                rec = reconstruction(model, images[current])
                # Convert both signals to a combined relative suspicion score.
                scores = pick_score(knn, rec)
                # Select exactly removal_count examples with the largest combined scores.
                selected = np.argsort(scores)[-removal_count:]
                # Build a boolean mask local to current, True for chosen removals.
                removed_mask = np.zeros(len(current), dtype=bool)
                removed_mask[selected] = True
                # Count how many selected examples are injected, for evaluation only.
                hits = int(injected[current][removed_mask].sum())
                # Save round-level cleaning counts and quality measures.
                round_rows.append({"seed": seed + 1, "k": k, "round": round_index + 1,
                                   "before": len(current), "anomalies_before": int(injected[current].sum()),
                                   "removed": removal_count, "anomalies_removed": hits,
                                   # Precision is anomaly removals / all removals.
                                   "precision": hits / removal_count,
                                   # Random precision is the current anomaly fraction.
                                   "random_precision": float(injected[current].mean()),
                                   # Recall is the share of the original injected pool removed so far.
                                   "recall_cumulative": (inject_count - int(injected[current].sum()) + hits) / inject_count})
                # Record how many examples of each digit are available and removed this round.
                for digit in NORMAL_DIGITS + ANOMALY_DIGITS:
                    digit_mask = labels[current] == digit
                    digit_rows.append({"seed": seed + 1, "k": k, "round": round_index + 1,
                                       "digit": digit, "available": int(digit_mask.sum()),
                                       "removed": int((digit_mask & removed_mask).sum())})
                # Save every current example's original MNIST index, label, scores,
                # injected status, and whether it is selected for removal.
                rows = [{"mnist_train_index": int(all_indices[index]), "digit": int(labels[index]),
                     "injected": bool(injected[index]), "removed": bool(removed_mask[position]),
                     "knn_distance": float(knn[position]), "reconstruction_error_per_ink": float(rec[position]),
                     "rank_score": float(scores[position])}
                    # position indexes arrays aligned to current; index maps back
                    # to the original position in images/all_indices.
                    for position, index in enumerate(current)]
                # Save the score/removal table for this round.
                write_csv(k_dir / f"round_{round_index + 1:02d}_scores.csv", rows)
                # Plot this round's training set before its removal is applied.
                make_umap(latents, labels[current], all_indices[current], removed_mask,
                      k_dir / f"round_{round_index + 1:02d}_umap",
                      f"Seed {seed + 1}, before removal round {round_index + 1}, k={k}",
                      args.umap_sample, seed + round_index)
                # Prepare a directory for the signal and removed-image diagnostics.
                round_dir = figures_dir / f"round_{round_index + 1:02d}"
                round_dir.mkdir()
                # Save histograms, signal scatter, and example removed images.
                plot_removal_diagnostics(images[current], labels[current], injected[current],
                             removed_mask, knn, rec, scores, round_dir, args.score_mode)
                # Report how many injected anomalies were among this round's removals.
                print(f"  k={k} round={round_index + 1}: removed {hits}/{removal_count} injected", flush=True)
                # Apply the removal mask; later rounds train only on retained examples.
                current = current[~removed_mask]
        # Train a separate comparison model on the initial normal-only subset.
        set_seed(seed * 100 + 92)
        ceiling_model = train_model(images[:pool_size])
        # Evaluate that same normal-only model for every requested k.
        for k in args.k_values:
            # Set the neighbor count used by the notebook evaluation helper.
            namespace["K_NEIGHBORS"] = k
            # Evaluate against the normal-only images and record the ceiling result.
            result = scalar_metrics(evaluate(ceiling_model, images[:pool_size], labels[:pool_size]))
            metrics_rows.append({"seed": seed + 1, "k": k, "method": "ceiling",
                                 "reference_count": pool_size, "injected_remaining": 0, **result})
        # Persist cumulative results after each seed so partial runs retain progress.
        write_csv(output_dir / "data" / "metrics.csv", metrics_rows)
        write_csv(output_dir / "data" / "rounds.csv", round_rows)
        write_csv(output_dir / "data" / "digits.csv", digit_rows)
    # Return all rows so write_summary can add aggregate explanations and tables.
    return metrics_rows, round_rows, digit_rows


def write_summary(output_dir, args, metrics, rounds, digits):
    """Write an interpretation guide and aggregate/per-seed results as Markdown."""
    # Begin with the experimental setup and essential cautions for interpreting plots.
    score_description = {
        "combined": "the average of the two relative ranks",
        "latent": "the relative rank of the kNN distance alone",
        "recon": "the relative rank of reconstruction error alone",
    }[args.score_mode]
    lines = [f"# Iterative cleaning diagnostics: {args.score_mode} scoring", "",
             "Normal digits: 0-6. Injected and test anomaly digits: 7-9.",
             f"Spaces: {args.seeds} independently sampled seeds; each model has 64 latent dimensions.",
             "Seed folder numbers start at 1; the corresponding initial sampling seeds start at 0. "
             "For zero-based run index s, training uses seeds 100*s + round index (0-3), and 100*s + 92 for the ceiling.",
             f"Each seed has {args.rounds} cleaning rounds and one final plot. K values: {args.k_values}.",
             "Each k has its own retrained cleaning trajectory. The same k is used for cleaning and evaluation.",
             "The uncleaned and iterative evaluations use their contaminated training sets as KNN references.",
             "The ceiling uses an independently trained normal-only model and reference. Labels are used for evaluation and visualizations, not for training or removal.",
             "The anomaly test prevalence is determined by the selected MNIST test digits, not the injection fraction.",
             "UMAP fits independently at every stage: coordinates and distances are not comparable across plots.",
             "By default all points are shown. If sampling is enabled, all removed and injected points are included and retained normal points are sampled.",
             "UMAP is a visualization, not the space used for KNN or scoring.", "",
             # Explain the visual encoding of the two UMAP panels.
             "## How to read the plots", "",
             "For each seed, open `figures/seed_NN/k_NN/cleaning_overview.png`. "
             "The round plots show the training set *before* that round's removal; the final plot shows what remains after all rounds and retraining. "
             "Shapes identify the true digits; normal digits are grey and injected digits are colored. "
             "A black ring marks an image removed in that round. "
             "Labels reveal the true outcome for analysis; the cleaning algorithm does not use them. "
             "Counts in the legends are plotted counts, not necessarily full-dataset counts when sampling is enabled.", "",
             "Each plot is a fresh UMAP fit of that stage's 64-dimensional representations. "
             "UMAP places similar examples near each other in a two-dimensional view, but it can distort distances and group sizes. "
             "Orientation, position, and distance between separate plots have no common meaning. "
             "Apparent clusters or overlap are clues, not proof of separation or removal quality.", "",
             # Give a plain-language description of the model, latent vectors, and cleaning signals.
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
             f"Each signal is converted to a relative rank from 0 to 1 among the current images; this method uses {score_description}. "
             "This is not a probability. The highest-scoring images are removed, without using their labels. "
             "Each round removes the same number, calculated from the original mixed pool, not from the shrinking pool. "
             "The final model is retrained after the last removal.", "",
             # Define each reported metric and explain which model is its reference.
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
             # Start an aggregate table comparing the uncleaned, cleaned, and ceiling models.
             "## Results", "", "Values are mean +/- population standard deviation across seeds.", "",
             "| k | method | accuracy | AUROC | average precision | latent AUROC | reconstruction AUROC |",
             "|---|---|---|---|---|---|---|"]
    # Compute one row for each k value and evaluation stage.
    for k in args.k_values:
        for method in ("uncleaned", "iterative", "ceiling"):
            # Select results across seeds for this k and method.
            selected = [row for row in metrics if row["k"] == k and row["method"] == method]
            # Collect all five reported metrics as arrays for mean and population std.
            values = [np.array([row[key] for row in selected]) for key in ("acc", "auc", "ap", "auc_latent", "auc_recon")]
            # Format each metric as mean +/- standard deviation.
            lines.append(f"| {k} | {method} | " + " | ".join(f"{value.mean():.3f} +/- {value.std():.3f}" for value in values) + " |")
    # Start the aggregate removal-quality table.
    lines += ["", "## Removal", "", "| k | round | precision | random precision | cumulative anomaly recall |",
              "|---|---|---|---|---|"]
    # Average each cleaning metric across seeds for every k and round.
    for k in args.k_values:
        for round_index in range(1, args.rounds + 1):
            # Select round records for this specific trajectory and round number.
            selected = [row for row in rounds if row["k"] == k and row["round"] == round_index]
            # Report average removal precision, random baseline, and cumulative recall.
            lines.append(f"| {k} | {round_index} | " + " | ".join(
                f"{np.mean([row[key] for row in selected]):.3f}" for key in
                ("precision", "random_precision", "recall_cumulative")) + " |")
    # Add prose translating counts and metrics into a per-k summary.
    lines += ["", "## What these results mean", ""]
    for k in args.k_values:
        # First and last rounds describe the starting and ending contamination.
        first = [row for row in rounds if row["k"] == k and row["round"] == 1]
        last = [row for row in rounds if row["k"] == k and row["round"] == args.rounds]
        # Keep all rounds for cumulative removals.
        selected = [row for row in rounds if row["k"] == k]
        # Average starting anomaly count across seeds.
        initial_anomalies = np.mean([row["anomalies_before"] for row in first])
        # Sum removed anomaly and normal counts over rounds, then average per seed.
        removed_anomalies = sum(row["anomalies_removed"] for row in selected) / len(first)
        removed_normal = sum(row["removed"] - row["anomalies_removed"] for row in selected) / len(first)
        # Average the final remaining anomaly count and total retained set size.
        remaining_anomalies = np.mean([row["anomalies_before"] - row["anomalies_removed"] for row in last])
        remaining_total = np.mean([row["before"] - row["removed"] for row in last])
        # Group model metrics for the three comparison stages.
        by_method = {method: [row for row in metrics if row["k"] == k and row["method"] == method]
                     for method in ("uncleaned", "iterative", "ceiling")}

        def mean_metric(method, key):
            # Return the across-seed mean for one metric and one evaluation stage.
            return np.mean([row[key] for row in by_method[method]])

        # Add text on the amount removed, remaining contamination, recall, and
        # descriptive accuracy/AUROC changes versus the comparison models.
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
    # Include every seed's per-round counts instead of only aggregate means.
    lines += ["", "## Every seed and round", "",
              "Counts refer to the full training set, even when plots are sampled.", "",
              "| Seed | k | Round | Before | Injected before | Removed | Injected removed | Normal removed | Injected left | Precision | Random precision | Cumulative recall |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    # Translate each recorded round dictionary into one Markdown table row.
    for row in rounds:
        lines.append(
            f"| {row['seed']} | {row['k']} | {row['round']} | {row['before']} | "
            f"{row['anomalies_before']} | {row['removed']} | {row['anomalies_removed']} | "
            f"{row['removed'] - row['anomalies_removed']} | {row['anomalies_before'] - row['anomalies_removed']} | "
            f"{row['precision']:.1%} | {row['random_precision']:.1%} | {row['recall_cumulative']:.1%} |"
        )
    # Add a table of removed/available images for each digit in every round.
    lines += ["", "## Removed by digit", "",
              "Each entry is removed / available before that round. Digits 0-6 are normal; 7-9 are injected.", "",
              "| Seed | k | Round | " + " | ".join(str(d) for d in NORMAL_DIGITS + ANOMALY_DIGITS) + " |",
              "|---|---|---|" + "---|" * 10]
    # Match each round with its per-digit counts and format removed / available.
    for row in rounds:
        group = [d for d in digits if d["seed"] == row["seed"] and d["k"] == row["k"] and d["round"] == row["round"]]
        lines.append(f"| {row['seed']} | {row['k']} | {row['round']} | " +
                     " | ".join(f"{d['removed']} / {d['available']}" for d in group) + " |")
    # Include raw evaluation metrics for every seed, k, and method combination.
    lines += ["", "## Per-seed evaluation", "",
              "| Seed | k | Method | Reference size | Injected remaining | Accuracy | AUROC | AP |",
              "|---|---|---|---|---|---|---|---|"]
    # Format one detailed result row per evaluation record.
    for row in metrics:
        lines.append(f"| {row['seed']} | {row['k']} | {row['method']} | {row['reference_count']} | "
                     f"{row['injected_remaining']} | {row['acc']:.3f} | {row['auc']:.3f} | {row['ap']:.3f} |")
    # List the output files and what analysis data each contains.
    lines += ["", "## Files", "", "- data/metrics.csv: per-seed evaluation metrics and retained anomaly counts.",
              "- data/rounds.csv: per-seed removal precision, current random baseline and cumulative recall.",
              "- data/digits.csv: available and removed counts for each digit, round, seed and k.",
              "- data/seed_NN/k_NN/round_NN_scores.csv: all current images, scores and removal decisions.",
              "- data/seed_NN/k_NN/*_umap.csv: projections, true digits, injection and removal status.",
              "- figures/seed_NN/k_NN/: four-panel overview, signal distributions and examples of removed digits.",
              "- config.json: full settings and source notebook."]
    # Join Markdown lines with newline characters and end the file with a newline.
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """Parse settings, create a run folder, and execute the full diagnostic."""
    # Use this module's docstring as the command-line help description.
    parser = argparse.ArgumentParser(description=__doc__)
    # Choose one or more k values; each gets an independent cleaning trajectory.
    parser.add_argument("--k-values", type=int, nargs="+", default=[5],
                        help="K values for independent cleaning trajectories and plots")
    # Number of independent random seeds/models to run.
    parser.add_argument("--seeds", type=int, default=5)
    # Training epochs for each autoencoder fit.
    parser.add_argument("--epochs", type=int, default=15)
    # Maximum number of normal training images in the initial pool.
    parser.add_argument("--pool-size", type=int, default=20000)
    # Target fraction of injected anomalies in the combined initial training pool.
    parser.add_argument("--inject-fraction", type=float, default=0.05)
    # Fraction of the original mixed pool removed at each cleaning round.
    parser.add_argument("--remove-fraction", type=float, default=0.02)
    # Number of iterative cleaning rounds.
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--score-mode", choices=("combined", "latent", "recon"), default="combined",
                        help="Use both ranks, latent kNN distance only, or reconstruction error only")
    parser.add_argument("--output-root", type=Path, default=AUTOENCODER / "Diagnostics runs",
                        help="Root directory for method folders")
    # Maximum UMAP points; zero means plot all points, otherwise removed/injected
    # examples are still always retained and only normal retained points are sampled.
    parser.add_argument("--umap-sample", type=int, default=0,
                        help="Maximum plotted points (0: all); removed and injected points are always included")
    # Parse command-line text into typed argument values.
    args = parser.parse_args()
    # Validate positive counts, unique k values, valid fractions, and sample size.
    if (args.seeds < 1 or args.epochs < 1 or args.pool_size < 1 or args.rounds != 3 or
            (args.umap_sample != 0 and args.umap_sample < 3) or not 0 < args.inject_fraction < 1 or
            not 0 < args.remove_fraction < 1 or any(k < 1 for k in args.k_values) or
            len(set(args.k_values)) != len(args.k_values)):
        # argparse.error prints usage and exits with a nonzero status.
        parser.error("Counts and k values must be positive, rounds must be 3, k values unique, fractions between 0 and 1, and UMAP sample 0 or >= 3")
    # Check UMAP availability before creating output artifacts or training.
    try:
        importlib.import_module("umap")
    except ModuleNotFoundError as error:
        # Re-raise unrelated missing imports; only translate a missing UMAP package.
        if error.name != "umap":
            raise
        # Tell the user the dependency and expected package to install.
        parser.error("UMAP is required: install umap-learn in the Python environment running this script")
    # Place timestamped run output in the Autoencoder diagnostics archive.
    output_dir = args.output_root / args.score_mode / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    # Create parent folders and this unique run folder.
    output_dir.mkdir(parents=True)
    # Capture all settings and initial status for provenance/recovery.
    config = {**vars(args), "output_root": str(args.output_root)} | {"normal_digits": NORMAL_DIGITS, "anomaly_digits": ANOMALY_DIGITS,
                          "latent_dim": 64, "source_notebook": str(NOTEBOOK), "status": "running"}
    # Keep the config path so status and elapsed time can be rewritten at completion.
    config_path = output_dir / "config.json"
    # Save the initial running configuration before starting the expensive work.
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    # Print the output path before model training starts.
    print(f"Saving diagnostics to {output_dir}", flush=True)
    # Start a monotonic timer for reliable elapsed-time measurement.
    started = time.monotonic()
    try:
        # Provide notebook code with the imported libraries and metrics it expects.
        namespace = {"np": np, "torch": torch, "nn": nn, "F": F, "random": random,
                     "average_precision_score": average_precision_score, "roc_auc_score": roc_auc_score}
        # Train/evaluate each seed and write the explanatory report.
        metrics, rounds, digits = run_experiment(args, output_dir, namespace)
        write_summary(output_dir, args, metrics, rounds, digits)
        write_results_text(output_dir, config, args.score_mode, metrics, rounds)
    except Exception as error:
        # Record the exception type/message, then re-raise so the run fails visibly.
        config.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        # Mark the run as successful only if both experiment and report completed.
        config["status"] = "completed"
    finally:
        # Always record elapsed time and the final status, including on failure.
        config["elapsed_seconds"] = round(time.monotonic() - started, 2)
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    # Print the final report path for convenient access.
    print(f"Finished: {output_dir / 'summary.md'}", flush=True)


# Run main() only when this file is executed directly, not when imported.
if __name__ == "__main__":
    main()