"""Generate per-space UMAPs and voting diagnostics for the MNIST ensembles.

Each trained model makes its own predictions in its own latent space. UMAP maps a
sample of those latent vectors to 2D only for visualization; kNN and ensemble
voting use the original model outputs, not the plotted coordinates.
"""

# argparse reads command-line options such as the number of spaces to train.
import argparse
# csv writes the experiment's tabular outputs.
import csv
# importlib loads one of the model scripts by its file path at runtime.
import importlib.util
# json saves the run settings in a machine-readable config file.
import json
# random, numpy, and torch each have random-number generators that are seeded.
import random
# time measures how long the experiment takes.
import time
# Counter counts predicted labels to implement majority voting.
from collections import Counter
# datetime gives each run a unique timestamped output folder.
from datetime import datetime
# Path builds filesystem paths portably.
from pathlib import Path

# Matplotlib creates the maps and writes image files without opening a GUI.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# NumPy handles arrays, sampling, comparisons, and summary statistics.
import numpy as np
# PyTorch is used by both neural-network implementations.
import torch
# These utilities wrap tensors in batches for the autoencoder workflow.
from torch.utils.data import DataLoader, TensorDataset
# torchvision loads the local MNIST training and test datasets.
from torchvision import datasets
# UMAP reduces a model's high-dimensional latent vectors to two plot coordinates.
from umap.umap_ import UMAP


# ROOT is the directory containing this diagnostics script.
ROOT = Path(__file__).resolve().parent
# Map short command-line names to the folder and Python module for each model type.
METHODS = {
    "autoencoder": ("Unsupervised Autoencoder", "unsupervised_latent_ensemble.py"),
    "simclr": ("Semi-Supervised SimCLR", "semi_supervised_latent.py"),
}


def load_module(folder, filename):
    """Load a model implementation from a path beneath ROOT."""
    # Construct the exact model-script path from the selected method.
    path = ROOT / folder / filename
    # Build an import specification for a Python module stored at that path.
    spec = importlib.util.spec_from_file_location(path.stem, path)
    # A missing spec or loader means Python cannot import this file by path.
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load model script: {path}")
    # Create a module object from the import specification.
    module = importlib.util.module_from_spec(spec)
    # Execute the model script so its classes, functions, and settings are defined.
    spec.loader.exec_module(module)
    # Return the loaded module so run() can configure and call it.
    return module


def write_rows(path, rows):
    """Write dictionaries with matching keys as a headered CSV file."""
    # Open in text mode; newline="" avoids extra blank rows on some platforms.
    with path.open("w", newline="", encoding="utf-8") as file:
        # Use the first row's keys as the CSV column names.
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        # Write one header row followed by one row for each dictionary.
        writer.writeheader()
        writer.writerows(rows)


def plot_space(path, features, labels, predictions, indices, seed, title):
    """Plot and export one model's sampled test-image latent representations.

    The coordinates from UMAP are for the figure and CSV only. Predictions were
    already made with kNN in the original latent feature space.
    """
    # Fit a fresh UMAP projection to the selected test features for this model.
    # n_neighbors is capped at 15 and kept below the number of sampled points.
    # random_state makes this projection repeatable; n_jobs=1 supports that
    # repeatability. This does not align this model's plot with other models'.
    coords = np.asarray(UMAP(n_neighbors=min(15, len(indices) - 1), random_state=seed,
                             n_jobs=1).fit_transform(features[indices]))
    # Compare each selected model prediction with its true digit label.
    correct = predictions[indices] == labels[indices]
    # Create two panels: one colored by ground-truth digit, one by correctness.
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    # tab10 assigns a distinct categorical color to each of the ten digits.
    colors = plt.get_cmap("tab10")
    # Draw each true digit as a separate group in the left-hand map.
    for digit in range(10):
        # Select the plotted rows whose true label is this digit.
        mask = labels[indices] == digit
        # Each point is one image's 2D UMAP coordinate; the legend reports its count.
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.55,
                        color=colors(digit), label=f"{digit} ({mask.sum()})", rasterized=True)
    # Show digit labels in a compact two-column legend.
    axes[0].legend(title="True digit", ncol=2, markerscale=2)
    # Draw correct and incorrect predictions as separate groups on the right.
    for mask, label, color in [(correct, "Correct", "#a9a9a9"),
                               (~correct, "Incorrect", "#d62728")]:
        # The same UMAP coordinates are reused so correctness is overlaid on the
        # identical projection as the true-digit panel.
        axes[1].scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6,
                        color=color, label=f"{label} ({mask.sum()})", rasterized=True)
    # Label the two correctness groups.
    axes[1].legend(markerscale=2)
    # Give each panel a title describing the information encoded by its color.
    axes[0].set_title("True digit")
    axes[1].set_title("Individual space: kNN predictions")
    # Both axes are the two UMAP coordinates, not two original latent dimensions.
    for axis in axes:
        axis.set(xlabel="UMAP 1", ylabel="UMAP 2")
    # Include the model/method and number of plotted held-out images above both panels.
    fig.suptitle(f"{title} | {len(indices)} held-out test images")
    # Save the figure and close it to release memory before the next space.
    fig.savefig(path / "umap.png", dpi=150)
    plt.close(fig)
    # Save the plotted point's test-set index, true label, prediction, and 2D position.
    write_rows(path / "umap.csv", [
        {"test_index": int(index), "digit": int(labels[index]),
         "prediction": int(predictions[index]), "umap_1": float(point[0]),
         "umap_2": float(point[1])}
        # zip pairs each selected original test index with its projected point.
        for index, point in zip(indices, coords)
    ])


def run(method, args):
    """Train the requested independent spaces, evaluate them, and save diagnostics."""
    # Look up which source folder and model implementation belong to this method.
    folder, filename = METHODS[method]
    # Load that implementation as a module so its model-specific helpers are usable.
    model_module = load_module(folder, filename)
    # Override the model script's defaults with this run's command-line settings.
    setattr(model_module, "NUM_MODELS", args.spaces)
    setattr(model_module, "LATENT_DIM", args.dimensions)
    # Only override epochs when the user explicitly supplied --epochs.
    if args.epochs is not None:
        setattr(model_module, "EPOCHS", args.epochs)
    # Reject invalid training settings before loading data or creating outputs.
    if args.spaces < 1 or args.dimensions < 1 or model_module.EPOCHS < 1:
        raise ValueError("spaces, dimensions, and epochs must be positive")

    # Preserve the naming convention used by the existing model iteration folders.
    spaces_folder = f"SimCLR {args.spaces} Spaces" if method == "simclr" else f"{args.spaces} Spaces"
    # Use a caller-provided output root when present; otherwise store under Iterations.
    base = args.output_root if args.output_root else ROOT / folder / "Iterations"
    if method == "simclr":
        base = base / model_module.reference_folder_name(model_module.REFERENCE_LABELS_PER_CLASS)
    # Separate outputs by number of spaces, latent width, and timestamped run.
    output = (base / spaces_folder /
              f"{args.dimensions} Latent Dimensions" / "Diagnostics" /
              datetime.now().strftime("run_%Y%m%d_%H%M%S_%f"))
    # Create this run's directory; the timestamp is expected to make it unique.
    output.mkdir(parents=True)
    # Start elapsed-time measurement before loading MNIST and training models.
    started = time.time()
    # MNIST files are expected in the repository-level data directory.
    data_root = ROOT.parent / "data"
    # Load the standard 60,000-image MNIST training split without downloading.
    train = datasets.MNIST(data_root, train=True, download=False)
    # Load the standard 10,000-image MNIST held-out test split without downloading.
    test = datasets.MNIST(data_root, train=False, download=False)
    # Convert image byte values (0..255) to float pixel values (0..1), then add
    # a channel dimension so images have the shape expected by the model scripts.
    train_x = train.data.float().div(255).unsqueeze(1)
    # Keep only the first test_limit test images, with the same scaling/shape.
    test_x = test.data.float().div(255).unsqueeze(1)[:args.test_limit]
    # Keep labels corresponding exactly to the selected test images.
    test_y = test.targets[:len(test_x)].numpy()
    # The visualization and evaluation need at least three test examples.
    if args.umap_sample < 3 or len(test_x) < 3:
        raise ValueError("At least three test images and UMAP samples are required")
    # Select one fixed, sorted sample of test indices for every model's map.
    # This makes plots cover the same images; it does not make their UMAP
    # coordinate systems comparable because every model is projected separately.
    indices = np.sort(np.random.RandomState(42).choice(
        len(test_x), min(args.umap_sample, len(test_x)), replace=False))
    # Record settings and dataset sizes so the run can be interpreted later.
    config = {"method": method, "spaces": args.spaces, "latent_dimensions": args.dimensions,
              "epochs": model_module.EPOCHS, "batch_size": model_module.BATCH_SIZE,
              "k": model_module.K_NEIGHBORS, "train_images": len(train_x),
              "reference_images": (len(train_x) if method == "autoencoder"
                                   or model_module.REFERENCE_LABELS_PER_CLASS is None
                                   else 10 * model_module.REFERENCE_LABELS_PER_CLASS),
              "test_images": len(test_x), "umap_points_per_space": len(indices),
              # Each independent model receives its own training seed.
              "seeds": list(range(1000, 1000 + args.spaces)),
              "device": str(model_module.DEVICE), "status": "running"}
    # Write an initial config so settings remain available even if training fails.
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # Print the location and hardware device for progress visibility.
    print(f"\n{method}: output {output} | device {model_module.DEVICE}", flush=True)

    # Initialize branch-specific data objects; unused objects remain None.
    train_loader = reference_loader = test_loader = None
    reference_indices = None
    if method == "autoencoder":
        # Pair each training image with its label for the labeled kNN reference.
        train_data = TensorDataset(train_x, train.targets)
        # Batch test images in stable order; labels are carried for evaluation.
        test_loader = DataLoader(TensorDataset(test_x, test.targets[:len(test_x)]),
                                 batch_size=model_module.BATCH_SIZE, shuffle=False)
        # Shuffle training batches to train the autoencoder.
        train_loader = DataLoader(train_data, batch_size=model_module.BATCH_SIZE, shuffle=True)
        # Visit training examples in stable order when extracting kNN references.
        reference_loader = DataLoader(train_data, batch_size=model_module.BATCH_SIZE, shuffle=False)
    else:
        # Labels are used only for the kNN reference set, not encoder training.
        reference_indices = model_module.choose_labeled_indices(
            train.targets, model_module.REFERENCE_LABELS_PER_CLASS)

    # Keep every model's predictions so they can be combined after training.
    all_predictions = []
    # Store one accuracy record per independently trained model.
    metrics = []
    for number in range(args.spaces):
        # Seed is unique per model, so spaces have independent training randomness.
        seed = 1000 + number
        # Seed Python's standard-library random generator.
        random.seed(seed)
        # Seed NumPy's random generator.
        np.random.seed(seed)
        # Seed PyTorch's random generator.
        torch.manual_seed(seed)
        # Report which model is being trained.
        print(f"\n{method}: space {number + 1}/{args.spaces}, seed {seed}", flush=True)
        if method == "autoencoder":
            # Ensure the autoencoder data loaders were created above.
            if train_loader is None or reference_loader is None or test_loader is None:
                raise RuntimeError("Autoencoder data loaders were not initialized")
            # Create a fresh autoencoder and move its weights to the chosen device.
            model = model_module.Autoencoder(args.dimensions).to(model_module.DEVICE)
            # Train to reconstruct input images; labels are not its training target.
            model_module.train_autoencoder(model, train_loader)
            # Encode labeled training examples: their vectors and labels are kNN references.
            references, ref_labels = model_module.extract_latent_features(model, reference_loader)
            # Encode held-out test examples; these are the query vectors to classify.
            features, _ = model_module.extract_latent_features(model, test_loader)
            # Predict each query using neighbors in this model's full latent space.
            predictions = model_module.batched_knn_predictions(
                features, references, ref_labels, model_module.K_NEIGHBORS)
        else:
            # Ensure SimCLR's labeled reference indices were selected.
            if reference_indices is None:
                raise RuntimeError("SimCLR reference indices were not initialized")
            # Randomly select the unlabeled images used to train this SimCLR model.
            subset = torch.randperm(len(train_x))[:model_module.TRAIN_SUBSET]
            # Create a fresh SimCLR encoder and move it to the selected device.
            model = model_module.SimCLRModel(args.dimensions).to(model_module.DEVICE)
            # Train the representation learner on the selected training subset.
            model_module.train_simclr(model, train_x[subset])
            # Encode the labeled reference images in this model's latent space.
            references = model_module.encode(model, train_x[reference_indices])
            # Encode held-out test images in the same latent space.
            features_tensor = model_module.encode(model, test_x)
            # Classify using kNN against reference vectors and their known labels.
            predictions = model_module.knn_predict(
                features_tensor, references, train.targets[reference_indices],
                model_module.K_NEIGHBORS).numpy()
            # Convert test vectors to NumPy for UMAP and map plotting.
            features = features_tensor.cpu().numpy()
        # Store this model's predictions as one column in the eventual vote table.
        all_predictions.append(predictions)
        # Measure the fraction of all selected test images classified correctly.
        accuracy = float(np.mean(predictions == test_y))
        # Save identifying information and accuracy for this individual model.
        metrics.append({"space": number + 1, "seed": seed, "accuracy": accuracy})
        # Create a separate folder for this model's map.
        space_dir = output / f"space_{number + 1:02d}"
        space_dir.mkdir()
        # Plot the same sampled test cases for this space, colored by truth/outcome.
        plot_space(space_dir, features, test_y, predictions, indices, 42,
                   f"{method} | space {number + 1}, seed {seed}")
        # Report accuracy and total elapsed time so far.
        print(f"  accuracy {accuracy:.4%}; elapsed {(time.time() - started) / 60:.1f} min",
              flush=True)

    # Shape becomes (number of test images, number of spaces); each row contains
    # one image's predicted digit from every independently trained model.
    stacked = np.stack(all_predictions, axis=1)
    if method == "autoencoder":
        # Counter preserves first-seen order, so a tied vote chooses the earliest
        # space's prediction for that image.
        ensemble = np.array([Counter(row).most_common(1)[0][0] for row in stacked])
    else:
        # bincount counts votes for digits 0..9; argmax chooses the smallest digit
        # when vote totals tie.
        ensemble = np.array([np.bincount(row, minlength=10).argmax() for row in stacked])
    # Count how many spaces voted for each possible digit for every test image.
    counts = np.stack([(stacked == digit).sum(axis=1) for digit in range(10)], axis=1)
    # Record the number of votes received by the winning digit on each image.
    winning_votes = counts.max(axis=1)
    # A row is tied when more than one digit has the maximum vote count.
    tied = (counts == winning_votes[:, None]).sum(axis=1) > 1
    # Save the individual model accuracies and seeds.
    write_rows(output / "spaces.csv", metrics)
    # Save all test images' true labels, ensemble decisions, and per-space votes.
    write_rows(output / "predictions.csv", [
        {"test_index": i, "digit": int(label), "ensemble": int(ensemble[i]),
         "winning_votes": int(winning_votes[i]), "tie": bool(tied[i]),
         # Add a named column for each model's predicted digit.
         **{f"space_{j + 1:02d}": int(value) for j, value in enumerate(stacked[i])}}
        # Include every test image here, not only the sample displayed in UMAP.
        for i, label in enumerate(test_y)
    ])
    # Collect accuracy separately for each true MNIST digit.
    per_digit = []
    for digit in range(10):
        # Select test images whose ground-truth label is the current digit.
        mask = test_y == digit
        # Some small --test-limit values could omit a digit entirely.
        if mask.any():
            per_digit.append({"digit": digit, "test_images": int(mask.sum()),
                              # Accuracy among this digit's examples for the ensemble.
                              "ensemble_accuracy": float(np.mean(ensemble[mask] == digit)),
                              # Mean correctness across all spaces on these examples.
                              "mean_space_accuracy": float(np.mean(stacked[mask] == digit))})
    # Save the per-digit results as a separate table.
    write_rows(output / "digits.csv", per_digit)
    # Summarize mean and population standard deviation of individual-space accuracy.
    mean = float(np.mean([row["accuracy"] for row in metrics]))
    std = float(np.std([row["accuracy"] for row in metrics]))
    # Measure the ensemble's accuracy on the same held-out test images.
    ens = float(np.mean(ensemble == test_y))
    # Boolean table: row per test image, column per space, true when prediction is right.
    correct_by_space = stacked == test_y[:, None]
    # Count images every model predicted correctly.
    all_correct = int(correct_by_space.all(axis=1).sum())
    # Count images every model predicted incorrectly.
    all_wrong = int((~correct_by_space).all(axis=1).sum())
    # Mark whether the ensemble got each test image right.
    ensemble_correct = ensemble == test_y
    # Mark whether space 1 got each test image right for the comparison below.
    first_correct = correct_by_space[:, 0]
    # Count images the ensemble gets right when space 1 gets them wrong.
    ensemble_only = int((ensemble_correct & ~first_correct).sum())
    # Count images space 1 gets right when the ensemble gets them wrong.
    first_only = int((~ensemble_correct & first_correct).sum())
    # Describe the kNN reference examples used by this method.
    reference = f"{config['reference_images']:,} labeled training images"
    # State the applicable tie-break behavior in the human-readable report.
    tie_rule = ("the earliest space's tied prediction" if method == "autoencoder"
                else "the smallest tied digit")
    # Assemble the report as Markdown lines; it explains methods and interprets metrics.
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
        # The prose describes what the plot encodes and warns against comparing
        # separately fitted UMAP coordinate systems.
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
    # Add a table row for each individual space's accuracy.
    summary.extend(f"| {r['space']} | {r['seed']} | {r['accuracy']:.2%} |" for r in metrics)
    # Start the per-digit table in the report.
    summary.extend(["", "## Per-digit results", "", "| Digit | Test images | Mean space accuracy | Ensemble accuracy |",
                    "|---:|---:|---:|---:|"])
    # Add one row per digit with sample count and both kinds of accuracy.
    summary.extend(f"| {r['digit']} | {r['test_images']} | {r['mean_space_accuracy']:.2%} | "
                   f"{r['ensemble_accuracy']:.2%} |" for r in per_digit)
    # Document the artifacts produced by this run.
    summary.extend(["", "## Files", "",
                    "- `space_NN/umap.png` and `umap.csv`: map and coordinates for that space.",
                    "- `spaces.csv`: individual accuracies and seeds.",
                    "- `predictions.csv`: every test image's individual and ensemble predictions, "
                    "winning vote count, and tie flag.",
                    "- `digits.csv`: per-digit accuracy; `config.json`: run settings.",
                    ""])
    # Preserve the existing behavior: only autoencoder runs write summary.md.
    if method == "autoencoder":
        (output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    # Mark the config complete and include total wall-clock runtime.
    config.update(status="completed", elapsed_seconds=round(time.time() - started, 2))
    # Save final status over the initial config file.
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # Print the completed run folder.
    print(f"Completed {method}: {output}", flush=True)


def main():
    """Parse command-line options and run one or both model families."""
    # Set the command description from this module's docstring.
    parser = argparse.ArgumentParser(description=__doc__)
    # Choose one model type or run both; default runs each separately.
    parser.add_argument("--method", choices=("autoencoder", "simclr", "both"), default="both")
    # Number of independently initialized/trained models in each family.
    parser.add_argument("--spaces", type=int, default=10)
    # Width of each model's latent vector (and each point's original feature size).
    parser.add_argument("--dimensions", type=int, default=64)
    # Optional training-epoch override; None means use the model script's default.
    parser.add_argument("--epochs", type=int)
    # Number of held-out test examples to plot for each model's UMAP.
    parser.add_argument("--umap-sample", type=int, default=10000)
    # Limit evaluation to the first N test examples; defaults to the full test set.
    parser.add_argument("--test-limit", type=int, default=10000)
    # Optional alternate output folder, useful for test or temporary runs.
    parser.add_argument("--output-root", type=Path, help="Override output location (e.g. for test runs)")
    # Convert command-line text into typed Python settings.
    args = parser.parse_args()
    # Expand "both" into the two separate runs; otherwise run only the selected method.
    for method in METHODS if args.method == "both" else (args.method,):
        # Train, evaluate, plot, and save diagnostics for this model family.
        run(method, args)


# Run main() only when this file is executed as a script, not when imported.
if __name__ == "__main__":
    main()
