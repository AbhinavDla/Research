"""Compare three completed cleaning runs on their matched initial MNIST pools."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


NAMES = {"combined": "Combined ranks", "latent": "kNN distance only",
         "recon": "Reconstruction only"}
SETTINGS = ("k_values", "seeds", "epochs", "pool_size", "inject_fraction",
            "remove_fraction", "rounds", "umap_sample", "normal_digits",
            "anomaly_digits", "latent_dim")


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_initial_pool(run, seed, k):
    rows = load_csv(run / "data" / f"seed_{seed:02d}" / f"k_{k:02d}" /
                    "round_01_scores.csv")
    pool = {int(row["mnist_train_index"]): (
        int(row["digit"]), row["injected"], float(row["knn_distance"]),
        float(row["reconstruction_error_per_ink"]))
            for row in rows}
    if len(pool) != len(rows):
        raise ValueError(f"Duplicate MNIST indices in {run}, seed {seed}, k={k}")
    return pool


def write_results_text(run, config, mode, metrics, rounds):
    """Recreate the selected run-log format from the saved diagnostic tables."""
    lines = ["=" * 70,
             f"SCENARIO: Train on digits 0-6, inject 7-9  "
             f"(latent dim {config['latent_dim']}, score mode {mode})",
             "Reconstructed from saved diagnostics; this is not the original training log.",
             "=" * 70]
    for k in config["k_values"]:
        for seed in range(1, config["seeds"] + 1):
            selected = sorted((row for row in rounds if int(row["k"]) == k and
                               int(row["seed"]) == seed), key=lambda row: int(row["round"]))
            first = selected[0]
            initial_normals = int(first["before"]) - int(first["anomalies_before"])
            lines.extend(["", f"===== seed {seed}/{config['seeds']} | k={k} | {initial_normals} normal + "
                          f"{first['anomalies_before']} injected | removing {first['removed']} "
                          f"per round for {config['rounds']} rounds ====="])
            for row in selected:
                lines.append(f"      removed {row['removed']} | {row['anomalies_removed']} injected "
                             f"(precision {float(row['precision']):.1%}, "
                             f"chance would be {float(row['random_precision']):.1%})")
            for stage in ("iterative", "ceiling"):
                row = next(row for row in metrics if int(row["k"]) == k and
                           int(row["seed"]) == seed and row["method"] == stage)
                lines.append(f"  [{stage:<9}] n={int(row['reference_count']):6} | "
                             f"accuracy {float(row['acc']):.2%} | detection AUROC {float(row['auc']):.2%}")
        lines.extend(["", "=" * 70, f"RESULTS: Train on digits 0-6, inject 7-9 | k={k}",
                      "=" * 70, "Method                     Accuracy %    Detection AUROC %    Avg precision %"])
        by_stage = {}
        for stage, label in (("iterative", "Iterative cleaning"), ("ceiling", "Ceiling (no anomalies)")):
            ordered = sorted((row for row in metrics if int(row["k"]) == k and row["method"] == stage),
                             key=lambda row: int(row["seed"]))
            by_stage[stage] = np.array([[float(row[key]) for key in ("acc", "auc", "ap")]
                                        for row in ordered])
            values = by_stage[stage] * 100
            lines.append(f"{label:<24} " + "   ".join(
                f"{values[:, i].mean():.2f} +/- {values[:, i].std():.2f}" for i in range(3)))
        differences = (by_stage["iterative"] - by_stage["ceiling"]) * 100
        lines.extend(["", "Per-seed difference in percentage points (iterative cleaning minus ceiling):"])
        for i, label in enumerate(("Accuracy", "Detection AUROC", "Average precision")):
            lines.append(f"  {label:<20} {differences[:, i].mean():+.2f} +/- {differences[:, i].std():.2f}")
        selected = [row for row in rounds if int(row["k"]) == k]
        precisions = [np.mean([float(row["precision"]) for row in selected if int(row["round"]) == i])
                      for i in range(1, config["rounds"] + 1)]
        final = [row for row in selected if int(row["round"]) == config["rounds"]]
        hits = sum(int(row["anomalies_removed"]) for row in selected) / config["seeds"]
        initial = int(next(row for row in selected if int(row["round"]) == 1)["anomalies_before"])
        remaining = np.mean([(int(row["anomalies_before"]) - int(row["anomalies_removed"])) /
                             (int(row["before"]) - int(row["removed"])) for row in final])
        lines.extend(["", "REMOVAL QUALITY (did the suspicion score find the injected images?)",
                      "  Precision by round: " + ", ".join(f"{p:.1%}" for p in precisions),
                      f"  Injected images found after all rounds: {hits:.1f} of {initial} ({hits / initial:.1%})",
                      f"  Injected share left in the training set: {remaining:.2%}",
                      "", "The ceiling is the no-injected-anomaly reference; it is not a cleaning method."])
    (run / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare(runs, destination):
    configs = {mode: json.loads((path / "config.json").read_text(encoding="utf-8"))
               for mode, path in runs.items()}
    reference = configs["combined"]
    for mode, config in configs.items():
        if config["status"] != "completed" or config["score_mode"] != mode:
            raise ValueError(f"{mode} run is incomplete or has a different score mode")
        for key in SETTINGS:
            if config[key] != reference[key]:
                raise ValueError(f"{mode} setting {key} differs from combined")
    max_differences = {mode: [0.0, 0.0] for mode in ("latent", "recon")}
    for seed in range(1, reference["seeds"] + 1):
        for k in reference["k_values"]:
            original = read_initial_pool(runs["combined"], seed, k)
            for mode in ("latent", "recon"):
                candidate = read_initial_pool(runs[mode], seed, k)
                if (candidate.keys() != original.keys() or
                        any(candidate[index][:2] != original[index][:2] for index in original)):
                    raise ValueError(f"Initial pools differ: {mode}, seed {seed}, k={k}")
                for index in original:
                    differences = (abs(candidate[index][2] - original[index][2]),
                                   abs(candidate[index][3] - original[index][3]))
                    for signal, difference in enumerate(differences):
                        max_differences[mode][signal] = max(max_differences[mode][signal], difference)

    metrics = {mode: load_csv(path / "data" / "metrics.csv") for mode, path in runs.items()}
    rounds = {mode: load_csv(path / "data" / "rounds.csv") for mode, path in runs.items()}
    digits = {mode: load_csv(path / "data" / "digits.csv") for mode, path in runs.items()}
    for mode in NAMES:
        write_results_text(runs[mode], configs[mode], mode, metrics[mode], rounds[mode])
    result_rows = []
    for k in reference["k_values"]:
        for mode in NAMES:
            method_rounds = [row for row in rounds[mode] if int(row["k"]) == k]
            final = [row for row in method_rounds if int(row["round"]) == reference["rounds"]]
            method_metrics = [row for row in metrics[mode]
                              if int(row["k"]) == k and row["method"] == "iterative"]
            n = reference["seeds"]
            if len(final) != n or len(method_metrics) != n or len(method_rounds) != n * reference["rounds"]:
                raise ValueError(f"Incomplete rows for {mode}, k={k}")
            anomalies = sum(int(row["anomalies_removed"]) for row in method_rounds) / n
            normals = sum(int(row["removed"]) - int(row["anomalies_removed"])
                          for row in method_rounds) / n
            result_rows.append({
                "k": k, "method": mode,
                "anomalies_removed_mean": anomalies, "normals_removed_mean": normals,
                "overall_precision": anomalies / (anomalies + normals),
                "recall": np.mean([float(row["recall_cumulative"]) for row in final]),
                "final_contamination": np.mean([
                    (int(row["anomalies_before"]) - int(row["anomalies_removed"])) /
                    (int(row["before"]) - int(row["removed"])) for row in final]),
                "accuracy": np.mean([float(row["acc"]) for row in method_metrics]),
                "detection_auroc": np.mean([float(row["auc"]) for row in method_metrics]),
                "detection_ap": np.mean([float(row["ap"]) for row in method_metrics]),
            })

    destination.mkdir(parents=True, exist_ok=True)
    data = destination / "data"
    data.mkdir(exist_ok=True)
    with (data / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    initial_anomalies = int(rounds["combined"][0]["anomalies_before"])
    removal_count = int(rounds["combined"][0]["removed"])
    lines = [
        "# MNIST autoencoder cleaning: score comparison", "",
        "## Executive summary", "",
        f"Three scoring methods were compared using the same {reference['seeds']} initial MNIST pools. "
        f"Each pool has {reference['pool_size']:,} normal digits 0-6 and {initial_anomalies:,} injected digits 7-9. "
        f"The models use {reference['latent_dim']} latent dimensions, {reference['epochs']} epochs "
        f"and k values {reference['k_values']}; {reference['rounds']} rounds each remove "
        f"{removal_count:,} images. The combined run was previously completed and its saved data was "
        "reformatted, not retrained. The two single-score methods were retrained separately.",
        "Seed numbers 1-5 are run labels; their initial sampling seeds are 0-4. "
        "For zero-based run index s, training uses seeds 100*s + round index "
        "(0-3), and 100*s + 92 for the normal-only ceiling.",
        "",
        "The initial MNIST indices and labels have been checked for equality across all three "
        "methods for each seed. Later pools differ because the methods remove different images. "
        + ("The saved first-round kNN and reconstruction signals also match exactly across methods. "
           if all(value == 0 for differences in max_differences.values() for value in differences)
           else "The saved first-round signals are not exactly equal across methods. ")
        + "Equal scores do not prove that the original model weights were identical. "
        "The combined run does not include a historical notebook hash or complete environment "
        "record, so exact numerical reproducibility of its trained models cannot be established.",
        "",
        "| k | Scoring method | Injected removed | Normal removed | Removal precision | "
        "Anomaly recall | Remaining contamination |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        lines.append(f"| {row['k']} | {NAMES[row['method']]} | {row['anomalies_removed_mean']:.1f} | "
                     f"{row['normals_removed_mean']:.1f} | {row['overall_precision']:.1%} | "
                     f"{row['recall']:.1%} | {row['final_contamination']:.2%} |")
    for k in reference["k_values"]:
        ranked = sorted((row for row in result_rows if row["k"] == k),
                        key=lambda row: row["anomalies_removed_mean"], reverse=True)
        lines.extend(["", f"For k={k}, {NAMES[ranked[0]['method']]} removed the most injected "
                      f"images on average ({ranked[0]['anomalies_removed_mean']:.1f} per seed). "
                      f"This is a descriptive result for these {reference['seeds']} seeds, not a guarantee "
                      "for new data."])
    lines.extend([
        "", "Counts are averages per seed across all three rounds. Removal precision is the "
        "fraction of all removed images that were injected; anomaly recall is the fraction of "
        "the original injected images removed. A high recall can still come at the cost of "
        "removing many normal images.", "",
        "## Round-by-round removal", "",
        "| k | Round | Scoring method | Injected removed | Normal removed | Removal precision |",
        "|---:|---:|---|---:|---:|---:|",
    ])
    for k in reference["k_values"]:
        for number in range(1, reference["rounds"] + 1):
            for mode in NAMES:
                selected = [row for row in rounds[mode]
                            if int(row["k"]) == k and int(row["round"]) == number]
                hits = np.mean([int(row["anomalies_removed"]) for row in selected])
                removed = np.mean([int(row["removed"]) for row in selected])
                lines.append(f"| {k} | {number} | {NAMES[mode]} | {hits:.1f} | "
                             f"{removed - hits:.1f} | {hits / removed:.1%} |")
    lines.extend([
        "", "A method can lead in the first round but not overall: after each round, "
        "the retained pools and retrained models differ.", "",
        "## Held-out evaluation", "",
        "| k | Scoring method | Normal digit accuracy | Detection AUROC | Detection AP |",
        "|---:|---|---:|---:|---:|",
    ])
    for row in result_rows:
        lines.append(f"| {row['k']} | {NAMES[row['method']]} | {row['accuracy']:.3f} | "
                     f"{row['detection_auroc']:.3f} | {row['detection_ap']:.3f} |")
    lines.extend([
        "", "Accuracy is kNN classification accuracy on normal held-out digits; AUROC and "
        "average precision (AP) measure anomaly ranking on held-out digits. The selected "
        "scoring signal also defines the detection score, so differences reflect both cleaning "
        "and the evaluation score, not cleaning alone. The test anomaly prevalence differs "
        f"from the {reference['inject_fraction']:.1%} training contamination, so AP is not directly comparable to removal precision.",
        "", "## Every seed", "",
        "| k | Seed | Method | Injected removed | Normal removed | Anomalies remaining |",
        "|---:|---:|---|---:|---:|---:|",
    ])
    for k in reference["k_values"]:
        for seed in range(1, reference["seeds"] + 1):
            for mode in NAMES:
                selected = [row for row in rounds[mode]
                            if int(row["k"]) == k and int(row["seed"]) == seed]
                last = next(row for row in selected if int(row["round"]) == reference["rounds"])
                hits = sum(int(row["anomalies_removed"]) for row in selected)
                total = sum(int(row["removed"]) for row in selected)
                left = int(last["anomalies_before"]) - int(last["anomalies_removed"])
                lines.append(f"| {k} | {seed} | {NAMES[mode]} | {hits} | {total - hits} | {left} |")
    lines.extend(["", "## Removals by digit", "",
                  "Numbers are average images removed per seed over all rounds; 0-6 are normal, "
                  "7-9 are injected. Differences reflect both prevalence and detector behavior.", "",
                  "| k | Method | " + " | ".join(map(str, range(10))) + " |",
                  "|---:|---|" + "---:|" * 10])
    for k in reference["k_values"]:
        for mode in NAMES:
            counts = [sum(int(row["removed"]) for row in digits[mode]
                          if int(row["k"]) == k and int(row["digit"]) == digit) /
                      reference["seeds"] for digit in range(10)]
            lines.append(f"| {k} | {NAMES[mode]} | " +
                         " | ".join(f"{value:.1f}" for value in counts) + " |")
    lines.extend([
        "", "## How to interpret the maps", "",
        "Each method folder contains a summary and one four-panel map for each seed. "
        "The first three panels show the set before each removal; black rings mark that "
        "round's removals. The fourth shows the retained set after final retraining. "
        "Shapes denote true digits, normal digits are grey, and injected digits are colored. "
        "Labels are only used to evaluate and illustrate results, not to choose removals.",
        "Each panel is an independently fitted UMAP, so its orientation, position and "
        "distances cannot be compared directly with another panel. The algorithm scores "
        "64-dimensional representations, not UMAP coordinates.", "",
        "## Caveats", "",
        f"The {reference['seeds']} seeds are a descriptive sample, not confidence intervals. "
        "The original combined model weights were not saved, and repeated GPU training can "
        "vary. Do not infer causation or generalize to other anomaly types from these MNIST "
        f"runs alone. The fixed removal budget always removes {removal_count * reference['rounds']:,} examples per seed, even "
        "if a method cannot confidently separate them.", "",
        "## Files", "",
        "- `data/comparison.csv`: aggregate numeric comparison.",
        "- `combined/original_archive` preserves the previously generated run unchanged.", "",
    ])
    for mode in NAMES:
        relative = Path("..") / runs[mode].relative_to(destination.parent) / "summary.md"
        lines.append(f"- [{NAMES[mode]} details]({relative.as_posix().replace(' ', '%20')}).")
    (destination / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison written to {destination / 'summary.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for method in NAMES:
        parser.add_argument(f"--{method}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare({mode: getattr(args, mode) for mode in NAMES}, args.output)
