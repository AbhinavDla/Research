"""Reformat an existing combined-score run without retraining or changing its data."""

import argparse
import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import run_diagnostics


def read_rows(path, integer_columns):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key, value in row.items():
            if key == "method":
                continue
            row[key] = int(value) if key in integer_columns else float(value)
    return rows


def reorganize(source, destination):
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config["status"] != "completed" or config["rounds"] != 3:
        raise ValueError("Expected a completed combined run with exactly three rounds")
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    data = destination / "data"
    data.mkdir()
    for name in ("metrics.csv", "rounds.csv", "digits.csv"):
        shutil.copy2(source / name, data / name)
    for seed in range(1, config["seeds"] + 1):
        for k in config["k_values"]:
            relative = Path(f"seed_{seed:02d}") / f"k_{k:02d}"
            old = source / relative
            new_data = data / relative
            new_data.mkdir(parents=True)
            for name in ("round_01_umap.csv", "round_02_umap.csv", "round_03_umap.csv",
                         "final_umap.csv", "round_01_scores.csv", "round_02_scores.csv",
                         "round_03_scores.csv"):
                shutil.copy2(old / name, new_data / name)
            figures = destination / "figures" / relative
            for round_number in range(1, 4):
                round_figures = figures / f"round_{round_number:02d}"
                shutil.copytree(old / f"round_{round_number:02d}", round_figures)
                with (new_data / f"round_{round_number:02d}_scores.csv").open(
                        newline="", encoding="utf-8") as file:
                    rows = list(csv.DictReader(file))
                run_diagnostics.plot_score_diagnostics(
                    np.array([row["injected"].lower() == "true" for row in rows]),
                    np.array([row["removed"].lower() == "true" for row in rows]),
                    np.array([float(row["knn_distance"]) for row in rows]),
                    np.array([float(row["reconstruction_error_per_ink"]) for row in rows]),
                    np.array([float(row["combined_rank_score"]) for row in rows]),
                    round_figures, "combined")
            run_diagnostics.make_overview(new_data, figures / "cleaning_overview.png",
                                          seed, k, "combined")
    config["score_mode"] = "combined"
    config["original_run"] = str(source)
    config["presentation_note"] = "Reformatted from the original CSVs; no model was retrained."
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    metrics = read_rows(data / "metrics.csv", {"seed", "k", "reference_count", "injected_remaining"})
    rounds = read_rows(data / "rounds.csv", {"seed", "k", "round", "before", "anomalies_before",
                                              "removed", "anomalies_removed"})
    digits = read_rows(data / "digits.csv", {"seed", "k", "round", "digit", "available", "removed"})
    run_diagnostics.write_summary(destination, SimpleNamespace(**config), metrics, rounds, digits)
    run_diagnostics.write_results_text(destination, config, "combined", metrics, rounds)
    print(f"Reformatted combined results: {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    reorganize(arguments.source, arguments.destination)
