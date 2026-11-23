import csv
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import run_diagnostics as diagnostics


class FakeUMAP:
    def __init__(self, **kwargs):
        pass

    def fit_transform(self, values):
        return values[:, :2]


class DiagnosticsTests(unittest.TestCase):
    def test_sample_always_includes_removed_and_all_anomaly_digits(self):
        labels = np.arange(10)
        removed = np.zeros(10, dtype=bool)
        removed[0] = True
        latents = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "projection"
            original_import = diagnostics.importlib.import_module
            def import_module(name, package=None):
                if name == "umap":
                    return types.SimpleNamespace(UMAP=FakeUMAP)
                return original_import(name, package)
            with patch.object(diagnostics.importlib, "import_module", side_effect=import_module):
                diagnostics.make_umap(latents, labels, np.arange(10), removed,
                                      path, "test", sample_size=3, seed=0)
            with path.with_suffix(".csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual({int(row["digit"]) for row in rows}, {0, 7, 8, 9})
            self.assertEqual({int(row["digit"]) for row in rows if row["injected"] == "True"},
                             {7, 8, 9})
            self.assertEqual([int(row["digit"]) for row in rows if row["removed_this_round"] == "True"],
                             [0])

    def test_overview_reads_all_four_saved_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            data.mkdir()
            fields = ["mnist_train_index", "digit", "injected", "removed_this_round",
                      "umap_1", "umap_2"]
            for name in ("round_01_umap", "round_02_umap", "round_03_umap", "final_umap"):
                with (data / f"{name}.csv").open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(dict(mnist_train_index=1, digit=2, injected=False,
                                         removed_this_round=name != "final_umap", umap_1=0, umap_2=1))
                    writer.writerow(dict(mnist_train_index=2, digit=8, injected=True,
                                         removed_this_round=False, umap_1=1, umap_2=0))
            output = Path(directory) / "figures" / "overview.png"
            diagnostics.make_overview(data, output, 1, 5, "latent")
            self.assertGreater(output.stat().st_size, 0)

    def test_notebook_score_modes_rank_only_the_selected_signal(self):
        namespace = {"np": np, "torch": torch, "nn": torch.nn}
        diagnostics.load_original_functions(namespace)
        knn = np.array([3.0, 1.0, 2.0])
        recon = np.array([1.0, 3.0, 2.0])
        namespace["SCORE_MODE"] = "latent"
        np.testing.assert_array_equal(namespace["pick_score"](knn, recon), [1, 0, .5])
        namespace["SCORE_MODE"] = "recon"
        np.testing.assert_array_equal(namespace["pick_score"](knn, recon), [0, 1, .5])
        namespace["SCORE_MODE"] = "combined"
        np.testing.assert_array_equal(namespace["pick_score"](knn, recon), [.5, .5, .5])


if __name__ == "__main__":
    unittest.main()
