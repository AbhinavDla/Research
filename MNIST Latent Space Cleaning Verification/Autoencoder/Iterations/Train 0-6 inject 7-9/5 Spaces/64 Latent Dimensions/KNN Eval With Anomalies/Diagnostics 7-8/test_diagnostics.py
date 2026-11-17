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


if __name__ == "__main__":
    unittest.main()
