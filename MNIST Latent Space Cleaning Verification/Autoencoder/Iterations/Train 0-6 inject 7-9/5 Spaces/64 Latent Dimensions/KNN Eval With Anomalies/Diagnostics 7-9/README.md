# Autoencoder cleaning diagnostics

These diagnostics compare three scoring methods:

- `combined`: the mean of the kNN-distance and reconstruction-error ranks.
- `latent`: the kNN-distance rank alone.
- `recon`: the reconstruction-error rank alone.

All three use the same notebook model, MNIST selection, seeds, training,
evaluation and fixed removal budget. The score determines both removal and
the reported anomaly-detection ranking. The runs are separate retrainings:
identical seeds do not guarantee identical weights across software versions or
devices. The original combined run was reformatted from its saved coordinates;
it was not retrained.

Default settings are normal digits 0-6, injected digits 7-9, five seeds,
64 latent dimensions, 15 epochs, k=5, 20,000 normal images, 5% initial
contamination and three rounds removing 2% of the original mixed pool per
round. Test anomaly prevalence is different from training contamination.

Each method has a folder under `Autoencoder\Diagnostics runs`. A run's
`summary.md` explains its results; `results.txt` reproduces the selected
experiment log's layout from the saved
diagnostic tables rather than claiming to be the original console output.
Under `figures\seed_NN\k_05`, the
`cleaning_overview.png` shows all three pre-removal stages and the final
retained set in a 2-by-2 image. Marker shape denotes digit, normal digits
are grey, injected digits are colored, and black rings mark points removed
in that round. Labels are only used for analysis, not scoring or removal.
Counts in the image are plotted counts. UMAP is independently fitted at each
stage: positions and distances across panels cannot be compared. The 64-D
features, not the 2-D projection, are used for scoring.

The `data` folder holds aggregate metrics, removal counts, and per-image
scores and UMAP coordinates. The `figures` folder also contains score
distributions and example removed images. UMAP plots include all points by
default; `--umap-sample 5000` samples retained normals while keeping all
injected and removed examples, so the plotted count may exceed 5,000.

From the repository root, with local MNIST data and PyTorch, torchvision,
NumPy, matplotlib, scikit-learn and `umap-learn` installed:

```powershell
python "MNIST Latent Space Cleaning Verification\Autoencoder\Iterations\Train 0-6 inject 7-9\5 Spaces\64 Latent Dimensions\KNN Eval With Anomalies\Diagnostics 7-9\run_diagnostics.py" --score-mode latent
python "MNIST Latent Space Cleaning Verification\Autoencoder\Iterations\Train 0-6 inject 7-9\5 Spaces\64 Latent Dimensions\KNN Eval With Anomalies\Diagnostics 7-9\run_diagnostics.py" --score-mode recon
```

The runner uses CUDA when available, otherwise MPS or CPU; UMAP runs on the
CPU. Each run creates a timestamped folder and records its settings and
completion status in `config.json`. Use `--help` for the other settings. The
four-panel output requires exactly three rounds.
