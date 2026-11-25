# Averaging diagnostics

The two experiments combine predictions by voting; they do not average latent
coordinates. Run the diagnostics from the repository root with a Python
environment containing PyTorch, torchvision, NumPy, matplotlib and umap-learn:

```powershell
python "MNIST Latent Space Averaging Verification\run_averaging_diagnostics.py"
```

MNIST must already be present in the repository's `data\MNIST\raw` directory.
The default runs 10 spaces with 64 dimensions. SimCLR uses CUDA when available,
then Apple MPS, then CPU; the autoencoder uses CUDA when available, otherwise
CPU. Training and kNN use the selected PyTorch device; UMAP runs on the CPU.
Install a missing UMAP dependency in your active environment with
`python -m pip install umap-learn`.

Each method's results are stored under its own `Iterations` directory, in a
folder named for the space count and latent dimensions, with a unique run
directory. Each of the 10 maps shows held-out test digits and individual-space
classification errors. CSV files record predictions, accuracies and coordinates.
The autoencoder run also generates a plain-language `summary.md`. SimCLR's
existing summary is left unchanged.

Both methods train their encoders without digit labels and use all 60,000 labeled
training images as the kNN reference set. The test images remain held out.
SimCLR's previous 50-labels-per-class results are under
`Semi-Supervised SimCLR/Iterations/50 Labels Per Class/`; new runs use
`Semi-Supervised SimCLR/Iterations/All Training Labels/`. The standalone SimCLR
script also saves each run in a separate timestamped directory there.

Use `--method autoencoder` or `--method simclr` to run just one method.
`--spaces` and `--dimensions` change the experiment and output folder names.
For other options, run the command with `--help`.
