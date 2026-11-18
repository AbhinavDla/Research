<<<<<<< HEAD
<<<<<<< HEAD
# Iterative cleaning diagnostics: inject 7-9

The folder's historical name says `Diagnostics 7-8`, but this experiment uses
digits **7, 8 and 9** as anomalies and digits **0-6** as normal. It reruns the
model, scoring and contaminated-reference evaluation from the original **KNN
With Anomalies** notebook; it cannot recreate the original run's exact trained
models because those models were not saved.

Run from the workspace root in a Python environment with PyTorch, torchvision,
NumPy, matplotlib, scikit-learn and `umap-learn` installed:

```powershell
python "MNIST Latent Space Cleaning Verification\Autoencoder\Iterations\Train 0-6 inject 7-9\5 Spaces\64 Latent Dimensions\KNN Eval With Anomalies\Diagnostics 7-8\run_diagnostics.py"
```

MNIST must already be present in a parent directory's `data\MNIST` folder, and
the source notebook must remain in `Autoencoder\Versions\KNN With Anomalies`.
If UMAP or one of its dependencies is missing, install `umap-learn` in the
environment used to run the script. Its dependency versions must be compatible
with that environment.

On this machine, the following creates a separate environment using the
existing system packages and installs a NumPy version compatible with its Numba:

```powershell
python -m venv --system-site-packages .venv-diagnostics
.\.venv-diagnostics\Scripts\python.exe -m pip install "numpy<2.5" umap-learn
.\.venv-diagnostics\Scripts\python.exe "MNIST Latent Space Cleaning Verification\Autoencoder\Iterations\Train 0-6 inject 7-9\5 Spaces\64 Latent Dimensions\KNN Eval With Anomalies\Diagnostics 7-8\run_diagnostics.py"
```

The autoencoder and k-nearest-neighbor calculations use CUDA when the installed
PyTorch can access a CUDA GPU, otherwise MPS or CPU. UMAP runs on the CPU.
The script writes the summary automatically when the run finishes.

The defaults use five seeds (five independently trained spaces), 64 latent
dimensions, 15 epochs, k=5, and three cleaning rounds. They produce **20 UMAP
figures**: three before-removal figures and one final retained-set figure for
each seed. Each UMAP figure has two views of the same projection: digit colors
and removal outcomes. Further score and example figures are not UMAPs. The
default run is computationally expensive.

Each invocation creates a timestamped `Autoencoder\Diagnostics runs\run_...`
directory (shorter to avoid Windows path-length errors), containing
`summary.md`, `config.json`, per-seed metrics, per-round statistics and counts
by digit. Open the summary for a plain-language explanation, the per-seed and
per-round outcomes, and definitions of the measurements. Inside each
`seed_NN\k_05` folder, read `round_01_umap.png` through `round_03_umap.png` in
order, followed by `final_umap.png`. Each UMAP CSV records plotted MNIST
indices, true digits, injection and removal status, and coordinates. Round
score CSVs cover all current images; round folders show score distributions and
examples of removed images.

By default UMAP includes every current point. To limit plotting work, use for
example `--umap-sample 5000`: all injected and removed points are still
included, and retained normal points are sampled to fill the limit. The limit
can be exceeded when required points alone exceed it. Every stage has a
separate UMAP fit, so coordinates, orientations and distances are not
comparable across figures. Apparent separation alone does not establish
separation in the 64-dimensional space.

For an optional sensitivity comparison, use `--k-values 5 10 20`; this trains
independent cleaning trajectories for each k and produces 20 UMAP figures per
k under the default five-seed, three-round settings. Both cleaning and
evaluation use the selected k, so this is not an evaluation-only sweep.
The uncleaned and iterative evaluations use their contaminated training sets
as references; the ceiling uses a separately trained normal-only model and
reference. Labels are used for evaluation and visualizations, not for training
or removal. Test anomaly prevalence differs from the training injection
fraction.
=======
=======
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
# Iterative cleaning diagnostics: inject 7-8

This separate experiment uses the model, scoring and contaminated-reference evaluation from the original **KNN With Anomalies** notebook. It trains on normal digits 0-6, injects digits 7 and 8, and excludes digit 9. Existing 7-9 results are not overwritten or directly comparable.

When ready, from the workspace root run:

```sh
python3 'MNIST Latent Space Cleaning Verification/Autoencoder/Iterations/Train 0-6 inject 7-9/5 Spaces/64 Latent Dimensions/KNN Eval With Anomalies/Diagnostics 7-8/run_diagnostics.py'
```

Dependencies are PyTorch, torchvision, NumPy, matplotlib, scikit-learn and `umap-learn`. MNIST must already be present in a parent directory's `data/MNIST` folder. The source notebook must remain in `Autoencoder/Versions/KNN With Anomalies`.

The defaults train five seeds, with 64 dimensions, 15 epochs, three cleaning rounds, and independent trajectories for k=5, 10 and 20. Both cleaning and evaluation use the selected k, so this is a joint sensitivity comparison rather than an isolated evaluation-only sweep. The default run is computationally expensive: it fits 60 UMAPs in addition to the autoencoders and produces many plots.

Each invocation creates a timestamped `run_...` directory containing `config.json`, a `summary.md`, per-seed metrics, per-round removal statistics, and counts by digit. Every `seed_NN/k_NN` folder has UMAP plots and coordinates before every removal and after the final retraining. The overview identifies normal and injected digits, highlighting which are removed or retained; a second figure highlights each digit separately. The CSV identifies each plotted point by its MNIST training index, true digit, injection status and removal status. Round score CSVs contain all current points and their underlying cleaning signals, and round folders contain score distributions and examples of removed normal and injected digits.

By default UMAP includes every current point. To limit plotting work, use for example `--umap-sample 5000`: all injected and removed points are still included, and retained normal points are sampled to fill the limit. The limit can be exceeded when the required points alone exceed it. UMAP fits independently for each model and stage; coordinates, cluster sizes, and distances cannot be compared directly across plots, and visual separation does not prove separation in the 64-dimensional space. Use the per-point CSVs, score distributions and evaluation metrics alongside the plots.

<<<<<<< HEAD
The uncleaned baseline and iterative test evaluation deliberately use their contaminated training sets as references, matching the original notebook's evaluation approach. The ceiling uses a separately trained normal-only model and reference. Ground-truth digit labels are used only for evaluation and visualizations, not for training or removal. Test anomaly prevalence differs from the training injection fraction.
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
=======
The uncleaned baseline and iterative test evaluation deliberately use their contaminated training sets as references, matching the original notebook's evaluation approach. The ceiling uses a separately trained normal-only model and reference. Ground-truth digit labels are used only for evaluation and visualizations, not for training or removal. Test anomaly prevalence differs from the training injection fraction.
>>>>>>> 0d0fe084b37c813077df5428afe4048c7b4b732e
