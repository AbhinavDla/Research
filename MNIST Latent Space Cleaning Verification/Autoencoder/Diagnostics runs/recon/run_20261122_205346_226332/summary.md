# Iterative cleaning diagnostics: recon scoring

Normal digits: 0-6. Injected and test anomaly digits: 7-9.
Spaces: 5 independently sampled seeds; each model has 64 latent dimensions.
Seed folder numbers start at 1; the corresponding initial sampling seeds start at 0. For zero-based run index s, training uses seeds 100*s + round index (0-3), and 100*s + 92 for the ceiling.
Each seed has 3 cleaning rounds and one final plot. K values: [5].
Each k has its own retrained cleaning trajectory. The same k is used for cleaning and evaluation.
The uncleaned and iterative evaluations use their contaminated training sets as KNN references.
The ceiling uses an independently trained normal-only model and reference. Labels are used for evaluation and visualizations, not for training or removal.
The anomaly test prevalence is determined by the selected MNIST test digits, not the injection fraction.
UMAP fits independently at every stage: coordinates and distances are not comparable across plots.
By default all points are shown. If sampling is enabled, all removed and injected points are included and retained normal points are sampled.
UMAP is a visualization, not the space used for KNN or scoring.

## How to read the plots

For each seed, open `figures/seed_NN/k_NN/cleaning_overview.png`. The round plots show the training set *before* that round's removal; the final plot shows what remains after all rounds and retraining. Shapes identify the true digits; normal digits are grey and injected digits are colored. A black ring marks an image removed in that round. Labels reveal the true outcome for analysis; the cleaning algorithm does not use them. Counts in the legends are plotted counts, not necessarily full-dataset counts when sampling is enabled.

Each plot is a fresh UMAP fit of that stage's 64-dimensional representations. UMAP places similar examples near each other in a two-dimensional view, but it can distort distances and group sizes. Orientation, position, and distance between separate plots have no common meaning. Apparent clusters or overlap are clues, not proof of separation or removal quality.

## What the experiment does

A seed controls the random selection of normal and injected MNIST training images and the model's training randomness. The seeds are separate experimental runs, not separate groups on one plot. An autoencoder learns to reconstruct its input image without using its digit label. Its encoder turns each image into a 64-number representation called a latent vector; together these vectors form that model's latent space. An epoch is one pass through the current training set.

At each round a fresh autoencoder is trained on the current images. For each image, the method measures (1) the average distance to its k nearest other images in the latent space, and (2) reconstruction error, the squared pixel differences between the image and its reconstruction, divided by the image's total ink. Larger values can indicate unusual images. Each signal is converted to a relative rank from 0 to 1 among the current images; this method uses the relative rank of reconstruction error alone. This is not a probability. The highest-scoring images are removed, without using their labels. Each round removes the same number, calculated from the original mixed pool, not from the shrinking pool. The final model is retrained after the last removal.

## Meaning of the measurements

- **Removal precision:** injected images removed divided by all images removed. The other removed images are normal images incorrectly discarded.
- **Random precision:** proportion of injected images present before the round; the expected precision of random removal.
- **Cumulative anomaly recall:** fraction of the original injected images removed through the current round.
- **Accuracy:** fraction of normal test images correctly classified by a labeled k-nearest-neighbor vote, using the current training set as its reference. This is classification, not anomaly detection.
- **AUROC:** how often an injected test image receives a higher anomaly score than a normal test image; 0.5 is chance and 1 is perfect. Latent and reconstruction AUROC use each signal separately.
- **Average precision (AP):** summarizes precision across detection thresholds; higher is better. Its random baseline is the fraction of anomalies in the test set, which is not the training injection fraction.
- **Uncleaned:** model and reference before removal. **Iterative:** after all rounds. **Ceiling:** a separately trained model given only normal training images; a reference, not a cleaning method.
- **Mean +/- population standard deviation:** average across seeds and how much the seed results vary. These are descriptive, not confidence intervals.

## Results

Values are mean +/- population standard deviation across seeds.

| k | method | accuracy | AUROC | average precision | latent AUROC | reconstruction AUROC |
|---|---|---|---|---|---|---|
| 5 | uncleaned | 0.983 +/- 0.001 | 0.654 +/- 0.004 | 0.418 +/- 0.004 | 0.630 +/- 0.005 | 0.654 +/- 0.004 |
| 5 | iterative | 0.983 +/- 0.001 | 0.660 +/- 0.003 | 0.423 +/- 0.002 | 0.638 +/- 0.002 | 0.660 +/- 0.003 |
| 5 | ceiling | 0.986 +/- 0.001 | 0.756 +/- 0.005 | 0.537 +/- 0.006 | 0.879 +/- 0.006 | 0.756 +/- 0.005 |

## Removal

| k | round | precision | random precision | cumulative anomaly recall |
|---|---|---|---|---|
| 5 | 1 | 0.127 | 0.050 | 0.051 |
| 5 | 2 | 0.111 | 0.048 | 0.095 |
| 5 | 3 | 0.090 | 0.047 | 0.131 |

## What these results mean

For k=5, each seed started with an average of 1052.0 injected images. Across all rounds, the method removed an average of 138.2 injected images and 1124.8 normal images per seed. An average of 913.8 injected images remained among 19789.0 retained images (4.6% of the retained set).
Average cumulative anomaly recall was 13.1%: this is the share of initially injected images found, not the share of all removals that were correct.
Compared with the uncleaned model, the final model's mean test accuracy changed by -0.07 percentage points and its detection AUROC changed by +0.62 points. Its detection AUROC was -9.56 points relative to the normal-only ceiling. These are descriptive comparisons, not evidence that cleaning caused the differences; models are retrained and can vary with randomness.


## Every seed and round

Counts refer to the full training set, even when plots are sampled.

| Seed | k | Round | Before | Injected before | Removed | Injected removed | Normal removed | Injected left | Precision | Random precision | Cumulative recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 1 | 21052 | 1052 | 421 | 54 | 367 | 998 | 12.8% | 5.0% | 5.1% |
| 1 | 5 | 2 | 20631 | 998 | 421 | 45 | 376 | 953 | 10.7% | 4.8% | 9.4% |
| 1 | 5 | 3 | 20210 | 953 | 421 | 39 | 382 | 914 | 9.3% | 4.7% | 13.1% |
| 2 | 5 | 1 | 21052 | 1052 | 421 | 53 | 368 | 999 | 12.6% | 5.0% | 5.0% |
| 2 | 5 | 2 | 20631 | 999 | 421 | 54 | 367 | 945 | 12.8% | 4.8% | 10.2% |
| 2 | 5 | 3 | 20210 | 945 | 421 | 26 | 395 | 919 | 6.2% | 4.7% | 12.6% |
| 3 | 5 | 1 | 21052 | 1052 | 421 | 49 | 372 | 1003 | 11.6% | 5.0% | 4.7% |
| 3 | 5 | 2 | 20631 | 1003 | 421 | 45 | 376 | 958 | 10.7% | 4.9% | 8.9% |
| 3 | 5 | 3 | 20210 | 958 | 421 | 31 | 390 | 927 | 7.4% | 4.7% | 11.9% |
| 4 | 5 | 1 | 21052 | 1052 | 421 | 53 | 368 | 999 | 12.6% | 5.0% | 5.0% |
| 4 | 5 | 2 | 20631 | 999 | 421 | 51 | 370 | 948 | 12.1% | 4.8% | 9.9% |
| 4 | 5 | 3 | 20210 | 948 | 421 | 47 | 374 | 901 | 11.2% | 4.7% | 14.4% |
| 5 | 5 | 1 | 21052 | 1052 | 421 | 58 | 363 | 994 | 13.8% | 5.0% | 5.5% |
| 5 | 5 | 2 | 20631 | 994 | 421 | 39 | 382 | 955 | 9.3% | 4.8% | 9.2% |
| 5 | 5 | 3 | 20210 | 955 | 421 | 47 | 374 | 908 | 11.2% | 4.7% | 13.7% |

## Removed by digit

Each entry is removed / available before that round. Digits 0-6 are normal; 7-9 are injected.

| Seed | k | Round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 1 | 19 / 2794 | 4 / 3265 | 80 / 2880 | 51 / 2893 | 70 / 2731 | 84 / 2587 | 59 / 2850 | 4 / 359 | 35 / 343 | 15 / 350 |
| 1 | 5 | 2 | 28 / 2775 | 15 / 3261 | 73 / 2800 | 47 / 2842 | 76 / 2661 | 81 / 2503 | 56 / 2791 | 12 / 355 | 19 / 308 | 14 / 335 |
| 1 | 5 | 3 | 27 / 2747 | 14 / 3246 | 79 / 2727 | 61 / 2795 | 70 / 2585 | 66 / 2422 | 65 / 2735 | 11 / 343 | 17 / 289 | 11 / 321 |
| 2 | 5 | 1 | 15 / 2797 | 3 / 3224 | 81 / 2879 | 50 / 2900 | 65 / 2813 | 85 / 2593 | 69 / 2794 | 10 / 364 | 28 / 351 | 15 / 337 |
| 2 | 5 | 2 | 27 / 2782 | 13 / 3221 | 66 / 2798 | 49 / 2850 | 73 / 2748 | 85 / 2508 | 54 / 2725 | 13 / 354 | 34 / 323 | 7 / 322 |
| 2 | 5 | 3 | 15 / 2755 | 4 / 3208 | 85 / 2732 | 70 / 2801 | 76 / 2675 | 73 / 2423 | 72 / 2671 | 3 / 341 | 19 / 289 | 4 / 315 |
| 3 | 5 | 1 | 18 / 2817 | 10 / 3228 | 75 / 2881 | 47 / 2930 | 69 / 2783 | 92 / 2585 | 61 / 2776 | 7 / 371 | 28 / 339 | 14 / 342 |
| 3 | 5 | 2 | 26 / 2799 | 13 / 3218 | 72 / 2806 | 67 / 2883 | 69 / 2714 | 72 / 2493 | 57 / 2715 | 13 / 364 | 23 / 311 | 9 / 328 |
| 3 | 5 | 3 | 33 / 2773 | 11 / 3205 | 84 / 2734 | 54 / 2816 | 61 / 2645 | 83 / 2421 | 64 / 2658 | 7 / 351 | 18 / 288 | 6 / 319 |
| 4 | 5 | 1 | 23 / 2816 | 7 / 3221 | 83 / 2783 | 49 / 2903 | 72 / 2776 | 82 / 2665 | 52 / 2836 | 6 / 388 | 39 / 350 | 8 / 314 |
| 4 | 5 | 2 | 22 / 2793 | 21 / 3214 | 65 / 2700 | 48 / 2854 | 69 / 2704 | 81 / 2583 | 64 / 2784 | 10 / 382 | 26 / 311 | 15 / 306 |
| 4 | 5 | 3 | 30 / 2771 | 8 / 3193 | 70 / 2635 | 59 / 2806 | 61 / 2635 | 83 / 2502 | 63 / 2720 | 11 / 372 | 22 / 285 | 14 / 291 |
| 5 | 5 | 1 | 10 / 2916 | 7 / 3234 | 98 / 2808 | 61 / 2966 | 63 / 2709 | 81 / 2562 | 43 / 2805 | 5 / 351 | 38 / 354 | 15 / 347 |
| 5 | 5 | 2 | 34 / 2906 | 19 / 3227 | 72 / 2710 | 48 / 2905 | 74 / 2646 | 75 / 2481 | 60 / 2762 | 5 / 346 | 20 / 316 | 14 / 332 |
| 5 | 5 | 3 | 31 / 2872 | 10 / 3208 | 73 / 2638 | 67 / 2857 | 64 / 2572 | 64 / 2406 | 65 / 2702 | 6 / 341 | 24 / 296 | 17 / 318 |

## Per-seed evaluation

| Seed | k | Method | Reference size | Injected remaining | Accuracy | AUROC | AP |
|---|---|---|---|---|---|---|---|
| 1 | 5 | uncleaned | 21052 | 1052 | 0.983 | 0.651 | 0.413 |
| 1 | 5 | iterative | 19789 | 914 | 0.983 | 0.662 | 0.422 |
| 1 | 5 | ceiling | 20000 | 0 | 0.985 | 0.760 | 0.536 |
| 2 | 5 | uncleaned | 21052 | 1052 | 0.983 | 0.649 | 0.414 |
| 2 | 5 | iterative | 19789 | 919 | 0.985 | 0.655 | 0.419 |
| 2 | 5 | ceiling | 20000 | 0 | 0.986 | 0.748 | 0.530 |
| 3 | 5 | uncleaned | 21052 | 1052 | 0.984 | 0.653 | 0.419 |
| 3 | 5 | iterative | 19789 | 927 | 0.982 | 0.658 | 0.422 |
| 3 | 5 | ceiling | 20000 | 0 | 0.987 | 0.753 | 0.532 |
| 4 | 5 | uncleaned | 21052 | 1052 | 0.984 | 0.661 | 0.424 |
| 4 | 5 | iterative | 19789 | 901 | 0.981 | 0.662 | 0.426 |
| 4 | 5 | ceiling | 20000 | 0 | 0.986 | 0.760 | 0.544 |
| 5 | 5 | uncleaned | 21052 | 1052 | 0.983 | 0.656 | 0.419 |
| 5 | 5 | iterative | 19789 | 908 | 0.982 | 0.662 | 0.423 |
| 5 | 5 | ceiling | 20000 | 0 | 0.985 | 0.757 | 0.543 |

## Files

- data/metrics.csv: per-seed evaluation metrics and retained anomaly counts.
- data/rounds.csv: per-seed removal precision, current random baseline and cumulative recall.
- data/digits.csv: available and removed counts for each digit, round, seed and k.
- data/seed_NN/k_NN/round_NN_scores.csv: all current images, scores and removal decisions.
- data/seed_NN/k_NN/*_umap.csv: projections, true digits, injection and removal status.
- figures/seed_NN/k_NN/: four-panel overview, signal distributions and examples of removed digits.
- config.json: full settings and source notebook.
