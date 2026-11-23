# Iterative cleaning diagnostics: latent scoring

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

At each round a fresh autoencoder is trained on the current images. For each image, the method measures (1) the average distance to its k nearest other images in the latent space, and (2) reconstruction error, the squared pixel differences between the image and its reconstruction, divided by the image's total ink. Larger values can indicate unusual images. Each signal is converted to a relative rank from 0 to 1 among the current images; this method uses the relative rank of the kNN distance alone. This is not a probability. The highest-scoring images are removed, without using their labels. Each round removes the same number, calculated from the original mixed pool, not from the shrinking pool. The final model is retrained after the last removal.

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
| 5 | uncleaned | 0.983 +/- 0.001 | 0.630 +/- 0.005 | 0.421 +/- 0.006 | 0.630 +/- 0.005 | 0.654 +/- 0.004 |
| 5 | iterative | 0.982 +/- 0.001 | 0.635 +/- 0.003 | 0.428 +/- 0.005 | 0.635 +/- 0.003 | 0.659 +/- 0.003 |
| 5 | ceiling | 0.986 +/- 0.001 | 0.879 +/- 0.006 | 0.749 +/- 0.009 | 0.879 +/- 0.006 | 0.756 +/- 0.005 |

## Removal

| k | round | precision | random precision | cumulative anomaly recall |
|---|---|---|---|---|
| 5 | 1 | 0.175 | 0.050 | 0.070 |
| 5 | 2 | 0.110 | 0.047 | 0.114 |
| 5 | 3 | 0.100 | 0.046 | 0.154 |

## What these results mean

For k=5, each seed started with an average of 1052.0 injected images. Across all rounds, the method removed an average of 161.8 injected images and 1101.2 normal images per seed. An average of 890.2 injected images remained among 19789.0 retained images (4.5% of the retained set).
Average cumulative anomaly recall was 15.4%: this is the share of initially injected images found, not the share of all removals that were correct.
Compared with the uncleaned model, the final model's mean test accuracy changed by -0.15 percentage points and its detection AUROC changed by +0.55 points. Its detection AUROC was -24.42 points relative to the normal-only ceiling. These are descriptive comparisons, not evidence that cleaning caused the differences; models are retrained and can vary with randomness.


## Every seed and round

Counts refer to the full training set, even when plots are sampled.

| Seed | k | Round | Before | Injected before | Removed | Injected removed | Normal removed | Injected left | Precision | Random precision | Cumulative recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 1 | 21052 | 1052 | 421 | 77 | 344 | 975 | 18.3% | 5.0% | 7.3% |
| 1 | 5 | 2 | 20631 | 975 | 421 | 49 | 372 | 926 | 11.6% | 4.7% | 12.0% |
| 1 | 5 | 3 | 20210 | 926 | 421 | 47 | 374 | 879 | 11.2% | 4.6% | 16.4% |
| 2 | 5 | 1 | 21052 | 1052 | 421 | 62 | 359 | 990 | 14.7% | 5.0% | 5.9% |
| 2 | 5 | 2 | 20631 | 990 | 421 | 51 | 370 | 939 | 12.1% | 4.8% | 10.7% |
| 2 | 5 | 3 | 20210 | 939 | 421 | 42 | 379 | 897 | 10.0% | 4.6% | 14.7% |
| 3 | 5 | 1 | 21052 | 1052 | 421 | 65 | 356 | 987 | 15.4% | 5.0% | 6.2% |
| 3 | 5 | 2 | 20631 | 987 | 421 | 42 | 379 | 945 | 10.0% | 4.8% | 10.2% |
| 3 | 5 | 3 | 20210 | 945 | 421 | 37 | 384 | 908 | 8.8% | 4.7% | 13.7% |
| 4 | 5 | 1 | 21052 | 1052 | 421 | 76 | 345 | 976 | 18.1% | 5.0% | 7.2% |
| 4 | 5 | 2 | 20631 | 976 | 421 | 45 | 376 | 931 | 10.7% | 4.7% | 11.5% |
| 4 | 5 | 3 | 20210 | 931 | 421 | 35 | 386 | 896 | 8.3% | 4.6% | 14.8% |
| 5 | 5 | 1 | 21052 | 1052 | 421 | 88 | 333 | 964 | 20.9% | 5.0% | 8.4% |
| 5 | 5 | 2 | 20631 | 964 | 421 | 44 | 377 | 920 | 10.5% | 4.7% | 12.5% |
| 5 | 5 | 3 | 20210 | 920 | 421 | 49 | 372 | 871 | 11.6% | 4.6% | 17.2% |

## Removed by digit

Each entry is removed / available before that round. Digits 0-6 are normal; 7-9 are injected.

| Seed | k | Round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 1 | 22 / 2794 | 7 / 3265 | 116 / 2880 | 65 / 2893 | 40 / 2731 | 70 / 2587 | 24 / 2850 | 16 / 359 | 50 / 343 | 11 / 350 |
| 1 | 5 | 2 | 29 / 2772 | 3 / 3258 | 128 / 2764 | 87 / 2828 | 26 / 2691 | 81 / 2517 | 18 / 2826 | 10 / 343 | 29 / 293 | 10 / 339 |
| 1 | 5 | 3 | 21 / 2743 | 5 / 3255 | 157 / 2636 | 59 / 2741 | 41 / 2665 | 76 / 2436 | 15 / 2808 | 10 / 333 | 29 / 264 | 8 / 329 |
| 2 | 5 | 1 | 23 / 2797 | 4 / 3224 | 123 / 2879 | 86 / 2900 | 41 / 2813 | 63 / 2593 | 19 / 2794 | 13 / 364 | 41 / 351 | 8 / 337 |
| 2 | 5 | 2 | 27 / 2774 | 7 / 3220 | 133 / 2756 | 72 / 2814 | 38 / 2772 | 69 / 2530 | 24 / 2775 | 11 / 351 | 33 / 310 | 7 / 329 |
| 2 | 5 | 3 | 21 / 2747 | 6 / 3213 | 150 / 2623 | 74 / 2742 | 51 / 2734 | 58 / 2461 | 19 / 2751 | 12 / 340 | 25 / 277 | 5 / 322 |
| 3 | 5 | 1 | 19 / 2817 | 9 / 3228 | 136 / 2881 | 66 / 2930 | 42 / 2783 | 60 / 2585 | 24 / 2776 | 13 / 371 | 38 / 339 | 14 / 342 |
| 3 | 5 | 2 | 28 / 2798 | 5 / 3219 | 125 / 2745 | 91 / 2864 | 42 / 2741 | 71 / 2525 | 17 / 2752 | 7 / 358 | 27 / 301 | 8 / 328 |
| 3 | 5 | 3 | 36 / 2770 | 2 / 3214 | 146 / 2620 | 70 / 2773 | 27 / 2699 | 77 / 2454 | 26 / 2735 | 4 / 351 | 27 / 274 | 6 / 320 |
| 4 | 5 | 1 | 17 / 2816 | 4 / 3221 | 129 / 2783 | 66 / 2903 | 43 / 2776 | 66 / 2665 | 20 / 2836 | 20 / 388 | 43 / 350 | 13 / 314 |
| 4 | 5 | 2 | 29 / 2799 | 5 / 3217 | 127 / 2654 | 87 / 2837 | 40 / 2733 | 68 / 2599 | 20 / 2816 | 15 / 368 | 20 / 307 | 10 / 301 |
| 4 | 5 | 3 | 25 / 2770 | 3 / 3212 | 150 / 2527 | 85 / 2750 | 25 / 2693 | 79 / 2531 | 19 / 2796 | 5 / 353 | 26 / 287 | 4 / 291 |
| 5 | 5 | 1 | 20 / 2916 | 6 / 3234 | 121 / 2808 | 74 / 2966 | 36 / 2709 | 60 / 2562 | 16 / 2805 | 13 / 351 | 56 / 354 | 19 / 347 |
| 5 | 5 | 2 | 28 / 2896 | 4 / 3228 | 136 / 2687 | 66 / 2892 | 37 / 2673 | 81 / 2502 | 25 / 2789 | 7 / 338 | 28 / 298 | 9 / 328 |
| 5 | 5 | 3 | 31 / 2868 | 3 / 3224 | 147 / 2551 | 64 / 2826 | 37 / 2636 | 74 / 2421 | 16 / 2764 | 9 / 331 | 35 / 270 | 5 / 319 |

## Per-seed evaluation

| Seed | k | Method | Reference size | Injected remaining | Accuracy | AUROC | AP |
|---|---|---|---|---|---|---|---|
| 1 | 5 | uncleaned | 21052 | 1052 | 0.983 | 0.635 | 0.431 |
| 1 | 5 | iterative | 19789 | 879 | 0.980 | 0.636 | 0.432 |
| 1 | 5 | ceiling | 20000 | 0 | 0.985 | 0.873 | 0.743 |
| 2 | 5 | uncleaned | 21052 | 1052 | 0.983 | 0.621 | 0.415 |
| 2 | 5 | iterative | 19789 | 897 | 0.983 | 0.639 | 0.435 |
| 2 | 5 | ceiling | 20000 | 0 | 0.986 | 0.885 | 0.758 |
| 3 | 5 | uncleaned | 21052 | 1052 | 0.984 | 0.631 | 0.422 |
| 3 | 5 | iterative | 19789 | 908 | 0.983 | 0.635 | 0.421 |
| 3 | 5 | ceiling | 20000 | 0 | 0.987 | 0.888 | 0.760 |
| 4 | 5 | uncleaned | 21052 | 1052 | 0.984 | 0.626 | 0.416 |
| 4 | 5 | iterative | 19789 | 896 | 0.981 | 0.634 | 0.429 |
| 4 | 5 | ceiling | 20000 | 0 | 0.986 | 0.873 | 0.736 |
| 5 | 5 | uncleaned | 21052 | 1052 | 0.983 | 0.635 | 0.424 |
| 5 | 5 | iterative | 19789 | 871 | 0.981 | 0.631 | 0.426 |
| 5 | 5 | ceiling | 20000 | 0 | 0.985 | 0.878 | 0.749 |

## Files

- data/metrics.csv: per-seed evaluation metrics and retained anomaly counts.
- data/rounds.csv: per-seed removal precision, current random baseline and cumulative recall.
- data/digits.csv: available and removed counts for each digit, round, seed and k.
- data/seed_NN/k_NN/round_NN_scores.csv: all current images, scores and removal decisions.
- data/seed_NN/k_NN/*_umap.csv: projections, true digits, injection and removal status.
- figures/seed_NN/k_NN/: four-panel overview, signal distributions and examples of removed digits.
- config.json: full settings and source notebook.
