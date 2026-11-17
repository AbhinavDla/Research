# Iterative cleaning diagnostics

Normal digits: 0-6. Injected and test anomaly digits: 7-9.
Spaces: 5 independently sampled seeds; each model has 64 latent dimensions.
Each seed has 3 cleaning rounds and one final plot. K values: [5].
Each k has its own retrained cleaning trajectory. The same k is used for cleaning and evaluation.
The uncleaned and iterative evaluations use their contaminated training sets as KNN references.
The ceiling uses an independently trained normal-only model and reference. Labels are used for evaluation and visualizations, not for training or removal.
The anomaly test prevalence is determined by the selected MNIST test digits, not the injection fraction.
UMAP fits independently at every stage: coordinates and distances are not comparable across plots.
By default all points are shown. If sampling is enabled, all removed and injected points are included and retained normal points are sampled.
UMAP is a visualization, not the space used for KNN or scoring.

## How to read the plots

For each seed, open `round_01_umap.png` through `round_03_umap.png` in order, then `final_umap.png`. The round plots show the training set *before* that round's removal; the final plot shows what remains after all rounds and retraining. The left panel colors each digit, using crosses for the injected digits 7-9. The right panel distinguishes retained and removed normal images and retained and removed injected images. Labels reveal the true outcome for analysis; the cleaning algorithm does not use them. Counts in the legends are plotted counts, not necessarily full-dataset counts when sampling is enabled.

Each plot is a fresh UMAP fit of that stage's 64-dimensional representations. UMAP places similar examples near each other in a two-dimensional view, but it can distort distances and group sizes. Orientation, position, and distance between separate plots have no common meaning. Apparent clusters or overlap are clues, not proof of separation or removal quality.

## What the experiment does

A seed controls the random selection of normal and injected MNIST training images and the model's training randomness. The seeds are separate experimental runs, not separate groups on one plot. An autoencoder learns to reconstruct its input image without using its digit label. Its encoder turns each image into a 64-number representation called a latent vector; together these vectors form that model's latent space. An epoch is one pass through the current training set.

At each round a fresh autoencoder is trained on the current images. For each image, the method measures (1) the average distance to its k nearest other images in the latent space, and (2) reconstruction error, the squared pixel differences between the image and its reconstruction, divided by the image's total ink. Larger values can indicate unusual images. Each signal is converted to a relative rank from 0 to 1 among the current images; their average is the combined suspicion score. This is not a probability. The highest-scoring images are removed, without using their labels. Each round removes the same number, calculated from the original mixed pool, not from the shrinking pool. The final model is retrained after the last removal.

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
| 5 | uncleaned | 0.983 +/- 0.001 | 0.661 +/- 0.003 | 0.435 +/- 0.003 | 0.630 +/- 0.005 | 0.654 +/- 0.004 |
| 5 | iterative | 0.983 +/- 0.001 | 0.666 +/- 0.002 | 0.441 +/- 0.002 | 0.638 +/- 0.002 | 0.655 +/- 0.003 |
| 5 | ceiling | 0.986 +/- 0.001 | 0.864 +/- 0.002 | 0.714 +/- 0.002 | 0.879 +/- 0.006 | 0.756 +/- 0.005 |

## Removal

| k | round | precision | random precision | cumulative anomaly recall |
|---|---|---|---|---|
| 5 | 1 | 0.144 | 0.050 | 0.058 |
| 5 | 2 | 0.119 | 0.048 | 0.105 |
| 5 | 3 | 0.126 | 0.047 | 0.156 |

## What these results mean

For k=5, each seed started with an average of 1052.0 injected images. Across all rounds, the method removed an average of 163.6 injected images and 1099.4 normal images per seed. An average of 888.4 injected images remained among 19789.0 retained images (4.5% of the retained set).
Average cumulative anomaly recall was 15.6%: this is the share of initially injected images found, not the share of all removals that were correct.
Compared with the uncleaned model, the final model's mean test accuracy changed by -0.06 percentage points and its detection AUROC changed by +0.49 points. Its detection AUROC was -19.86 points relative to the normal-only ceiling. These are descriptive comparisons, not evidence that cleaning caused the differences; models are retrained and can vary with randomness.


## Every seed and round

Counts refer to the full training set, even when plots are sampled.

| Seed | k | Round | Before | Injected before | Removed | Injected removed | Normal removed | Injected left | Precision | Random precision | Cumulative recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 1 | 21052 | 1052 | 421 | 62 | 359 | 990 | 14.7% | 5.0% | 5.9% |
| 1 | 5 | 2 | 20631 | 990 | 421 | 42 | 379 | 948 | 10.0% | 4.8% | 9.9% |
| 1 | 5 | 3 | 20210 | 948 | 421 | 48 | 373 | 900 | 11.4% | 4.7% | 14.4% |
| 2 | 5 | 1 | 21052 | 1052 | 421 | 52 | 369 | 1000 | 12.4% | 5.0% | 4.9% |
| 2 | 5 | 2 | 20631 | 1000 | 421 | 57 | 364 | 943 | 13.5% | 4.8% | 10.4% |
| 2 | 5 | 3 | 20210 | 943 | 421 | 61 | 360 | 882 | 14.5% | 4.7% | 16.2% |
| 3 | 5 | 1 | 21052 | 1052 | 421 | 58 | 363 | 994 | 13.8% | 5.0% | 5.5% |
| 3 | 5 | 2 | 20631 | 994 | 421 | 51 | 370 | 943 | 12.1% | 4.8% | 10.4% |
| 3 | 5 | 3 | 20210 | 943 | 421 | 44 | 377 | 899 | 10.5% | 4.7% | 14.5% |
| 4 | 5 | 1 | 21052 | 1052 | 421 | 57 | 364 | 995 | 13.5% | 5.0% | 5.4% |
| 4 | 5 | 2 | 20631 | 995 | 421 | 53 | 368 | 942 | 12.6% | 4.8% | 10.5% |
| 4 | 5 | 3 | 20210 | 942 | 421 | 55 | 366 | 887 | 13.1% | 4.7% | 15.7% |
| 5 | 5 | 1 | 21052 | 1052 | 421 | 74 | 347 | 978 | 17.6% | 5.0% | 7.0% |
| 5 | 5 | 2 | 20631 | 978 | 421 | 47 | 374 | 931 | 11.2% | 4.7% | 11.5% |
| 5 | 5 | 3 | 20210 | 931 | 421 | 57 | 364 | 874 | 13.5% | 4.6% | 16.9% |

## Removed by digit

Each entry is removed / available before that round. Digits 0-6 are normal; 7-9 are injected.

| Seed | k | Round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 1 | 26 / 2794 | 0 / 3265 | 115 / 2880 | 71 / 2893 | 39 / 2731 | 77 / 2587 | 31 / 2850 | 7 / 359 | 48 / 343 | 7 / 350 |
| 1 | 5 | 2 | 21 / 2768 | 7 / 3265 | 133 / 2765 | 70 / 2822 | 25 / 2692 | 89 / 2510 | 34 / 2819 | 8 / 352 | 25 / 295 | 9 / 343 |
| 1 | 5 | 3 | 27 / 2747 | 9 / 3258 | 112 / 2632 | 57 / 2752 | 54 / 2667 | 93 / 2421 | 21 / 2785 | 15 / 344 | 26 / 270 | 7 / 334 |
| 2 | 5 | 1 | 27 / 2797 | 1 / 3224 | 114 / 2879 | 72 / 2900 | 34 / 2813 | 90 / 2593 | 31 / 2794 | 13 / 364 | 35 / 351 | 4 / 337 |
| 2 | 5 | 2 | 15 / 2770 | 5 / 3223 | 121 / 2765 | 71 / 2828 | 56 / 2779 | 72 / 2503 | 24 / 2763 | 18 / 351 | 33 / 316 | 6 / 333 |
| 2 | 5 | 3 | 25 / 2755 | 6 / 3218 | 126 / 2644 | 64 / 2757 | 48 / 2723 | 61 / 2431 | 30 / 2739 | 9 / 333 | 47 / 283 | 5 / 327 |
| 3 | 5 | 1 | 22 / 2817 | 3 / 3228 | 120 / 2881 | 80 / 2930 | 39 / 2783 | 72 / 2585 | 27 / 2776 | 12 / 371 | 38 / 339 | 8 / 342 |
| 3 | 5 | 2 | 22 / 2795 | 6 / 3225 | 125 / 2761 | 67 / 2850 | 41 / 2744 | 84 / 2513 | 25 / 2749 | 8 / 359 | 35 / 301 | 8 / 334 |
| 3 | 5 | 3 | 34 / 2773 | 4 / 3219 | 129 / 2636 | 68 / 2783 | 44 / 2703 | 62 / 2429 | 36 / 2724 | 5 / 351 | 32 / 266 | 7 / 326 |
| 4 | 5 | 1 | 22 / 2816 | 6 / 3221 | 112 / 2783 | 66 / 2903 | 45 / 2776 | 81 / 2665 | 32 / 2836 | 11 / 388 | 39 / 350 | 7 / 314 |
| 4 | 5 | 2 | 21 / 2794 | 3 / 3215 | 113 / 2671 | 73 / 2837 | 35 / 2731 | 96 / 2584 | 27 / 2804 | 11 / 377 | 33 / 311 | 9 / 307 |
| 4 | 5 | 3 | 14 / 2773 | 6 / 3212 | 134 / 2558 | 58 / 2764 | 62 / 2696 | 62 / 2488 | 30 / 2777 | 15 / 366 | 33 / 278 | 7 / 298 |
| 5 | 5 | 1 | 24 / 2916 | 4 / 3234 | 111 / 2808 | 83 / 2966 | 33 / 2709 | 67 / 2562 | 25 / 2805 | 8 / 351 | 51 / 354 | 15 / 347 |
| 5 | 5 | 2 | 20 / 2892 | 6 / 3230 | 136 / 2697 | 68 / 2883 | 33 / 2676 | 79 / 2495 | 32 / 2780 | 10 / 343 | 31 / 303 | 6 / 332 |
| 5 | 5 | 3 | 31 / 2872 | 4 / 3224 | 120 / 2561 | 63 / 2815 | 44 / 2643 | 84 / 2416 | 18 / 2748 | 3 / 333 | 43 / 272 | 11 / 326 |

## Per-seed evaluation

| Seed | k | Method | Reference size | Injected remaining | Accuracy | AUROC | AP |
|---|---|---|---|---|---|---|---|
| 1 | 5 | uncleaned | 21052 | 1052 | 0.983 | 0.661 | 0.437 |
| 1 | 5 | iterative | 19789 | 900 | 0.982 | 0.664 | 0.440 |
| 1 | 5 | ceiling | 20000 | 0 | 0.985 | 0.865 | 0.713 |
| 2 | 5 | uncleaned | 21052 | 1052 | 0.983 | 0.655 | 0.430 |
| 2 | 5 | iterative | 19789 | 882 | 0.983 | 0.664 | 0.440 |
| 2 | 5 | ceiling | 20000 | 0 | 0.986 | 0.862 | 0.715 |
| 3 | 5 | uncleaned | 21052 | 1052 | 0.984 | 0.661 | 0.436 |
| 3 | 5 | iterative | 19789 | 899 | 0.984 | 0.669 | 0.441 |
| 3 | 5 | ceiling | 20000 | 0 | 0.987 | 0.869 | 0.716 |
| 4 | 5 | uncleaned | 21052 | 1052 | 0.984 | 0.662 | 0.436 |
| 4 | 5 | iterative | 19789 | 887 | 0.982 | 0.665 | 0.439 |
| 4 | 5 | ceiling | 20000 | 0 | 0.986 | 0.863 | 0.712 |
| 5 | 5 | uncleaned | 21052 | 1052 | 0.983 | 0.665 | 0.435 |
| 5 | 5 | iterative | 19789 | 874 | 0.982 | 0.666 | 0.445 |
| 5 | 5 | ceiling | 20000 | 0 | 0.985 | 0.862 | 0.716 |

## Files

- metrics.csv: per-seed evaluation metrics and retained anomaly counts.
- rounds.csv: per-seed removal precision, current random baseline and cumulative recall.
- digits.csv: available and removed counts for each digit, round, seed and k.
- seed_NN/k_NN/round_NN_scores.csv: all current images, scores and removal decisions.
- seed_NN/k_NN/*_umap.png and .csv: projections, true digits, injection and removal status.
- seed_NN/k_NN/round_NN/: signal distributions and examples of removed normal and injected digits.
- config.json: full settings and source notebook.
