# MNIST autoencoder cleaning: score comparison

## Executive summary

Three scoring methods were compared using the same 5 initial MNIST pools. Each pool has 20,000 normal digits 0-6 and 1,052 injected digits 7-9. The models use 64 latent dimensions, 15 epochs and k values [5]; 3 rounds each remove 421 images. The combined run was previously completed and its saved data was reformatted, not retrained. The two single-score methods were retrained separately.
Seed numbers 1-5 are run labels; their initial sampling seeds are 0-4. For zero-based run index s, training uses seeds 100*s + round index (0-3), and 100*s + 92 for the normal-only ceiling.

The initial MNIST indices and labels have been checked for equality across all three methods for each seed. Later pools differ because the methods remove different images. The saved first-round kNN and reconstruction signals also match exactly across methods. Equal scores do not prove that the original model weights were identical. The combined run does not include a historical notebook hash or complete environment record, so exact numerical reproducibility of its trained models cannot be established.

| k | Scoring method | Injected removed | Normal removed | Removal precision | Anomaly recall | Remaining contamination |
|---:|---|---:|---:|---:|---:|---:|
| 5 | Combined ranks | 163.6 | 1099.4 | 13.0% | 15.6% | 4.49% |
| 5 | kNN distance only | 161.8 | 1101.2 | 12.8% | 15.4% | 4.50% |
| 5 | Reconstruction only | 138.2 | 1124.8 | 10.9% | 13.1% | 4.62% |

For k=5, Combined ranks removed the most injected images on average (163.6 per seed). This is a descriptive result for these 5 seeds, not a guarantee for new data.

Counts are averages per seed across all three rounds. Removal precision is the fraction of all removed images that were injected; anomaly recall is the fraction of the original injected images removed. A high recall can still come at the cost of removing many normal images.

## Round-by-round removal

| k | Round | Scoring method | Injected removed | Normal removed | Removal precision |
|---:|---:|---|---:|---:|---:|
| 5 | 1 | Combined ranks | 60.6 | 360.4 | 14.4% |
| 5 | 1 | kNN distance only | 73.6 | 347.4 | 17.5% |
| 5 | 1 | Reconstruction only | 53.4 | 367.6 | 12.7% |
| 5 | 2 | Combined ranks | 50.0 | 371.0 | 11.9% |
| 5 | 2 | kNN distance only | 46.2 | 374.8 | 11.0% |
| 5 | 2 | Reconstruction only | 46.8 | 374.2 | 11.1% |
| 5 | 3 | Combined ranks | 53.0 | 368.0 | 12.6% |
| 5 | 3 | kNN distance only | 42.0 | 379.0 | 10.0% |
| 5 | 3 | Reconstruction only | 38.0 | 383.0 | 9.0% |

A method can lead in the first round but not overall: after each round, the retained pools and retrained models differ.

## Held-out evaluation

| k | Scoring method | Normal digit accuracy | Detection AUROC | Detection AP |
|---:|---|---:|---:|---:|
| 5 | Combined ranks | 0.983 | 0.666 | 0.441 |
| 5 | kNN distance only | 0.982 | 0.635 | 0.428 |
| 5 | Reconstruction only | 0.983 | 0.660 | 0.423 |

Accuracy is kNN classification accuracy on normal held-out digits; AUROC and average precision (AP) measure anomaly ranking on held-out digits. The selected scoring signal also defines the detection score, so differences reflect both cleaning and the evaluation score, not cleaning alone. The test anomaly prevalence differs from the 5.0% training contamination, so AP is not directly comparable to removal precision.

## Every seed

| k | Seed | Method | Injected removed | Normal removed | Anomalies remaining |
|---:|---:|---|---:|---:|---:|
| 5 | 1 | Combined ranks | 152 | 1111 | 900 |
| 5 | 1 | kNN distance only | 173 | 1090 | 879 |
| 5 | 1 | Reconstruction only | 138 | 1125 | 914 |
| 5 | 2 | Combined ranks | 170 | 1093 | 882 |
| 5 | 2 | kNN distance only | 155 | 1108 | 897 |
| 5 | 2 | Reconstruction only | 133 | 1130 | 919 |
| 5 | 3 | Combined ranks | 153 | 1110 | 899 |
| 5 | 3 | kNN distance only | 144 | 1119 | 908 |
| 5 | 3 | Reconstruction only | 125 | 1138 | 927 |
| 5 | 4 | Combined ranks | 165 | 1098 | 887 |
| 5 | 4 | kNN distance only | 156 | 1107 | 896 |
| 5 | 4 | Reconstruction only | 151 | 1112 | 901 |
| 5 | 5 | Combined ranks | 178 | 1085 | 874 |
| 5 | 5 | kNN distance only | 181 | 1082 | 871 |
| 5 | 5 | Reconstruction only | 144 | 1119 | 908 |

## Removals by digit

Numbers are average images removed per seed over all rounds; 0-6 are normal, 7-9 are injected. Differences reflect both prevalence and detector behavior.

| k | Method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | Combined ranks | 70.2 | 14.0 | 364.2 | 206.2 | 126.4 | 233.8 | 84.6 | 30.6 | 109.8 | 23.2 |
| 5 | kNN distance only | 75.2 | 14.6 | 404.8 | 222.4 | 113.2 | 210.6 | 60.4 | 33.0 | 101.4 | 27.4 |
| 5 | Reconstruction only | 71.6 | 31.8 | 231.2 | 165.6 | 206.4 | 237.4 | 180.8 | 24.6 | 78.0 | 35.6 |

## How to interpret the maps

Each method folder contains a summary and one four-panel map for each seed. The first three panels show the set before each removal; black rings mark that round's removals. The fourth shows the retained set after final retraining. Shapes denote true digits, normal digits are grey, and injected digits are colored. Labels are only used to evaluate and illustrate results, not to choose removals.
Each panel is an independently fitted UMAP, so its orientation, position and distances cannot be compared directly with another panel. The algorithm scores 64-dimensional representations, not UMAP coordinates.

## Caveats

The 5 seeds are a descriptive sample, not confidence intervals. The original combined model weights were not saved, and repeated GPU training can vary. Do not infer causation or generalize to other anomaly types from these MNIST runs alone. The fixed removal budget always removes 1,263 examples per seed, even if a method cannot confidently separate them.

## Files

- `data/comparison.csv`: aggregate numeric comparison.
- `combined/original_archive` preserves the previously generated run unchanged.

- [Combined ranks details](../combined/run_20261115_155310_778479/summary.md).
- [kNN distance only details](../latent/run_20261122_204402_056774/summary.md).
- [Reconstruction only details](../recon/run_20261122_205346_226332/summary.md).