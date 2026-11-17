# autoencoder latent-space voting diagnostics

10 independently trained spaces, 64 dimensions each; 10 training epochs per space. Seeds: 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009.
Training: 60,000 MNIST images. Evaluation: 10,000 held-out test images. kNN reference: all 60,000 labeled training images; k=5.

## What the experiment means

A latent space is the collection of numerical representations learned by one neural network. The autoencoder learns to reconstruct images without digit labels; SimCLR learns by bringing two randomly augmented views of an image together. Labels are used afterward for the reference set and evaluation, not for training either encoder.
For each test image, k-nearest neighbors (kNN) finds the closest labeled reference images in that model's latent space and votes on their digits. The ensemble then votes across the spaces' predicted digits. Tied ensemble votes select the earliest space's tied prediction. This is prediction voting, not an arithmetic average of latent coordinates: independently trained spaces are not aligned.

A model is a set of learned numerical weights; a seed initializes its random training choices. An epoch is one pass through the training images. The autoencoder's encoder compresses an image into a vector of numbers, and its decoder attempts to reconstruct the image from that vector. The training objective is reconstruction error, not digit classification.

The labeled reference set supplies example answers for kNN after training. The test set is separate from that reference set and is used to measure performance. The nearest neighbors are selected in the full latent space using Euclidean distance; the most common neighbor digit is the prediction. Neighbor-vote ties select the smallest digit.

## Reading the maps

Each space has a UMAP plot of the same 10,000 held-out images, with true digits on the left and that space's correct/incorrect predictions on the right. UMAP compresses 64-dimensional representations into two dimensions for visualization; the actual kNN uses the original representations. Each UMAP is fitted separately, so position, rotation and distances across different plots are not comparable. Clusters are suggestive, not proof.

## Results

- Mean individual accuracy: 96.84%; population standard deviation: 0.10%.
- Ensemble accuracy: 97.33%; difference from mean individual: +0.49 percentage points.
- Images with tied highest ensemble vote: 31 of 10,000.
- All spaces were correct on 9,378 images, all were wrong on 111, and they had mixed correctness on 511.
- Compared with space 1 specifically, the ensemble corrected 71 images that space 1 missed, but missed 42 images that space 1 got right.
- Accuracy is the fraction of test images classified correctly. Standard deviation describes variation between these spaces, not a confidence interval.
- A percentage-point gain is a direct subtraction of percentages. It does not imply the same gain on a different dataset, and the ten spaces are not ten independent test sets.

| Space | Seed | Accuracy |
|---:|---:|---:|
| 1 | 1000 | 97.04% |
| 2 | 1001 | 96.92% |
| 3 | 1002 | 96.85% |
| 4 | 1003 | 96.68% |
| 5 | 1004 | 96.78% |
| 6 | 1005 | 96.90% |
| 7 | 1006 | 96.73% |
| 8 | 1007 | 96.81% |
| 9 | 1008 | 96.79% |
| 10 | 1009 | 96.87% |

## Per-digit results

| Digit | Test images | Mean space accuracy | Ensemble accuracy |
|---:|---:|---:|---:|
| 0 | 980 | 99.34% | 99.49% |
| 1 | 1135 | 99.37% | 99.38% |
| 2 | 1032 | 97.34% | 97.87% |
| 3 | 1010 | 95.77% | 96.44% |
| 4 | 982 | 96.02% | 96.44% |
| 5 | 892 | 96.35% | 97.53% |
| 6 | 958 | 98.65% | 98.85% |
| 7 | 1028 | 95.84% | 96.50% |
| 8 | 974 | 94.74% | 95.79% |
| 9 | 1009 | 94.66% | 94.85% |

## Files

- `space_NN/umap.png` and `umap.csv`: map and coordinates for that space.
- `spaces.csv`: individual accuracies and seeds.
- `predictions.csv`: every test image's individual and ensemble predictions, winning vote count, and tie flag.
- `digits.csv`: per-digit accuracy; `config.json`: run settings.
