"""
SimCLR LATENT-SPACE ENSEMBLE FOR MNIST  (self-supervised encoders + few labels)
================================================================================

BIG PICTURE (written for someone who knows CNNs but not self-supervised learning)
-------------------------------------------------------------------------------
1. We train 20 separate CNN encoders. Each maps a 28x28 image to a 64-number
   vector (a "latent space" is just the space these vectors live in).

2. NO LABELS are used in training. Instead we use SimCLR:
      - Take an image and make two randomly augmented copies (rotated, shifted,
        noisy...). Call them "view A" and "view B".
      - Train the CNN so that view A and view B of the SAME image get similar
        vectors, while views of DIFFERENT images get dissimilar vectors.
      - The "different image" part stops the CNN from cheating by outputting the
        same vector for everything.
   The CNN ends up ignoring rotation/shift/noise and keeping the shape of the
   digit, which is exactly what you want for classification.

3. After training, the encoder is FROZEN. We encode the training images, attach
   labels to a SMALL subset of them (this is the "semi-supervised" part), and
   classify test images with k-nearest-neighbors (kNN): find the k closest
   labeled points and take a majority vote. kNN has no trainable parameters.

4. The 20 spaces each predict a digit, then vote.

5. As a sanity check, we also run kNN on raw pixels with the same labels.
   If SimCLR doesn't beat that, it isn't adding value.
"""

import math
import random
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets


# ============================================================
# CONFIGURATION
# ============================================================

NUM_MODELS = 20          # number of independent latent spaces (ensemble size)
LATENT_DIM = 64          # length of each image's embedding vector
K_NEIGHBORS = 5          # neighbors used in kNN voting

EPOCHS = 15              # passes over each model's training subset
TRAIN_SUBSET = 25000     # images each model trains on (lower = faster)
BATCH_SIZE = 512         # SimCLR uses the rest of the batch as "different images",
                         # so bigger batches generally help (but cost more memory)
LEARNING_RATE = 1e-3
TEMPERATURE = 0.5        # SimCLR "sharpness" knob; 0.5 is a common default for small batches

# How many LABELED images per class the kNN reference set gets.
# None means "use all 60,000 labels".
LABELS_PER_CLASS = [50]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Results go into a parent folder called "Iterations", next to this script.
OUTPUT_DIR = Path(__file__).parent / "Iterations" / f"SimCLR {NUM_MODELS} Spaces"
OUTPUT_FILE = OUTPUT_DIR / f"{LATENT_DIM} Latent Dimensions.txt"


# ============================================================
# DATA (loaded once, kept in memory as tensors)
# ============================================================

def load_mnist():
    """
    Load MNIST as plain tensors instead of using a DataLoader.
    MNIST is tiny (60k small images), so keeping everything in memory and
    slicing it ourselves is much faster than a DataLoader on CPU.

    Returns images shaped [N, 1, 28, 28] with values in [0, 1], plus labels.
    """
    train = datasets.MNIST(root="./data", train=True, download=True)
    test = datasets.MNIST(root="./data", train=False, download=True)

    train_x = train.data.float().div(255).unsqueeze(1)   # add channel dim
    test_x = test.data.float().div(255).unsqueeze(1)
    return train_x, train.targets, test_x, test.targets


# ============================================================
# AUGMENTATION  (this defines what the model learns to IGNORE)
# ============================================================

def augment(x):
    """
    Randomly transform a whole batch of images [B, 1, 28, 28].

    Each image gets its own random rotation, zoom, shift and pixel noise.
    The CNN is trained to produce the same embedding no matter which of these
    transforms was applied, so it learns to ignore them.

    Implementation note: we build one affine transformation matrix per image and
    apply them all at once with grid_sample. This is far faster than looping
    over images with torchvision transforms.
    """
    B = x.size(0)
    dev = x.device

    # Random values in [-1, 1], scaled to the range we want.
    angle = (torch.rand(B, device=dev) * 2 - 1) * math.radians(20)   # +/- 20 degrees
    scale = 1 + (torch.rand(B, device=dev) * 2 - 1) * 0.15           # zoom +/- 15%
    tx = (torch.rand(B, device=dev) * 2 - 1) * 0.15                  # shift +/- 15%
    ty = (torch.rand(B, device=dev) * 2 - 1) * 0.15

    # Build the 2x3 affine matrix for each image (rotation + zoom + shift).
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([
        torch.stack([cos, -sin, tx], dim=1),
        torch.stack([sin,  cos, ty], dim=1),
    ], dim=1)                                                         # [B, 2, 3]

    # Warp the images according to those matrices.
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    out = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")

    # Add Gaussian pixel noise, then keep pixel values valid.
    out = out + 0.1 * torch.randn_like(out)
    return out.clamp(0.0, 1.0)


# ============================================================
# MODEL
# ============================================================

class SimCLRModel(nn.Module):
    """
    Two parts:

    backbone   : a small CNN, image -> 64-number embedding.
                 THIS is the "latent space" we keep after training.

    projector  : a tiny MLP (stack of Linear layers) on top of the backbone.
                 SimCLR computes its loss on the projector's output, not the
                 backbone's. This protects the backbone's embedding from being
                 distorted by the loss. We throw the projector away after training.
    """

    def __init__(self, latent_dim=64):
        super().__init__()

        # Standard strided-conv CNN: 28x28 -> 14x14 -> 7x7 -> 4x4.
        # BatchNorm + ReLU after each conv, like any normal CNN.
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Flatten(),                              # 128 channels * 4 * 4 = 2048 numbers
            nn.Linear(128 * 4 * 4, latent_dim),        # -> 64-number embedding
        )

        # Small MLP head used only during training.
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

    def embed(self, x):
        """
        Final latent vector used for kNN.
        We L2-normalize it (length 1), so every image sits on a sphere and
        "closeness" becomes cosine similarity. This tends to work better than
        raw Euclidean distance.
        """
        return F.normalize(self.backbone(x), dim=1)


# ============================================================
# SimCLR LOSS (NT-Xent)
# ============================================================

def nt_xent_loss(z1, z2, temperature):
    """
    z1, z2: projector outputs for view A and view B, each [N, D].
    Row i of z1 and row i of z2 come from the SAME original image.

    Idea: treat it as a classification problem.
    For each of the 2N embeddings, the "correct class" is its partner view,
    and the 2N-2 other embeddings in the batch are wrong answers (negatives).
    Cross-entropy then pulls partners together and pushes everything else apart.
    """
    N = z1.size(0)

    # Stack both views and normalize so dot product = cosine similarity.
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)      # [2N, D]

    # Similarity of every embedding with every other, divided by temperature.
    # Lower temperature = the loss focuses harder on the most confusable negatives.
    sim = (z @ z.T) / temperature                           # [2N, 2N]

    # An embedding must not count itself as a candidate partner: mask the diagonal.
    self_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    # Row i's correct answer: i+N for the first half, i-N for the second half.
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)

    return F.cross_entropy(sim, targets)


# ============================================================
# TRAINING (no labels used anywhere here)
# ============================================================

def train_simclr(model, images):
    """
    Train one encoder with SimCLR on `images` ([N, 1, 28, 28]).
    Labels are never passed in, so this is fully self-supervised.
    """
    model.train()   # BatchNorm uses batch statistics while training
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    N = images.size(0)

    for epoch in range(EPOCHS):
        perm = torch.randperm(N)    # reshuffle each epoch
        total_loss, num_batches = 0.0, 0

        # Step through in batches. We skip the final partial batch so every
        # batch has the same number of negatives.
        for start in range(0, N - BATCH_SIZE + 1, BATCH_SIZE):
            batch = images[perm[start:start + BATCH_SIZE]].to(DEVICE)

            # Two different random augmentations of the same images.
            view_a, view_b = augment(batch), augment(batch)

            # Backbone -> projector for both views.
            z_a = model.projector(model.backbone(view_a))
            z_b = model.projector(model.backbone(view_b))

            loss = nt_xent_loss(z_a, z_b, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        print(f"    Epoch {epoch + 1:02d}/{EPOCHS} | SimCLR Loss: {total_loss / num_batches:.4f}")


# ============================================================
# ENCODING + kNN
# ============================================================

@torch.no_grad()
def encode(model, images, batch_size=2048):
    """
    Run images through the FROZEN encoder and return their embeddings.
    eval() switches BatchNorm to its stored running statistics, and
    no_grad() skips gradient bookkeeping (faster, less memory).
    """
    model.eval()
    out = []
    for start in range(0, images.size(0), batch_size):
        batch = images[start:start + batch_size].to(DEVICE)
        out.append(model.embed(batch).cpu())
    return torch.cat(out)


def classify_one_point(new_point, reference_points, reference_labels, k):
    """
    Classify ONE point using only distance and majority voting.
    (Same logic as classify_from_latent_space in your original file.)

    new_point        : numpy array [64]
    reference_points : numpy array [num_refs, 64]  (the labeled points)
    reference_labels : numpy array [num_refs]      (their digits)
    """
    # 1. Euclidean distance from the new point to every reference point:
    #    sqrt( (r1 - n1)^2 + (r2 - n2)^2 + ... + (r64 - n64)^2 )
    distances = np.linalg.norm(reference_points - new_point, axis=1)

    # 2. Indices of the k smallest distances. argpartition avoids a full sort.
    nearest_indices = np.argpartition(distances, k - 1)[:k]

    # 3. The digits those k neighbors carry.
    nearest_labels = reference_labels[nearest_indices]

    # 4. Majority vote: count how many neighbors voted for each digit 0-9,
    #    then pick the digit with the most votes (ties go to the lower digit).
    vote_counts = np.bincount(nearest_labels, minlength=10)
    return int(np.argmax(vote_counts))


def knn_predict(query_feats, ref_feats, ref_labels, k, chunk=None):
    """
    Classify every query point, one at a time, with the manual math above.
    Takes and returns torch tensors so the rest of the script is unchanged.
    (`chunk` is unused; it's kept so existing calls still work.)
    """
    ref_np = ref_feats.numpy()
    labels_np = ref_labels.numpy()
    k = min(k, len(ref_np))        # can't ask for more neighbors than exist

    predictions = [
        classify_one_point(point, ref_np, labels_np, k)
        for point in query_feats.numpy()
    ]
    return torch.tensor(predictions)


def choose_labeled_indices(train_labels, per_class, seed=0):
    """
    Pick which training images get to keep their labels.
    per_class=None -> all of them. Otherwise `per_class` random images per digit.
    The same seed is used for every latent space, so spaces are compared fairly.
    """
    if per_class is None:
        return torch.arange(train_labels.size(0))

    rng = np.random.RandomState(seed)
    labels_np = train_labels.numpy()
    chosen = []
    for digit in range(10):
        digit_idx = np.where(labels_np == digit)[0]
        chosen.extend(rng.choice(digit_idx, per_class, replace=False))
    return torch.tensor(sorted(chosen))


def setting_name(per_class):
    return "all labels" if per_class is None else f"{per_class} labels/class"


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()

    print("=" * 70)
    print("SimCLR LATENT-SPACE ENSEMBLE (self-supervised encoders)")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Number of latent spaces: {NUM_MODELS}")
    print(f"Latent dimensions: {LATENT_DIM}")
    print(f"Nearest neighbors: {K_NEIGHBORS}")
    print(f"Epochs per space: {EPOCHS}")
    print(f"Training images per space: {TRAIN_SUBSET}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Label settings: {[setting_name(s) for s in LABELS_PER_CLASS]}\n")

    train_x, train_y, test_x, test_y = load_mnist()

    # Which images keep labels for each setting (same for every latent space).
    labeled_idx = {s: choose_labeled_indices(train_y, s) for s in LABELS_PER_CLASS}

    # all_preds[setting] = list of test predictions, one tensor per latent space.
    all_preds = {s: [] for s in LABELS_PER_CLASS}
    # per_space_acc[setting] = list of accuracies, one per latent space.
    per_space_acc = {s: [] for s in LABELS_PER_CLASS}

    # ---- train and evaluate each latent space ----
    for m in range(NUM_MODELS):
        print("\n" + "=" * 70)
        print(f"LATENT SPACE {m + 1}/{NUM_MODELS}")
        print("=" * 70)

        # Different seed -> different init, different data subset, different augmentations.
        seed = 1000 + m
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Each model trains on its own random subset of the (unlabeled) training images.
        subset = torch.randperm(train_x.size(0))[:TRAIN_SUBSET]

        model = SimCLRModel(LATENT_DIM).to(DEVICE)
        train_simclr(model, train_x[subset])      # labels not passed in
        print("\n    Encoder training complete and frozen.")

        # Encode ALL training images (as potential reference points) and the test set.
        train_feats = encode(model, train_x)
        test_feats = encode(model, test_x)

        # Classify the test set once per label setting.
        for s in LABELS_PER_CLASS:
            idx = labeled_idx[s]
            preds = knn_predict(test_feats, train_feats[idx], train_y[idx], K_NEIGHBORS)
            acc = (preds == test_y).float().mean().item()
            all_preds[s].append(preds)
            per_space_acc[s].append(acc)
            print(f"    {setting_name(s):>16}: {acc * 100:.2f}%")

        elapsed = (time.time() - start_time) / 60
        print(f"    (elapsed: {elapsed:.1f} min)")

    # ---- raw-pixel kNN baseline (no learning at all) ----
    print("\n" + "=" * 70)
    print("BASELINE: kNN ON RAW PIXELS")
    print("=" * 70)
    raw_train = F.normalize(train_x.flatten(1), dim=1)
    raw_test = F.normalize(test_x.flatten(1), dim=1)
    raw_acc = {}
    for s in LABELS_PER_CLASS:
        idx = labeled_idx[s]
        preds = knn_predict(raw_test, raw_train[idx], train_y[idx], K_NEIGHBORS)
        raw_acc[s] = (preds == test_y).float().mean().item()
        print(f"    {setting_name(s):>16}: {raw_acc[s] * 100:.2f}%")

    # ---- ensemble vote + summary ----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Labels':>16} | {'Raw pixels':>10} | {'Mean single':>11} | {'Std':>6} | {'Ensemble':>8} | {'Gain':>6}")
    print("-" * 70)

    for s in LABELS_PER_CLASS:
        # Ensemble vote: count each space's predicted digit, pick the winner.
        stacked = torch.stack(all_preds[s]).numpy()           # [num_spaces, num_test]
        ensemble_preds = torch.tensor([
            np.argmax(np.bincount(stacked[:, i], minlength=10))   # most common vote
            for i in range(stacked.shape[1])
        ])
        ens_acc = (ensemble_preds == test_y).float().mean().item()

        mean_acc = float(np.mean(per_space_acc[s]))
        std_acc = float(np.std(per_space_acc[s]))
        gain = ens_acc - mean_acc

        print(
            f"{setting_name(s):>16} | {raw_acc[s] * 100:9.2f}% | {mean_acc * 100:10.2f}% "
            f"| {std_acc * 100:5.2f}% | {ens_acc * 100:7.2f}% | {gain * 100:+5.2f}"
        )

    print("\nGain = ensemble accuracy minus mean single-space accuracy (percentage points).")
    print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")


# ============================================================
# RUN (prints to the screen AND saves to Iterations/...)
# ============================================================

class Tee:
    """Sends everything printed to several places at once (screen + file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(stream.isatty() for stream in self.streams)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # creates Iterations/... if missing

    with OUTPUT_FILE.open("w", encoding="utf-8") as output_file:
        output = Tee(sys.stdout, output_file)
        with redirect_stdout(output), redirect_stderr(output):
            main()