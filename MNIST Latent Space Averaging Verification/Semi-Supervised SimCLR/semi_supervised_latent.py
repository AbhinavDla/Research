"""
SimCLR LATENT-SPACE ENSEMBLE FOR MNIST  (self-supervised encoders + few labels)
================================================================================
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
LATENT_DIM = 32         # length of each image's embedding vector
K_NEIGHBORS = 5          # neighbors used in kNN voting

EPOCHS = 10              # increased epochs for better convergence
TRAIN_SUBSET = 60000     # images each model trains on
BATCH_SIZE = 512         
LEARNING_RATE = 3e-3     # adjusted base LR for AdamW
TEMPERATURE = 0.5        # SimCLR sharpness knob

# How many LABELED images per class the kNN reference set gets.
LABELS_PER_CLASS = [50]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path(__file__).parent / "Iterations" / f"SimCLR {NUM_MODELS} Spaces"
OUTPUT_FILE = OUTPUT_DIR / f"{LATENT_DIM} Latent Dimensions.txt"


# ============================================================
# DATA (loaded once, kept in memory as tensors)
# ============================================================

def load_mnist():
    """
    Load MNIST as plain tensors instead of using a DataLoader.
    Returns images shaped [N, 1, 28, 28] with values in [0, 1], plus labels.
    """
    train = datasets.MNIST(root="./data", train=True, download=True)
    test = datasets.MNIST(root="./data", train=False, download=True)

    train_x = train.data.float().div(255).unsqueeze(1)
    test_x = test.data.float().div(255).unsqueeze(1)
    return train_x, train.targets, test_x, test.targets


# ============================================================
# AUGMENTATION (Toned down for MNIST digit preservation)
# ============================================================

def augment(x):
    """
    Randomly transform a whole batch of images [B, 1, 28, 28].
    Milder rotations and shifts prevent 6s/9s or 1s/7s from flipping identities.
    """
    B = x.size(0)
    dev = x.device

    # Milder ranges: +/- 7 degrees rotation, +/- 10% zoom/shift
    angle = (torch.rand(B, device=dev) * 2 - 1) * math.radians(7)
    scale = 1 + (torch.rand(B, device=dev) * 2 - 1) * 0.10
    tx = (torch.rand(B, device=dev) * 2 - 1) * 0.10
    ty = (torch.rand(B, device=dev) * 2 - 1) * 0.10

    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([
        torch.stack([cos, -sin, tx], dim=1),
        torch.stack([sin,  cos, ty], dim=1),
    ], dim=1)

    grid = F.affine_grid(theta, x.size(), align_corners=False)
    out = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")

    # Light noise
    out = out + 0.05 * torch.randn_like(out)
    return out.clamp(0.0, 1.0)


# ============================================================
# MODEL
# ============================================================

class SimCLRModel(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
        )

        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

    def embed(self, x):
        return F.normalize(self.backbone(x), dim=1)


# ============================================================
# SimCLR LOSS (NT-Xent)
# ============================================================

def nt_xent_loss(z1, z2, temperature):
    N = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = (z @ z.T) / temperature

    self_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)


# ============================================================
# TRAINING 
# ============================================================

def train_simclr(model, images):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Cosine Annealing with Warmup scheduling
    total_steps = EPOCHS * (images.size(0) // BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    N = images.size(0)

    for epoch in range(EPOCHS):
        perm = torch.randperm(N)
        total_loss, num_batches = 0.0, 0

        for start in range(0, N - BATCH_SIZE + 1, BATCH_SIZE):
            batch = images[perm[start:start + BATCH_SIZE]].to(DEVICE)

            view_a, view_b = augment(batch), augment(batch)

            z_a = model.projector(model.backbone(view_a))
            z_b = model.projector(model.backbone(view_b))

            loss = nt_xent_loss(z_a, z_b, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1:02d}/{EPOCHS} | SimCLR Loss: {total_loss / num_batches:.4f}")


# ============================================================
# ENCODING + kNN
# ============================================================

@torch.no_grad()
def encode(model, images, batch_size=2048):
    model.eval()
    out = []
    for start in range(0, images.size(0), batch_size):
        batch = images[start:start + batch_size].to(DEVICE)
        out.append(model.embed(batch))
    return torch.cat(out)


@torch.no_grad()
def knn_predict(query_feats, ref_feats, ref_labels, k, chunk_size=2048):
    query_feats = query_feats.to(DEVICE)
    ref_feats = ref_feats.to(DEVICE)
    ref_labels = ref_labels.to(DEVICE)
    k = min(k, ref_feats.size(0))

    preds = []
    for start in range(0, query_feats.size(0), chunk_size):
        q_chunk = query_feats[start:start + chunk_size]
        
        dist_sq = (
            q_chunk.pow(2).sum(dim=1, keepdim=True)
            + ref_feats.pow(2).sum(dim=1).unsqueeze(0)
            - 2 * (q_chunk @ ref_feats.T)
        )
        
        _, topk_indices = torch.topk(dist_sq, k, dim=1, largest=False, sorted=False)
        topk_labels = ref_labels[topk_indices]
        for i in range(topk_labels.size(0)):
            counts = torch.bincount(topk_labels[i], minlength=10)
            preds.append(counts.argmax().item())

    return torch.tensor(preds, device="cpu")


def choose_labeled_indices(train_labels, per_class, seed=0):
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
    print("SimCLR LATENT-SPACE ENSEMBLE (Optimized)")
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

    labeled_idx = {s: choose_labeled_indices(train_y, s) for s in LABELS_PER_CLASS}

    all_preds = {s: [] for s in LABELS_PER_CLASS}
    per_space_acc = {s: [] for s in LABELS_PER_CLASS}

    for m in range(NUM_MODELS):
        print("\n" + "=" * 70)
        print(f"LATENT SPACE {m + 1}/{NUM_MODELS}")
        print("=" * 70)

        seed = 1000 + m
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        subset = torch.randperm(train_x.size(0))[:TRAIN_SUBSET]

        model = SimCLRModel(LATENT_DIM).to(DEVICE)
        train_simclr(model, train_x[subset])
        print("\n    Encoder training complete and frozen.")

        train_feats = encode(model, train_x)
        test_feats = encode(model, test_x)

        for s in LABELS_PER_CLASS:
            idx = labeled_idx[s]
            preds = knn_predict(test_feats, train_feats[idx], train_y[idx], K_NEIGHBORS)
            acc = (preds == test_y).float().mean().item()
            all_preds[s].append(preds)
            per_space_acc[s].append(acc)
            print(f"    {setting_name(s):>16}: {acc * 100:.2f}%")

        elapsed = (time.time() - start_time) / 60
        print(f"    (elapsed: {elapsed:.1f} min)")

    # ---- raw-pixel kNN baseline ----
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
        stacked = torch.stack(all_preds[s]).numpy()
        ensemble_preds = torch.tensor([
            np.argmax(np.bincount(stacked[:, i], minlength=10))
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
# RUN
# ============================================================

class Tee:
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as output_file:
        output = Tee(sys.stdout, output_file)
        with redirect_stdout(output), redirect_stderr(output):
            main()