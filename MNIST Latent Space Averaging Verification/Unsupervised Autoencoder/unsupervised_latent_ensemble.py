"""
UNSUPERVISED 2D LATENT-SPACE ENSEMBLE FOR MNIST
================================================

PURPOSE
-------
This program tests whether UNSUPERVISED latent spaces naturally organize
MNIST images in a way that is useful for classification.

We train 20 completely independent autoencoders.

Each autoencoder compresses:

    784 pixel values -> 2 latent values

So every MNIST image becomes a point:

    (x, y)

in a two-dimensional latent space.


IMPORTANT EXPERIMENTAL IDEA
---------------------------
NO digit labels are used to train the autoencoders.

The autoencoder only learns:

    Image -> (x,y) -> Reconstructed Image

After training is COMPLETELY finished, the encoder is frozen.

We then:

    1. Pass all training images through the frozen encoder.
    2. Record their (x,y) coordinates.
    3. Attach their known labels to those coordinates.

For a NEW image:

    1. Use the frozen encoder to obtain its (x,y).
    2. Calculate its distance to every reference point.
    3. Find the K nearest points.
    4. Look at their labels.
    5. Use the local majority label as the prediction.

There is NO trained classifier.

The classification step is simply:

    Euclidean distance + local majority voting


WHY 20 LATENT SPACES?
---------------------
Each autoencoder starts with different random weights and experiences
a different shuffled training process.

Therefore each autoencoder may discover a different organization of MNIST.

For one unknown image:

    Space 1  -> 7
    Space 2  -> 7
    Space 3  -> 9
    ...
    Space 20 -> 7

Then the 20 spaces vote.

The most common prediction becomes the final prediction.
"""

# ============================================================
# IMPORTS
# ============================================================

import random
import sys
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# CONFIGURATION
# ============================================================

# Number of independently learned latent spaces.
NUM_MODELS = 25

# We specifically want a TWO-DIMENSIONAL latent space.
#
# Every MNIST image will become:
#
#     (x, y)
#
LATENT_DIM = 64

# Number of times each autoencoder sees the entire training dataset.
EPOCHS = 10

# Number of images processed at once.
BATCH_SIZE = 256

# Learning rate for Adam.
LEARNING_RATE = 1e-3

# Number of nearby reference points used to determine the local class.
#
# Using an odd number helps reduce ties.
K_NEIGHBORS = 5

# Automatically use GPU if available.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# AUTOENCODER
# ============================================================

class Autoencoder(nn.Module):
    """
    Autoencoder used to learn ONE 2D latent space.

    Architecture:

        784 pixels
            ↓
          Encoder
            ↓
          (x, y)
            ↓
          Decoder
            ↓
        784 pixels

    The network is trained ONLY to reconstruct its input.

    It receives no information about which digit an image represents.
    """

    def __init__(self, latent_dim=2):

        super().__init__()

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------
        #
        # Compress:
        #
        # 784 -> 256 -> 128 -> 2

        self.encoder = nn.Sequential(

            nn.Linear(28 * 28, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            # This layer creates our latent coordinates.
            nn.Linear(128, latent_dim)
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------
        #
        # Reconstruct:
        #
        # 2 -> 128 -> 256 -> 784

        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 28 * 28),

            # MNIST pixel values are between 0 and 1.
            nn.Sigmoid()
        )

    def forward(self, x):

        # Convert:
        #
        # [batch, 1, 28, 28]
        #
        # into:
        #
        # [batch, 784]

        x = x.view(x.size(0), -1)

        # Compress the image into the latent space.
        z = self.encoder(x)

        # Attempt to reconstruct the image.
        reconstruction = self.decoder(z)

        return reconstruction, z


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):
    """
    Set random seeds.

    Every autoencoder gets a DIFFERENT seed.

    Different seeds cause different initial neural-network weights.

    This helps our 20 autoencoders discover different latent spaces.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# TRAIN ONE AUTOENCODER
# ============================================================

def train_autoencoder(model, train_loader):
    """
    Train ONE autoencoder.

    IMPORTANT:

    Labels are deliberately ignored.

    The autoencoder's only objective is:

        reconstructed image ~= original image

    Therefore this representation-learning process is unsupervised.
    """

    model.train()

    # Measures the difference between the original image
    # and reconstructed image.
    reconstruction_loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        total_loss = 0.0

        # The dataset gives us labels, but "_labels" indicates
        # that we intentionally do not use them.
        for images, _labels in train_loader:

            images = images.to(DEVICE)

            # Flatten original images so they can be compared
            # with the decoder output.
            flattened_images = images.view(
                images.size(0),
                -1
            )

            optimizer.zero_grad()

            reconstruction, latent = model(images)

            loss = reconstruction_loss_function(
                reconstruction,
                flattened_images
            )

            # Calculate gradients.
            loss.backward()

            # Update the autoencoder.
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(
            f"    Epoch {epoch + 1:02d}/{EPOCHS}"
            f" | Reconstruction Loss: {average_loss:.6f}"
        )


# ============================================================
# EXTRACT LATENT POINTS
# ============================================================

def extract_latent_features(model, loader):
    """
    Pass a dataset through a FROZEN encoder.

    For every image we obtain:

        image -> encoder -> (x,y)

    We return:

        reference_points
        reference_labels

    Example:

        reference_points:

            [
                [-2.1,  1.7],
                [ 3.4, -0.2],
                [ 0.8,  2.9],
                ...
            ]

        reference_labels:

            [
                1,
                7,
                3,
                ...
            ]

    IMPORTANT:

    These labels DO NOT train or change the encoder.

    The encoder has already finished learning.

    We are simply attaching labels to locations that already exist.
    """

    model.eval()

    all_points = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            flattened_images = images.view(
                images.size(0),
                -1
            )

            # Ask the encoder where each image belongs.
            latent_points = model.encoder(
                flattened_images
            )

            all_points.append(
                latent_points.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

    points = np.concatenate(
        all_points,
        axis=0
    )

    labels = np.concatenate(
        all_labels,
        axis=0
    )

    return points, labels


# ============================================================
# CLASSIFY USING ONLY LATENT-SPACE DISTANCE
# ============================================================

def classify_from_latent_space(
    new_point,
    reference_points,
    reference_labels,
    k=5
):
    """
    Classify ONE new latent point.

    THERE IS NO MACHINE-LEARNING TRAINING IN THIS FUNCTION.

    We simply:

        1. Calculate distance to every reference point.
        2. Find the K closest points.
        3. Look at their labels.
        4. Return the most common label.

    Example:

        New point:
            (2.5, -1.3)

        Five nearest labels:
            7
            7
            7
            9
            7

        Prediction:
            7
    """

    # --------------------------------------------------------
    # CALCULATE EUCLIDEAN DISTANCE
    # --------------------------------------------------------
    #
    # For every reference point:
    #
    # distance =
    #
    # sqrt(
    #     (reference_x - new_x)^2
    #     +
    #     (reference_y - new_y)^2
    # )

    distances = np.linalg.norm(
        reference_points - new_point,
        axis=1
    )

    # --------------------------------------------------------
    # FIND K CLOSEST POINTS
    # --------------------------------------------------------

    # argpartition is faster than sorting ALL 60,000 distances.
    nearest_indices = np.argpartition(
        distances,
        k - 1
    )[:k]

    # Get the labels belonging to those nearby points.
    nearest_labels = reference_labels[
        nearest_indices
    ]

    # --------------------------------------------------------
    # LOCAL MAJORITY VOTE
    # --------------------------------------------------------

    vote_counts = np.bincount(
        nearest_labels.astype(int),
        minlength=10
    )

    prediction = np.argmax(
        vote_counts
    )

    return int(prediction)


# ============================================================
# PREDICT ONE IMAGE USING ALL 20 LATENT SPACES
# ============================================================

def predict_one_image(
    image,
    models,
    reference_points_list,
    reference_labels_list,
    k=5
):
    """
    Classify ONE unknown MNIST image.

    The same image is passed through all 20 encoders.

    Each encoder produces its own (x,y) coordinate.

    Each latent space then determines the local class using
    nearest-neighbor lookup.

    Finally, the 20 predictions vote.
    """

    predictions = []

    # Add batch dimension.
    #
    # [1, 28, 28]
    #
    # becomes:
    #
    # [1, 1, 28, 28]

    image = image.unsqueeze(0).to(DEVICE)

    flattened_image = image.view(
        1,
        -1
    )

    # --------------------------------------------------------
    # ASK EVERY LATENT SPACE
    # --------------------------------------------------------

    for (
        model,
        reference_points,
        reference_labels
    ) in zip(
        models,
        reference_points_list,
        reference_labels_list
    ):

        model.eval()

        with torch.no_grad():

            latent_point = model.encoder(
                flattened_image
            )

        # Convert:
        #
        # [[x,y]]
        #
        # into:
        #
        # [x,y]

        latent_point = (
            latent_point
            .cpu()
            .numpy()[0]
        )

        # Determine which class dominates this local region.
        prediction = classify_from_latent_space(
            latent_point,
            reference_points,
            reference_labels,
            k=k
        )

        predictions.append(
            prediction
        )

    # --------------------------------------------------------
    # ENSEMBLE VOTE
    # --------------------------------------------------------

    vote_counts = Counter(
        predictions
    )

    final_prediction = (
        vote_counts
        .most_common(1)[0][0]
    )

    return (
        final_prediction,
        predictions,
        vote_counts
    )


# ============================================================
# EVALUATE ONE LATENT SPACE
# ============================================================

def evaluate_single_latent_space(
    model,
    reference_points,
    reference_labels,
    test_loader,
    k=5
):
    """
    Determine how accurate ONE latent space is by itself.

    This is important because we want to compare:

        individual latent-space accuracy

    against:

        20-space ensemble accuracy
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            flattened_images = images.view(
                images.size(0),
                -1
            )

            # Convert entire test batch into 2D coordinates.
            latent_points = model.encoder(
                flattened_images
            )

            latent_points = (
                latent_points
                .cpu()
                .numpy()
            )

            labels = labels.numpy()

            # Classify each point using only local geometry.
            for point, true_label in zip(
                latent_points,
                labels
            ):

                prediction = classify_from_latent_space(
                    point,
                    reference_points,
                    reference_labels,
                    k=k
                )

                if prediction == int(true_label):

                    correct += 1

                total += 1

    return correct / total


# ============================================================
# EVALUATE THE COMPLETE ENSEMBLE
# ============================================================

def evaluate_ensemble(
    models,
    reference_points_list,
    reference_labels_list,
    test_loader,
    k=5
):
    """
    Evaluate the majority vote from all 20 latent spaces.
    """

    correct = 0
    total = 0

    for images, labels in test_loader:

        for image, true_label in zip(
            images,
            labels
        ):

            final_prediction, _, _ = predict_one_image(
                image,
                models,
                reference_points_list,
                reference_labels_list,
                k=k
            )

            if final_prediction == int(true_label):

                correct += 1

            total += 1

    return correct / total


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("UNSUPERVISED 2D LATENT-SPACE ENSEMBLE")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")
    print(f"Number of latent spaces: {NUM_MODELS}")
    print(f"Latent dimensions: {LATENT_DIM}")
    print(f"Nearest neighbors: {K_NEIGHBORS}")
    print()

    # Convert MNIST images into PyTorch tensors.
    transform = transforms.ToTensor()

    # --------------------------------------------------------
    # LOAD MNIST
    # --------------------------------------------------------

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # --------------------------------------------------------
    # IMPORTANT: TWO TRAIN LOADERS
    # --------------------------------------------------------
    #
    # During autoencoder training we WANT random shuffling.
    #
    # When extracting reference points, shuffling isn't necessary.
    # Keeping it off also makes the reference map reproducible.

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    reference_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    models = []

    # Each element contains the 60,000 coordinates belonging
    # to one latent space.
    reference_points_list = []

    # Each element contains the corresponding labels.
    reference_labels_list = []

    individual_accuracies = []

    # --------------------------------------------------------
    # TRAIN 20 INDEPENDENT LATENT SPACES
    # --------------------------------------------------------

    for model_number in range(
        NUM_MODELS
    ):

        print("\n" + "=" * 70)

        print(
            f"LATENT SPACE "
            f"{model_number + 1}/{NUM_MODELS}"
        )

        print("=" * 70)

        # Every model starts differently.
        seed = 1000 + model_number

        set_seed(seed)

        # Create a brand-new autoencoder.
        model = Autoencoder(
            latent_dim=LATENT_DIM
        ).to(DEVICE)

        # ----------------------------------------------------
        # LEARN LATENT SPACE
        # ----------------------------------------------------
        #
        # NO LABELS influence this process.

        train_autoencoder(
            model,
            train_loader
        )

        print(
            "\n    Latent-space learning complete."
        )

        print(
            "    Encoder is now frozen for evaluation."
        )

        # ----------------------------------------------------
        # CREATE REFERENCE MAP
        # ----------------------------------------------------

        print(
            "    Creating labeled reference map..."
        )

        reference_points, reference_labels = (
            extract_latent_features(
                model,
                reference_loader
            )
        )

        # Store this model.
        models.append(
            model
        )

        # Store this model's 2D reference map.
        reference_points_list.append(
            reference_points
        )

        reference_labels_list.append(
            reference_labels
        )

        # ----------------------------------------------------
        # INDIVIDUAL ACCURACY
        # ----------------------------------------------------

        print(
            "    Evaluating this latent space..."
        )

        accuracy = evaluate_single_latent_space(
            model,
            reference_points,
            reference_labels,
            test_loader,
            k=K_NEIGHBORS
        )

        individual_accuracies.append(
            accuracy
        )

        print(
            f"    Accuracy: {accuracy * 100:.2f}%"
        )

    # ========================================================
    # INDIVIDUAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("INDIVIDUAL LATENT-SPACE RESULTS")
    print("=" * 70)

    for index, accuracy in enumerate(
        individual_accuracies,
        start=1
    ):

        print(
            f"Latent Space {index:02d}: "
            f"{accuracy * 100:.2f}%"
        )

    mean_accuracy = np.mean(
        individual_accuracies
    )

    std_accuracy = np.std(
        individual_accuracies
    )

    print(
        f"\nMean individual accuracy: "
        f"{mean_accuracy * 100:.2f}%"
    )

    print(
        f"Standard deviation: "
        f"{std_accuracy * 100:.2f}%"
    )

    # ========================================================
    # DEMONSTRATE ONE IMAGE
    # ========================================================

    sample_image, true_label = (
        test_dataset[0]
    )

    (
        final_prediction,
        predictions,
        votes
    ) = predict_one_image(
        sample_image,
        models,
        reference_points_list,
        reference_labels_list,
        k=K_NEIGHBORS
    )

    print("\n" + "=" * 70)
    print("SINGLE IMAGE DEMONSTRATION")
    print("=" * 70)

    print(
        f"\nActual digit: {true_label}"
    )

    print(
        "\nPredictions from the 20 latent spaces:"
    )

    for index, prediction in enumerate(
        predictions,
        start=1
    ):

        print(
            f"    Space {index:02d}: "
            f"{prediction}"
        )

    print("\nFinal vote counts:")

    for digit, count in sorted(
        votes.items()
    ):

        print(
            f"    Digit {digit}: "
            f"{count} votes"
        )

    print(
        f"\nFINAL PREDICTION: "
        f"{final_prediction}"
    )

    # ========================================================
    # COMPLETE ENSEMBLE TEST
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "EVALUATING COMPLETE "
        "20-SPACE ENSEMBLE..."
    )

    print("=" * 70)

    ensemble_accuracy = evaluate_ensemble(
        models,
        reference_points_list,
        reference_labels_list,
        test_loader,
        k=K_NEIGHBORS
    )

    print(
        f"\nMean Individual Accuracy: "
        f"{mean_accuracy * 100:.2f}%"
    )

    print(
        f"Ensemble Accuracy: "
        f"{ensemble_accuracy * 100:.2f}%"
    )

    improvement = (
        ensemble_accuracy
        -
        mean_accuracy
    )

    print(
        f"Ensemble vs Mean Improvement: "
        f"{improvement * 100:+.2f} percentage points"
    )


# ============================================================
# RUN PROGRAM
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
    output_dir = Path(__file__).parent / "Iterations" / "25 Spaces"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{LATENT_DIM} Latent Dimensions.txt"

    with output_path.open("w", encoding="utf-8") as output_file:
        output = Tee(sys.stdout, output_file)

        with redirect_stdout(output), redirect_stderr(output):
            main()