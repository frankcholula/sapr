import logging
import numpy as np
import pickle
import os
from pathlib import Path
from custom_hmm import HMM
from hmmlearn_hmm import HMMLearnModel
from typing import List, Literal, Dict
from mfcc_extract import load_mfccs, load_mfccs_by_word
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def plot_training_progress(log_likelihoods: List[float], model_name: str) -> None:
    plt.figure(figsize=(10, 6))
    iterations = range(len(log_likelihoods))

    # Plot line and points
    plt.plot(iterations, log_likelihoods, "b-", linewidth=2, label="Log Likelihood")
    plt.plot(iterations, log_likelihoods, "bo", markersize=4)

    # Calculate improvement
    total_improvement = log_likelihoods[-1] - log_likelihoods[0]

    # Labels and title
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Log Likelihood", fontsize=12)
    plt.title(
        f"Training Progress for `{model_name}`\nTotal Improvement: {total_improvement:.2f}",
        fontsize=14,
    )

    # Grid and layout
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Print numeric summary
    print("\nTraining Summary:")
    print(f"Initial log likelihood: {log_likelihoods[0]:.2f}")
    print(f"Final log likelihood: {log_likelihoods[-1]:.2f}")
    print(f"Total improvement: {total_improvement:.2f}")


def pretty_print_matrix(matrix: np.ndarray, precision: int = 3) -> None:
    n = matrix.shape[0]
    df = pd.DataFrame(
        matrix,
        columns=[
            f"S{i}" if i != 0 and i != n - 1 else ("Entry" if i == 0 else "Exit")
            for i in range(n)
        ],
        index=[
            f"S{i}" if i != 0 and i != n - 1 else ("Entry" if i == 0 else "Exit")
            for i in range(n)
        ],
    )

    row_sums = df.sum(axis=1).round(precision)
    df = df.replace(0, ".")

    print("\nTransition Matrix:")
    print("==================")
    print(df.round(precision))
    assert np.allclose(row_sums, 1.0), "Row sums should be equal to 1.0"


def save_model(model, model_path: Path) -> None:
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Saved model to {model_path}")


def train_hmm(implementation: Literal["custom", "hmmlearn"] = "hmmlearn") -> Dict[str, HMM]:
    vocabs = [
        "heed",
        "hid",
        "head",
        "had",
        "hard",
        "hud",
        "hod",
        "hoard",
        "hood",
        "whod",
        "heard",
    ]

    models_dir = Path("trained_models")
    models_dir.mkdir(exist_ok=True)

    feature_set = load_mfccs("feature_set")
    features = {word: load_mfccs_by_word("feature_set", word) for word in vocabs}
    total_features_length = sum(len(features[word]) for word in vocabs)
    assert total_features_length == len(feature_set)

    hmms = {}
    for word in vocabs:
        logging.info(f"\nTraining model for word: {word}")

        # Define model path based on implementation
        model_path = models_dir / f"{word}_{implementation}.pkl"

        if implementation == "custom":
            hmm = HMM(8, 13, feature_set, model_name=word)
            log_likelihoods = hmm.baum_welch(features[word], 15)
            trained_model = hmm
        else:
            hmm = HMMLearnModel(num_states=8, model_name=word)
            trained_model, _ = hmm.fit(features[word])
            log_likelihoods = hmm.model.monitor_.history

        plot_training_progress(log_likelihoods, word)
        save_model(trained_model, model_path)
        hmms[word] = hmm

    return hmms


if __name__ == "__main__":
    # Use original implementation
    # print("Training with `custom` implementation:")
    # train_hmm("custom")

    # Use hmmlearn implementation
    print("\nTraining with `hmmlearn` implementation:")
    train_hmm("hmmlearn")
