import logging
import numpy as np
from hmm import HMM
from hmmlearn_model import HMMLearnModel
from typing import Literal, List
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


def train_hmm(implementation: Literal["custom", "hmmlearn"] = "custom"):
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

    feature_set = load_mfccs("feature_set")
    features = {word: load_mfccs_by_word("feature_set", word) for word in vocabs}
    total_features_length = sum(len(features[word]) for word in vocabs)
    assert total_features_length == len(feature_set)

    hmms = {}
    for word in vocabs:
        if implementation == "custom":
            hmms[word] = HMM(8, 13, feature_set, model_name=word)
        else:
            hmms[word] = HMMLearnModel(num_states=8, model_name=word)

    for word, hmm in hmms.items():
        if implementation == "custom":
            log_likelihoods = hmm.baum_welch(features[word], 15)
        else:
            hmm.fit(features[word])
            log_likelihoods = hmm.model.monitor_.history
        plot_training_progress(log_likelihoods, word)


if __name__ == "__main__":
    # Use original implementation
    # print("Training with custom implementation:")
    # train_hmm("custom")

    # Use hmmlearn implementation
    print("\nTraining with hmmlearn implementation:")
    train_hmm("hmmlearn")
