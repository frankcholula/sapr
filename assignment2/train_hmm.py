import logging
import numpy as np
import pickle
import pandas as pd
import os
from pathlib import Path
from custom_hmm import HMM
from hmmlearn_hmm import HMMLearnModel
from typing import List, Literal, Dict, Union
from mfcc_extract import load_mfccs, load_mfccs_by_word
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def plot_training_progress(all_likelihoods: Dict[str, List[float]]) -> None:
    num_models = len(all_likelihoods)
    # Calculate number of rows and columns for subplots
    n_cols = 4
    n_rows = (num_models + n_cols - 1) // n_cols  # Ceiling division

    plt.figure(figsize=(15, 3 * n_rows))

    for idx, (word, log_likelihoods) in enumerate(all_likelihoods.items(), 1):
        plt.subplot(n_rows, n_cols, idx)
        iterations = range(len(log_likelihoods))

        # Plot line and points
        plt.plot(iterations, log_likelihoods, "b-", linewidth=2)
        plt.plot(iterations, log_likelihoods, "bo", markersize=4)

        # Calculate improvement
        total_improvement = log_likelihoods[-1] - log_likelihoods[0]

        plt.xlabel("Iteration", fontsize=10)
        plt.ylabel("Log Likelihood", fontsize=10)
        plt.title(f"{word}\nImprovement: {total_improvement:.2f}", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        # Print numeric summary
        print(f"\nTraining Summary for {word}:")
        print(f"Initial log likelihood: {log_likelihoods[0]:.2f}")
        print(f"Final log likelihood: {log_likelihoods[-1]:.2f}")
        print(f"Total improvement: {total_improvement:.2f}")

    plt.tight_layout()
    plt.show()


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


def train_hmm(
    implementation: Literal["custom", "hmmlearn"] = "hmmlearn",
    num_states: int = 8,
    num_features: int = 13,
) -> Dict[str, Union[HMM, HMMLearnModel]]:
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
    training_histories = {}

    for word in vocabs:
        logging.info(f"\nTraining model for word: {word}")
        model_path = os.path.join(models_dir, f"{word}_{implementation}.pkl")

        if implementation == "custom":
            hmm = HMM(num_states, num_features, feature_set, model_name=word)
            log_likelihoods = hmm.baum_welch(features[word], 15)
            trained_model = hmm
        else:
            hmm = HMMLearnModel(num_states=num_states, model_name=word)
            trained_model, _ = hmm.fit(features[word])
            log_likelihoods = hmm.model.monitor_.history

        training_histories[word] = log_likelihoods
        save_model(trained_model, model_path)
        hmms[word] = hmm

    plot_training_progress(training_histories)
    return hmms


if __name__ == "__main__":
    print("\nTraining with `hmmlearn` implementation:")
    train_hmm("hmmlearn", num_states=8, num_features=13)
