import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mfcc_extract import load_mfccs_by_word
from pathlib import Path


def compare_utterances_pca():
    """
    Compare utterances between feature_set and eval_set using PCA.
    """

    # Create figures directory if it doesn't exist
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    # Words to compare
    words = [
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

    # Setup plot with only 11 subplots (4x3 grid)
    fig = plt.figure(figsize=(15, 20))

    for idx, word in enumerate(words):
        # Create subplot
        plt.subplot(4, 3, idx + 1)

        # Load features from both sets
        dev_features = load_mfccs_by_word("feature_set", word)[
            0
        ].T  # Taking first utterance
        eval_features = load_mfccs_by_word("eval_feature_set", word)[
            0
        ].T  # Taking first utterance

        # Combine and scale features
        combined = np.vstack([dev_features, eval_features])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(combined)

        # Apply PCA
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(scaled)

        # Split back into dev and eval
        dev_transformed = transformed[: len(dev_features)]
        eval_transformed = transformed[len(dev_features) :]

        # Plot
        plt.scatter(
            dev_transformed[:, 0],
            dev_transformed[:, 1],
            c="blue",
            label="Development",
            alpha=0.6,
        )
        plt.scatter(
            eval_transformed[:, 0],
            eval_transformed[:, 1],
            c="red",
            label="Evaluation",
            alpha=0.6,
        )
        plt.title(f'"{word}"')
        plt.grid(True)
        if idx == 0:  # Only show legend for first plot
            plt.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "pca_comparison.png")
    plt.close()


if __name__ == "__main__":
    compare_utterances_pca()
