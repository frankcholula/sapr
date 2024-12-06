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


def plot_training_error():
    """
    Plot error rate vs training iterations from saved models.
    """
    import matplotlib.pyplot as plt
    import pickle
    from pathlib import Path
    import numpy as np

    # Create figures directory if it doesn't exist
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    # Words in vocabulary
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

    # Load models and extract training history
    histories = {}
    models_dir = Path("trained_models/hmmlearn")

    for word in words:
        model_path = models_dir / f"{word}_hmmlearn_15.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                model = pickle.load(f)
                # Extract log probabilities from training history
                histories[word] = model.monitor_.history

    # Plot training curves
    plt.figure(figsize=(12, 8))

    for word, history in histories.items():
        # Convert log probability to error rate (normalized negative log probability)
        error_rates = [-log_prob / max(-np.array(history)) for log_prob in history]
        plt.plot(range(1, len(history) + 1), error_rates, marker="o", label=word)

    plt.xlabel("Iteration")
    plt.ylabel("Normalized Error Rate")
    plt.title("Training Error Rate vs Iterations")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(figures_dir / "training_error_rates.png")
    plt.close()


def plot_error_rates(dev_results, eval_results):
    """
    Plot recognition error rates for both development and evaluation sets
    """

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    # Get per-word error rates for both sets
    def get_word_error_rates(results):
        word_error_rates = {}
        for word, word_results in results["results"].items():
            correct = sum(r["correct"] for r in word_results)
            total = len(word_results)
            word_error_rates[word] = 1 - (correct / total)  # Convert to error rate
        return word_error_rates

    dev_errors = get_word_error_rates(dev_results)
    eval_errors = get_word_error_rates(eval_results)

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Bar plot comparison
    words = list(dev_errors.keys())
    x = range(len(words))
    width = 0.35

    # Plot error rates (not accuracies)
    ax1.bar(
        [i - width / 2 for i in x],
        list(dev_errors.values()),
        width,
        label="Development",
        color="blue",
        alpha=0.6,
    )
    ax1.bar(
        [i + width / 2 for i in x],
        list(eval_errors.values()),
        width,
        label="Evaluation",
        color="red",
        alpha=0.6,
    )

    ax1.set_ylabel("Error Rate")
    ax1.set_title("Recognition Error Rate by Word")
    ax1.set_xticks(x)
    ax1.set_xticklabels(words, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Overall error rates (not accuracies)
    overall_dev_error = dev_results["accuracy"]  # These are already error rates
    overall_eval_error = eval_results["accuracy"]

    ax2.bar(
        ["Development", "Evaluation"],
        [overall_dev_error, overall_eval_error],
        color=["blue", "red"],
        alpha=0.6,
    )
    ax2.set_ylabel("Error Rate")
    ax2.set_title("Overall Recognition Error Rate")
    ax2.grid(True, alpha=0.3)

    # Add error rate values on top of bars
    for i, v in enumerate([overall_dev_error, overall_eval_error]):
        ax2.text(i, v, f"{v:.1%}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(figures_dir / "recognition_error_rates.png")
    plt.close()


if __name__ == "__main__":
    compare_utterances_pca()
    plot_training_error()
