import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Dict, Union, List, Tuple
from sklearn.metrics import confusion_matrix, accuracy_score
from decoder import Decoder
import seaborn as sns
import pickle
from mfcc_extract import load_mfccs_by_word
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def extract_labels(all_results: Dict) -> Tuple[List[str], List[str]]:
    true_labels = []
    predicted_labels = []
    
    for results in all_results.values():
        for result in results:
            true_labels.append(result["true_word"])
            predicted_labels.append(result["predicted_word"])
            
    return true_labels, predicted_labels


def calculate_metrics(true_labels: List[str], predicted_labels: List[str], vocab: List[str]) -> Tuple[np.ndarray, float]:
    label_mapping = {word: idx for idx, word in enumerate(vocab)}
    true_labels_idx = [label_mapping[label] for label in true_labels]
    predicted_labels_idx = [label_mapping[label] for label in predicted_labels]

    cm = confusion_matrix(true_labels_idx, predicted_labels_idx)
    accuracy = accuracy_score(true_labels_idx, predicted_labels_idx)
    
    return cm, accuracy


def plot_confusion_matrix(cm: np.ndarray, vocab: List[str], implementation: str, feature_set_path: str) -> None:
    plt.figure(figsize=(12, 10))
    
    mask_correct = np.zeros_like(cm, dtype=bool)
    np.fill_diagonal(mask_correct, True)
    
    cm_correct = np.ma.masked_array(cm, ~mask_correct)
    cm_incorrect = np.ma.masked_array(cm, mask_correct)
    
    sns.heatmap(cm_incorrect, annot=True, cmap='Reds', fmt='d',
                xticklabels=vocab, yticklabels=vocab, cbar=False)
    
    sns.heatmap(cm_correct, annot=True, cmap='Greens', fmt='d',
                xticklabels=vocab, yticklabels=vocab, cbar=False)
    
    plt.title(f'Confusion Matrix - {implementation} ({Path(feature_set_path).stem})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    
    output_path = figures_dir / f"{implementation}_confusion_matrix_{Path(feature_set_path).stem}.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logging.info(f"\nSaved confusion matrix plot to: {output_path}")


def log_per_word_accuracy(all_results: Dict) -> None:
    logging.info("\nPer-word accuracy:")
    for word, results in all_results.items():
        word_correct = sum(r["correct"] for r in results)
        word_total = len(results)
        word_accuracy = word_correct / word_total
        logging.info(f"{word}: {word_accuracy:.2%}")


def eval_hmm(
    implementation: Literal["custom", "hmmlearn"] = "hmmlearn",
    feature_set_path: str = "eval_feature_set",
    model_iter: int = 15
) -> Dict[str, Union[dict, float, pd.DataFrame]]:
    decoder = Decoder(implementation=implementation, n_iter=model_iter)
    all_results = decoder.decode_vocabulary(feature_set_path, verbose=False)

    true_labels, predicted_labels = extract_labels(all_results)
    cm, accuracy = calculate_metrics(true_labels, predicted_labels, decoder.vocab)
    
    cm_df = pd.DataFrame(cm, index=decoder.vocab, columns=decoder.vocab)
    logging.info(f"\nConfusion Matrix:\n{cm_df}")
    logging.info(f"\nOverall Accuracy: {accuracy:.2%}")

    log_per_word_accuracy(all_results)
    plot_confusion_matrix(cm, decoder.vocab, implementation, feature_set_path)

    return {
        "results": all_results,
        "accuracy": accuracy,
        "confusion_matrix": cm_df,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
    }



def load_models(models_dir: str, implementation: str, epoch: int) -> Tuple[Dict, List[str]]:
    """
    Load HMM models for a specific epoch.
    """
    models = {}
    vocab = []
    impl_dir = Path(models_dir) / implementation
    pattern = f"*_{implementation}_epoch_{epoch}.pkl"
    
    for model_path in impl_dir.glob(pattern):
        word = model_path.stem.split("_")[0]
        with open(model_path, "rb") as f:
            models[word] = pickle.load(f)
            vocab.append(word)
    
    if not models:
        raise ValueError(f"No models found in {impl_dir} with pattern {pattern}")
    
    logging.info(f"Loaded {len(models)} models from {impl_dir} for words: {', '.join(vocab)}")
    return models, vocab


def decode_sequence(models: Dict, features: np.ndarray) -> Tuple[str, float, List[int]]:
    """
    Decode a single sequence using all models.
    """
    best_score = float("-inf")
    best_word = None
    best_states = None

    for word, model in models.items():
        log_prob, states = model.decode(features)
        if log_prob > best_score:
            best_score = log_prob
            best_word = word
            best_states = states

    return best_word, best_score, best_states


def decode_word_samples(word: str, vocab: List[str], models: Dict, feature_set: str = "feature_set") -> List[Dict]:
    """
    Decode all samples for a given word.
    """
    if word not in vocab:
        raise ValueError(f"Word '{word}' not in vocabulary: {vocab}")
        
    results = []
    features = load_mfccs_by_word(feature_set, word)
    
    for i, feat_seq in enumerate(features):
        test_features = feat_seq.T
        predicted_word, log_prob, state_sequence = decode_sequence(models, test_features)
        
        result = {
            "sample_index": i + 1,
            "true_word": word,
            "predicted_word": predicted_word,
            "log_likelihood": log_prob,
            "correct": predicted_word == word,
            "state_sequence": state_sequence
        }
        results.append(result)
        
    return results


def decode_vocabulary(models: Dict, vocab: List[str], feature_set: str = "feature_set") -> Dict[str, List[Dict]]:
    """
    Decode all words in the vocabulary.
    """
    all_results = {}
    
    for word in vocab:
        results = decode_word_samples(word, vocab, models, feature_set)
        all_results[word] = results
        
        correct = sum(r["correct"] for r in results)
        total = len(results)
        logging.info(f"Results for '{word}': Accuracy: {correct}/{total} ({correct/total:.1%})")
        
    return all_results




def plot_error_rate_all_classes_over_epochs(class_error_rates: Dict[str, List[float]], feature_set: str, figures_dir: str = "figures") -> None:
    """
    Plot and save error rate for all classes (words) over epochs in a single figure.
    """
    figures_path = Path(figures_dir)
    figures_path.mkdir(exist_ok=True)

    plt.figure(figsize=(12, 8))

    for word, error_rates in class_error_rates.items():
        epochs = range(len(error_rates))
        plt.plot(epochs, error_rates, marker='o', label=word)

    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.title(f"Error Rate Over Epochs for Feature Set: {feature_set}")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_path / f"{feature_set}_error_rate_over_epochs.png")
    plt.close()


def plot_overall_error_rate_over_epochs(overall_error_rates: List[float], feature_set: str, figures_dir: str = "figures") -> None:
    """
    Plot and save the overall error rate across epochs in a single figure.
    """
    figures_path = Path(figures_dir)
    figures_path.mkdir(exist_ok=True)

    epochs = range(len(overall_error_rates))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, overall_error_rates, marker='o', color='r')
    plt.xlabel("Epoch")
    plt.ylabel("Overall Error Rate")
    plt.title(f"Overall Error Rate Over Epochs for Feature Set: {feature_set}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_path / f"{feature_set}_overall_error_rate_over_epochs.png")
    plt.close()


def eval_hmm_every_epoch(models_dir: str, implementation: str, n_iter: int, feature_set: str = "feature_set", figures_dir: str = "figures") -> None:
    """
    Evaluate models for all epochs, log error rate, and save plots for class-wise and overall error rates over epochs.
    """
    class_error_rates = {}
    overall_error_rates = []

    for epoch in range(n_iter):
        try:
            models, vocab = load_models(models_dir, implementation, epoch)
        except ValueError as e:
            logging.warning(f"Skipping epoch {epoch}: {e}")
            continue

        logging.info(f"Evaluating models from epoch {epoch}...")
        results = decode_vocabulary(models, vocab, feature_set)

        # Calculate error rate per word
        total_correct = 0
        total_samples = 0
        for word in vocab:
            word_results = results[word]
            correct = sum(r["correct"] for r in word_results)
            total = len(word_results)
            error_rate = 1 - (correct / total if total > 0 else 0)

            if word not in class_error_rates:
                class_error_rates[word] = []
            class_error_rates[word].append(error_rate)

            total_correct += correct
            total_samples += total

        # Calculate overall error rate for the epoch
        overall_error_rate = 1 - (total_correct / total_samples if total_samples > 0 else 0)
        overall_error_rates.append(overall_error_rate)

    # Plot error rate for all classes over epochs
    plot_error_rate_all_classes_over_epochs(class_error_rates, feature_set, figures_dir)

    # Plot overall error rate over epochs
    plot_overall_error_rate_over_epochs(overall_error_rates, feature_set, figures_dir)

    # Print summary of results
    print("\nEpoch Evaluation Summary:")
    for epoch, error_rate in enumerate(overall_error_rates):
        print(f"Epoch {epoch}: Overall Error Rate = {error_rate:.2%}")


def compare_overall_error_rates(
    models_dir: str,
    implementation: str,
    n_iter: int,
    feature_sets: List[str],
    figures_dir: str = "figures",
) -> None:
    """
    Compare the overall error rates of multiple feature sets over epochs in a single plot.
    """
    figures_path = Path(figures_dir)
    figures_path.mkdir(exist_ok=True)

    overall_error_rates_comparison = {}

    # Evaluate overall error rates for each feature set
    for feature_set in feature_sets:
        overall_error_rates = []

        for epoch in range(n_iter):
            try:
                models, vocab = load_models(models_dir, implementation, epoch)
            except ValueError as e:
                logging.warning(f"Skipping epoch {epoch} for feature set '{feature_set}': {e}")
                continue

            logging.info(f"Evaluating models from epoch {epoch} for feature set '{feature_set}'...")
            results = decode_vocabulary(models, vocab, feature_set)

            # Calculate overall error rate for the epoch
            total_correct = 0
            total_samples = 0
            for word in vocab:
                word_results = results[word]
                correct = sum(r["correct"] for r in word_results)
                total = len(word_results)

                total_correct += correct
                total_samples += total

            overall_error_rate = 1 - (total_correct / total_samples if total_samples > 0 else 0)
            overall_error_rates.append(overall_error_rate)

        overall_error_rates_comparison[feature_set] = overall_error_rates

    # Plot overall error rate comparison
    plt.figure(figsize=(10, 6))

    for feature_set, error_rates in overall_error_rates_comparison.items():
        epochs = range(len(error_rates))
        plt.plot(epochs, error_rates, marker='o', label=feature_set)

    plt.xlabel("Epoch")
    plt.ylabel("Overall Error Rate")
    plt.title("Comparison of Overall Error Rates Across Feature Sets")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_path / "overall_error_rate_comparison.png")
    plt.close()

    # Print summary of results
    print("\nOverall Error Rate Comparison Summary:")
    for feature_set, error_rates in overall_error_rates_comparison.items():
        print(f"\nFeature Set: {feature_set}")
        for epoch, error_rate in enumerate(error_rates):
            print(f"Epoch {epoch}: Overall Error Rate = {error_rate:.2%}")









if __name__ == "__main__":
     print("\nEvaluating development set at every epoch:")
     eval_hmm_every_epoch(models_dir="trained_models", implementation="hmmlearn", n_iter=16)
     print("\nEvaluating evaluation set at every epoch:")
     eval_hmm_every_epoch(models_dir="trained_models", implementation="hmmlearn", n_iter=16, feature_set="eval_feature_set")
     compare_overall_error_rates(models_dir="trained_models", implementation="hmmlearn",  n_iter=16, feature_sets=["feature_set", "eval_feature_set"], ) # Replace with your feature set names)
        
    
    # print("\nEvaluating development set:")
    # custom_dev_results = eval_hmm("custom", "feature_set", model_iter=15)
    # hmmlearn_dev_results = eval_hmm("hmmlearn", "feature_set", model_iter=15)

    # print("\nEvaluating test set:")
    # custom_test_results = eval_hmm("hmmlearn", "eval_feature_set", model_iter=15)
    # hmmlearn_test_results = eval_hmm("custom", "eval_feature_set", model_iter=15)