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


def plot_accuracy_per_class(epoch: int, word_accuracies: Dict[str, float], figures_dir: str = "figures") -> None:
    """
    Plot and save accuracy per class (word) for a given epoch.
    """
    figures_path = Path(figures_dir)
    figures_path.mkdir(exist_ok=True)

    words = list(word_accuracies.keys())
    accuracies = list(word_accuracies.values())

    plt.figure(figsize=(10, 6))
    plt.bar(words, accuracies)
    plt.xlabel("Words")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy per Class for Epoch {epoch}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures_path / f"epoch_{epoch}_accuracy.png")
    plt.close()


def eval_hmm_every_epoch(models_dir: str, implementation: str, n_iter: int, feature_set: str = "feature_set") -> None:
    """
    Evaluate models for all epochs and log accuracy.
    """
    epoch_results = []

    for epoch in range(n_iter):
        try:
            models, vocab = load_models(models_dir, implementation, epoch)
        except ValueError as e:
            logging.warning(f"Skipping epoch {epoch}: {e}")
            continue

        logging.info(f"Evaluating models from epoch {epoch}...")
        results = decode_vocabulary(models, vocab, feature_set)

        # Calculate and log accuracy
        all_predictions = [result["correct"] for word_results in results.values() for result in word_results]
        accuracy = sum(all_predictions) / len(all_predictions) if all_predictions else 0
        epoch_results.append((epoch, accuracy))
        logging.info(f"Epoch {epoch} Accuracy: {accuracy:.2%}")

    # Print summary of results
    print("\nEpoch Evaluation Summary:")
    for epoch, acc in epoch_results:
        print(f"Epoch {epoch}: Accuracy = {acc:.2%}")



if __name__ == "__main__":
     print("\nEvaluating development set at every epoch:")
     eval_hmm_every_epoch(models_dir="trained_models", implementation="hmmlearn", n_iter=15)
     print("\nEvaluating evaluation set at every epoch:")
     eval_hmm_every_epoch(models_dir="trained_models", implementation="hmmlearn", n_iter=15, feature_set="eval_feature_set")
    
    
    
    
    # print("\nEvaluating development set:")
    # custom_dev_results = eval_hmm("custom", "feature_set", model_iter=15)
    # hmmlearn_dev_results = eval_hmm("hmmlearn", "feature_set", model_iter=15)

    # print("\nEvaluating test set:")
    # custom_test_results = eval_hmm("hmmlearn", "eval_feature_set", model_iter=15)
    # hmmlearn_test_results = eval_hmm("custom", "eval_feature_set", model_iter=15)