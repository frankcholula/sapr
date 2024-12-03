import logging
import pandas as pd
from pathlib import Path
from custom_hmm import HMM
from hmmlearn_hmm import HMMLearnModel
from typing import Literal, Dict, Union
from sklearn.metrics import confusion_matrix, accuracy_score
from decoder import Decoder

logging.basicConfig(level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def eval_hmm(
    implementation: Literal["custom", "hmmlearn"] = "hmmlearn",
    feature_set_path: str = "eval_feature_set"
) -> Dict[str, Union[dict, float, pd.DataFrame]]:
    
    decoder = Decoder(implementation=implementation)
    all_results = decoder.decode_vocabulary(feature_set_path, verbose=False)
    
    true_labels = []
    predicted_labels = []
    
    for word, results in all_results.items():
        for result in results:
            true_labels.append(result["true_word"])
            predicted_labels.append(result["predicted_word"])
    
    label_mapping = {word: idx for idx, word in enumerate(decoder.vocab)}
    true_labels_idx = [label_mapping[label] for label in true_labels]
    predicted_labels_idx = [label_mapping[label] for label in predicted_labels]

    cm = confusion_matrix(true_labels_idx, predicted_labels_idx)
    accuracy = accuracy_score(true_labels_idx, predicted_labels_idx)

    cm_df = pd.DataFrame(cm, index=decoder.vocab, columns=decoder.vocab)
    logging.info(f"\nConfusion Matrix:\n{cm_df}")
    logging.info(f"\nOverall Accuracy: {accuracy:.2%}")

    logging.info("\nPer-word accuracy:")
    for word, results in all_results.items():
        word_correct = sum(r["correct"] for r in results)
        word_total = len(results)
        word_accuracy = word_correct / word_total
        logging.info(f"{word}: {word_accuracy:.2%}")

    models_dir = Path("trained_models")
    output_path = models_dir / f"{implementation}_confusion_matrix_{Path(feature_set_path).stem}.csv"
    cm_df.to_csv(output_path)
    logging.info(f"\nSaved confusion matrix to: {output_path}")

    return {
        "results": all_results,
        "accuracy": accuracy,
        "confusion_matrix": cm_df,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
    }


if __name__ == "__main__":
    print("\nEvaluating development set:")
    dev_results = eval_hmm("hmmlearn", "feature_set")
    
    print("\nEvaluating test set:")
    test_results = eval_hmm("hmmlearn", "eval_feature_set")