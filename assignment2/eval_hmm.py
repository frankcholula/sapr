import logging
import pickle
import pandas as pd
import os
from pathlib import Path
from custom_hmm import HMM
from hmmlearn_hmm import HMMLearnModel
from typing import Literal, Dict, Union
from mfcc_extract import load_mfccs, load_mfccs_by_word
from sklearn.metrics import confusion_matrix, accuracy_score
from decode import decode_sequence

logging.basicConfig(level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def eval_hmm(
    implementation: Literal["custom", "hmmlearn"] = "hmmlearn",
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
    feature_set_path = "eval_feature_set"
    feature_set = load_mfccs(feature_set_path)
    features = {word: load_mfccs_by_word(feature_set_path, word) for word in vocabs}
    total_features_length = sum(len(features[word]) for word in vocabs)
    assert total_features_length == len(feature_set)

    # Load trained models
    hmms = {}
    for word in vocabs:
        model_path = os.path.join(models_dir, f"{word}_{implementation}.pkl")
        with open(model_path, "rb") as f:
            hmms[word] = pickle.load(f)

    true_labels = []
    predicted_labels = []

    # Perform evaluation using decode_sequence
    for true_word, mfcc_list in features.items():
        for mfcc in mfcc_list:
            # Use decode_sequence from decode.py
            predicted_word, log_prob, state_sequence = decode_sequence(
                hmms, mfcc.T
            )
            
            true_labels.append(true_word)
            predicted_labels.append(predicted_word)

            # Optional: Log individual predictions
            logging.debug(f"True: {true_word}, Predicted: {predicted_word}, Log prob: {log_prob:.2f}")

    # Calculate confusion matrix and accuracy
    label_mapping = {word: idx for idx, word in enumerate(vocabs)}
    true_labels_idx = [label_mapping[label] for label in true_labels]
    predicted_labels_idx = [label_mapping[label] for label in predicted_labels]

    cm = confusion_matrix(true_labels_idx, predicted_labels_idx)
    accuracy = accuracy_score(true_labels_idx, predicted_labels_idx)

    # Create a DataFrame for the confusion matrix
    cm_df = pd.DataFrame(cm, index=vocabs, columns=vocabs)

    # Log results
    logging.info(f"\nConfusion Matrix:\n{cm_df}")
    logging.info(f"\nOverall Accuracy: {accuracy:.2%}")

    # Calculate per-word accuracy
    logging.info("\nPer-word accuracy:")
    for word in vocabs:
        word_mask = [t == word for t in true_labels]
        word_true = [t for t, m in zip(true_labels, word_mask) if m]
        word_pred = [p for p, m in zip(predicted_labels, word_mask) if m]
        word_accuracy = accuracy_score(word_true, word_pred)
        logging.info(f"{word}: {word_accuracy:.2%}")

    # Save confusion matrix to a CSV
    output_path = models_dir / f"{implementation}_confusion_matrix_eval_set.csv"
    cm_df.to_csv(output_path)
    logging.info(f"\nSaved confusion matrix to: {output_path}")

    return {
        "hmms": hmms,
        "accuracy": accuracy,
        "confusion_matrix": cm_df,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels
    }


if __name__ == "__main__":
    print("\nEvaluating with `hmmlearn` implementation:")
    results = eval_hmm("hmmlearn")
