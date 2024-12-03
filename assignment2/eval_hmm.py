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
from sklearn.metrics import confusion_matrix, accuracy_score


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
    feature_set_path = "feature_set"
    feature_set = load_mfccs(feature_set_path)
    features = {word: load_mfccs_by_word(feature_set_path, word) for word in vocabs}
    total_features_length = sum(len(features[word]) for word in vocabs)
    assert total_features_length == len(feature_set)

    hmms = {}
    true_labels = []
    predicted_labels = []

    for word in vocabs:
        model_path = os.path.join(models_dir, f"{word}_{implementation}.pkl")
        hmms[word] = pickle.load(open(model_path, 'rb'))

    # for word, model in hmms.items():
    #   logging.info(f"Model for '{word}': n_components={model.n_components}, covariance_type={model.covariance_type}")
    #   assert model.n_features == num_features, f"Feature size mismatch for word '{word}'"
  
  # Perform evaluation
    for true_word, mfcc_list in features.items():
        for mfcc in mfcc_list:
            max_log_prob = float("-inf")
            predicted_word = None

            for word, model in hmms.items():
                if implementation == "hmmlearn":
                    log_prob, _ = model.decode(mfcc.T, algorithm="viterbi")
                else:
                    print("Not implemented yet")

                if log_prob > max_log_prob:
                    max_log_prob = log_prob
                    predicted_word = word

            true_labels.append(true_word)
            predicted_labels.append(predicted_word)

    print(true_labels)
    print(predicted_labels)
    # Calculate confusion matrix and accuracy
    label_mapping = {word: idx for idx, word in enumerate(vocabs)}
    true_labels_idx = [label_mapping[label] for label in true_labels]
    predicted_labels_idx = [label_mapping[label] for label in predicted_labels]

    cm = confusion_matrix(true_labels_idx, predicted_labels_idx)
    accuracy = accuracy_score(true_labels_idx, predicted_labels_idx)

    # Create a DataFrame for the confusion matrix
    cm_df = pd.DataFrame(
        cm, index=vocabs, columns=vocabs
    )

    # Log results
    logging.info(f"Confusion Matrix:\n{cm_df}")
    logging.info(f"Overall Accuracy: {accuracy:.2%}")

    # Save confusion matrix to a CSV
    cm_df.to_csv(f"figures/10epochs_{implementation}_confusion_matrix_dev_set.csv")

    return {
        "hmms": hmms,
        "accuracy": accuracy,
        "confusion_matrix": cm_df,
    }



if __name__ == "__main__":
    print("\Evaluating with `hmmlearn` implementation:")
    eval_hmm("hmmlearn")
