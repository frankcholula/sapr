import pickle
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple
import logging
from mfcc_extract import load_mfccs_by_word

logging.basicConfig(level=logging.INFO)


def load_models(models_dir: str = "trained_models") -> Dict:
    models = {}
    models_path = Path(models_dir)

    for model_path in models_path.glob("*_hmmlearn.pkl"):
        word = model_path.stem.split("_")[0]  # Get word from filename
        with open(model_path, "rb") as f:
            models[word] = pickle.load(f)

    return models


def decode_sequence(models: Dict, features: np.ndarray) -> Tuple[str, float, List[int]]:
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


if __name__ == "__main__":
    models = load_models()
    logging.info(f"Loaded {len(models)} models")

    # Test with some sequences from feature set
    test_words = ["heed", "hid", "head", "had", "hard", "hud", "hod", "hoard", "hood", "whod", "heard"]
    print("\nDecoding test sequences:")
    for true_word in test_words:
        features = load_mfccs_by_word("feature_set", true_word)

        for i, feat_seq in enumerate(features):
            test_features = feat_seq.T

            predicted_word, log_prob, state_sequence = decode_sequence(
                models, test_features
            )

            print(f"\nTest sequence {i+1} for '{true_word}':")
            print(f"Predicted: {predicted_word}")
            print(f"Log likelihood: {log_prob:.2f}")
            print(f"Correct: {'✓' if predicted_word == true_word else '✗'}")
