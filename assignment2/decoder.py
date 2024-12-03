import pickle
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from mfcc_extract import load_mfccs_by_word

logging.basicConfig(level=logging.INFO)


class Decoder:
    def __init__(self, models_dir: str = "trained_models", implementation: str = "hmmlearn"):
        self.models_dir = Path(models_dir)
        self.implementation = implementation
        self.models: Dict = {}
        self.vocab: List[str] = []
        self.load_models()

    def load_models(self) -> None:
        pattern = f"*_{self.implementation}.pkl"
        
        for model_path in self.models_dir.glob(pattern):
            word = model_path.stem.split("_")[0]
            with open(model_path, "rb") as f:
                self.models[word] = pickle.load(f)
                self.vocab.append(word)
        
        if not self.models:
            raise ValueError(f"No models found in {self.models_dir} with pattern {pattern}")
        
        logging.info(f"Loaded {len(self.models)} models for words: {', '.join(self.vocab)}")

    def decode_sequence(self, features: np.ndarray) -> Tuple[str, float, List[int]]:
        best_score = float("-inf")
        best_word = None
        best_states = None

        for word, model in self.models.items():
            log_prob, states = model.decode(features)
            if log_prob > best_score:
                best_score = log_prob
                best_word = word
                best_states = states

        return best_word, best_score, best_states

    def decode_word_samples(self, word: str, feature_set: str = "feature_set") -> List[Dict]:
        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not in vocabulary: {self.vocab}")
            
        results = []
        features = load_mfccs_by_word(feature_set, word)
        
        for i, feat_seq in enumerate(features):
            test_features = feat_seq.T
            predicted_word, log_prob, state_sequence = self.decode_sequence(test_features)
            
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

    def decode_vocabulary(self, feature_set: str = "feature_set", verbose: bool = True) -> Dict[str, List[Dict]]:
        all_results = {}
        
        for word in self.vocab:
            results = self.decode_word_samples(word, feature_set)
            all_results[word] = results
            
            if verbose:
                correct = sum(r["correct"] for r in results)
                total = len(results)
                print(f"\nResults for '{word}':")
                print(f"Accuracy: {correct}/{total} ({correct/total:.1%})")
                
                for r in results:
                    print(f"\nSample {r['sample_index']}:")
                    print(f"Predicted: {r['predicted_word']}")
                    print(f"Log likelihood: {r['log_likelihood']:.2f}")
                    print(f"Correct: {'✓' if r['correct'] else '✗'}")
        
        return all_results


if __name__ == "__main__":
    decoder = Decoder()
    results = decoder.decode_vocabulary()
    
    all_predictions = [result["correct"] for word_results in results.values() for result in word_results]
    accuracy = sum(all_predictions) / len(all_predictions)
    print(f"\nOverall accuracy: {accuracy:.1%}")