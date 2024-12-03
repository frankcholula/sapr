import numpy as np
from hmmlearn import hmm
from typing import List
from mfcc_extract import load_mfccs, load_mfccs_by_word
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

class HMMLearnModel:
    def __init__(
        self,
        num_states: int = 8,
        model_name: str = None,
        n_iter: int = 15,
    ):
        self.model_name = model_name
        self.num_states = num_states
        self.total_states = num_states + 2
        
        self.all_features = load_mfccs("feature_set")
        self.global_mean = self.calc_global_mean(self.all_features)
        self.global_cov = self.calc_global_cov(self.all_features)
        
        self.model = hmm.GaussianHMM(
            n_components=self.total_states,
            covariance_type="diag",
            n_iter=n_iter,
            params="stmc",
            implementation="log",
            min_covar=0.01,
            init_params=""
        )
        
        self.model.means_ = np.tile(self.global_mean, (self.total_states, 1))
        self.model.covars_ = np.tile(self.global_cov, (self.total_states, 1))
        
        self.model.transmat_ = self.initialize_transmat()
        self.model.startprob_ = np.zeros(self.total_states)
        self.model.startprob_[0] = 1.0

    def initialize_transmat(self) -> np.ndarray:
        total_frames = sum(f.shape[1] for f in self.all_features)
        num_sequences = len(self.all_features)
        avg_frames = total_frames / num_sequences
        avg_frames_per_state = avg_frames / self.num_states
        
        aii = np.exp(-1 / (avg_frames_per_state - 1))
        aij = 1 - aii
        
        print(f"\nTransition probability initialization:")
        print(f"Total frames: {total_frames}")
        print(f"Number of sequences: {num_sequences}")
        print(f"Average frames per sequence: {avg_frames:.2f}")
        print(f"Average frames per state: {avg_frames_per_state:.2f}")
        print(f"Self-transition probability (aii): {aii:.3f}")
        print(f"Next-state transition probability (aij): {aij:.3f}")
        
        transmat = np.zeros((self.total_states, self.total_states))
        
        transmat[0, 1] = 1.0
        
        # Real states
        for i in range(1, self.num_states + 1):
            if i < self.num_states:
                transmat[i, i] = aii
                transmat[i, i + 1] = aij
            else:
                # Last real state
                transmat[i, i] = aii
                transmat[i, i + 1] = aij
        
        transmat[self.num_states + 1, self.num_states + 1] = 1.0
        
        return transmat

    def prepare_data(self, feature_set: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([f.T for f in feature_set], axis=0)

    def calc_global_mean(self, feature_set: List[np.ndarray]) -> np.ndarray:
        X = self.prepare_data(feature_set)
        global_mean = np.mean(X, axis=0)
        print(f"Global mean shape: {global_mean.shape}")
        return global_mean

    def calc_global_cov(self, feature_set: List[np.ndarray]) -> np.ndarray:
        X = self.prepare_data(feature_set)
        global_cov = np.var(X, axis=0)
        print(f"Global variance shape: {global_cov.shape}")
        print(f"Variance range: [{global_cov.min():.6f}, {global_cov.max():.6f}]")
        return global_cov

    def fit(self, feature_set: List[np.ndarray]) -> None:
        logging.info(
            f"Training {self.model_name} HMM using hmmlearn in {self.model.n_iter} iterations..."
        )
        X = self.prepare_data(feature_set)
        lengths = [f.shape[1] for f in feature_set]
        try:
            self.model.fit(X, lengths)
            log_likelihood = self.model.score(X, lengths)
            logging.info(f"Training completed with log likelihood: {log_likelihood}")
            return self.model, log_likelihood
        except Exception as e:
            logging.error(f"Error occurred while training {self.model_name} HMM: {e}")


def pretty_print_matrix(matrix: np.ndarray, precision: int = 3) -> None:
    n = matrix.shape[0]
    df = pd.DataFrame(
        matrix, 
        columns=[f"S{i}" if i != 0 and i != n-1 else ("Entry" if i == 0 else "Exit") 
                for i in range(n)],
        index=[f"S{i}" if i != 0 and i != n-1 else ("Entry" if i == 0 else "Exit") 
               for i in range(n)]
    )

    row_sums = df.sum(axis=1).round(precision)
    df = df.replace(0, ".")

    print("\nTransition Matrix:")
    print("==================")
    print(df.round(precision))
    assert np.allclose(row_sums, 1.0), "Row sums should be equal to 1.0"


if __name__ == "__main__":
    myhmm = HMMLearnModel(8, model_name="heed")
    print("\nInitial transition matrix:")
    pretty_print_matrix(myhmm.model.transmat_)
    myhmm.fit(load_mfccs_by_word("feature_set", "heed"))
    print("\nFinal transition matrix:")
    pretty_print_matrix(myhmm.model.transmat_)
