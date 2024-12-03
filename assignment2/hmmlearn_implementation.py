import numpy as np
from hmmlearn import hmm
from typing import List
from mfcc_extract import load_mfccs, load_mfccs_by_word
import logging


logging.basicConfig(level=logging.INFO)


class HMMLearnModel:
    def __init__(
        self,
        num_states: int = 8,
        model_name: str = None,
        n_iter: int = 15,
    ):
        self.model_name = model_name
        self.global_mean = self.calc_gloabal_mean(load_mfccs("feature_set"))
        self.global_cov = self.clac_global_cov(load_mfccs("feature_set"))
        print(self.global_mean, self.global_cov)
        self.model = hmm.GaussianHMM(
            n_components=num_states,
            covariance_type="diag",
            n_iter=n_iter,
            init_params="mc",
            params="stmc",
            implementation="log"
        )

    def prepare_data(self, feature_set: List[np.ndarray]) -> tuple:
        return np.concatenate([f.T for f in feature_set], axis=0)
    
    def calc_gloabal_mean(self, feature_set: List[np.ndarray]) -> np.ndarray:
        X = self.prepare_data(feature_set)
        return np.mean(X, axis=0)
    
    def clac_global_cov(self, feature_set: List[np.ndarray]) -> np.ndarray:
        X = self.prepare_data(feature_set)
        return np.cov(X, rowvar=False)

    def fit(self, feature_set: List[np.ndarray]) -> None:
        logging.info(
            f"Training {self.model_name} HMM using hmmlearn in {self.model.n_iter} iterations..."
        )
        X = self.prepare_data(feature_set)
        try:
            self.model.fit(X)
            lengths = [f.shape[1] for f in feature_set]
            log_likelihood = self.model.score(X, lengths)
            logging.info(f"Training completed with log likelihood: {log_likelihood}")
            return self.model, log_likelihood
        except Exception as e:
            logging.error(f"Error occured while training {self.model_name} HMM: {e}")

    # def fit(self, feature_set: List[np.ndarray], n_iter: int = 15) -> None:
    #     logging.info(f"Training {self.model_name} HMM using hmmlearn...")

    #     X, lengths = self.prepare_data(feature_set)
    #     self.model.n_iter = n_iter
    #     self.model.fit(X, lengths=lengths)

    #     logging.info("Training completed")

    # def predict(self, features: np.ndarray) -> np.ndarray:
    #     X = features.T
    #     return self.model.predict(X)


if __name__ == "__main__":
    heed_features = load_mfccs_by_word("feature_set", "heed")
    myhmm = HMMLearnModel(8, model_name="heed")
    myhmm.fit(heed_features)
    print(myhmm.model.transmat_)
