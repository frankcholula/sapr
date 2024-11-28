from hmm import HMM
from mfcc_extract import load_mfccs
import numpy as np

if __name__ == "__main__":
    feature_set = load_mfccs("feature_set")
    hmm = HMM(8, 13, feature_set)
    emission_matrix = hmm.compute_log_emission_matrix(feature_set[0])
    alpha = hmm.forward(emission_matrix, use_log=True)
    beta = hmm.backward(emission_matrix, use_log=True)
    gamma = hmm.compute_gamma(alpha, beta)
    xi = hmm.compute_xi(alpha, beta, emission_matrix)
    