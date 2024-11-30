from hmm import HMM
from mfcc_extract import load_mfccs_by_word
import numpy as np

if __name__ == "__main__":
    feature_set = load_mfccs_by_word("feature_set", "hood")
    hmm = HMM(8, 13, feature_set)


    emission_matrix = hmm.compute_log_emission_matrix(feature_set[0])
    alpha = hmm.forward(emission_matrix, use_log=True)
    beta = hmm.backward(emission_matrix, use_log=True)
    gamma = hmm.compute_gamma(alpha, beta)
    xi = hmm.compute_xi(alpha, beta, emission_matrix)
    hmm.baum_welch(feature_set, 10)
    
    # hmm.print_matrix(emission_matrix, "Emission Matrix")
    # hmm.print_matrix(alpha, "Alpha")
    # hmm.print_matrix(gamma, "Gamma")
    # hmm.print_matrix(xi[0, :, :], "Xi")

    