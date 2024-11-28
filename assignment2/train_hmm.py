from hmm import HMM
from mfcc_extract import load_mfccs
import numpy as np

if __name__ == "__main__":
    feature_set = load_mfccs("feature_set")
    hmm = HMM(8, 13, feature_set)
    emission_matrix = hmm.compute_log_emission_matrix(feature_set[0])
    hmm.print_matrix(emission_matrix, "Log Emission Matrix")
    hmm.print_transition_matrix()
    alpha = hmm.forward(emission_matrix, use_log=True)
    beta = hmm.backward(emission_matrix, use_log=True)
    gamma = hmm.compute_gamma(alpha, beta)
    hmm.print_matrix(gamma, "Gamma Matrix")
    xi = hmm.compute_xi(alpha, beta, emission_matrix)


    # First, sum xi over the "to" states
    xi_summed = np.sum(xi, axis=2)  # Shape: (T-1, num_states)
    # Transpose xi_summed to match gamma's shape
    xi_summed = xi_summed.T  # Now shape: (num_states, T-1)
    hmm.print_matrix(gamma, "Gamma Matrix")
    hmm.print_matrix(xi_summed, "Summed Xi Matrix")