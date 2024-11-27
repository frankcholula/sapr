from hmm import HMM
from mfcc_extract import load_mfccs

if __name__ == "__main__":
    feature_set = load_mfccs("feature_set")
    hmm = HMM(8, 13, feature_set)
    log_emission_matrix = hmm.compute_log_emission_matrix(feature_set[0])
    hmm.print_matrix(log_emission_matrix, "Log Emission Matrix")
    hmm.print_transition_matrix()
    alpha_log = hmm.forward(log_emission_matrix, use_log=True)
    beta_log = hmm.backward(log_emission_matrix, use_log=True)
    gamma_log = hmm.compute_gamma(alpha_log, beta_log)
    hmm.print_matrix(gamma_log, "Gamma Matrix")

