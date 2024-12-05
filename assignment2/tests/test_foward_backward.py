# import pytest
# import numpy as np
# from custom_hmm import HMM
# from mfcc_extract import load_mfccs


# @pytest.fixture
# def feature_set():
#     return load_mfccs("feature_set")


# @pytest.fixture
# def hmm_model(feature_set):
#     return HMM(8, 13, feature_set)


# def test_emission_matrix(hmm_model, feature_set):
#     test_features = feature_set[0]
#     B_probs = hmm_model.compute_emission_matrix(test_features)
#     assert B_probs.shape == (test_features.shape[1], hmm_model.total_states)
#     assert np.all(B_probs <= 0), "Log probabilities should be non-positive"
#     assert np.all(
#         np.isfinite(B_probs[B_probs != -np.inf])
#     ), "Log probabilities should be finite where not -inf"

#     assert np.all(
#         B_probs[:, 0] == -np.inf
#     ), "Entry state should have -inf log probabilities"
#     assert np.all(
#         B_probs[:, -1] == -np.inf
#     ), "Exit state should have -inf log probabilities"

#     real_states_probs = B_probs[:, 1:-1]  # Exclude entry and exit states
#     first_frame_probs = real_states_probs[0, :]
#     print(f"\nFirst Frame Log Probabilities:\n{first_frame_probs}")
#     prob_std = np.std(first_frame_probs)
#     assert prob_std < 1e-10, "Initial log probabilities should be similar across states"
#     hmm_model.print_matrix(
#         B_probs, "Emission Matrix", col="State", idx="T", start_idx=1, start_col=1
#     )


# def test_fb_probabilities_basic(hmm_model, feature_set):
#     test_features = feature_set[0]
#     emission_matrix = hmm_model.compute_emission_matrix(test_features)
#     alpha = hmm_model.forward(emission_matrix)
#     beta = hmm_model.backward(emission_matrix)

#     T = emission_matrix.shape[0]

#     # Test shapes
#     assert alpha.shape == (
#         T,
#         hmm_model.total_states,
#     ), "Alpha should have shape (T, total_states)"
#     assert beta.shape == (
#         T,
#         hmm_model.total_states,
#     ), "Beta should have shape (T, total_states)"

#     # Test entry/exit state properties
#     assert np.all(alpha[1:, 0] == -np.inf), "Entry state should be -inf after t=0"
#     assert np.all(beta[:-1, -1] == -np.inf), "Exit state should be -inf except at final time"
#     assert beta[-1, -1] == 0, "Exit state at final time should be 0"

#     # Test log probability properties
#     assert np.all(
#         alpha[alpha != -np.inf] <= 0
#     ), "Finite log alpha values should be non-positive"
#     assert np.all(
#         beta[beta != -np.inf] <= 0
#     ), "Finite log beta values should be non-positive"
#     assert np.all(
#         np.isfinite(alpha[alpha != -np.inf])
#     ), "Non-inf alpha values should be finite"
#     assert np.all(
#         np.isfinite(beta[beta != -np.inf])
#     ), "Non-inf beta values should be finite"

#     # Print sample values for debugging
#     print("\nForward Probabilities (Alpha):")
#     hmm_model.print_matrix(alpha, "Alpha Matrix", col="State", idx="T")

#     print("\nBackward Probabilities (Beta):")
#     hmm_model.print_matrix(beta, "Beta Matrix", col="State", idx="T")

#     print("\nForward-Backward sums at each timestep:")
#     for t in range(T):
#         fb_sum = np.logaddexp.reduce(alpha[t] + beta[t])
#         print(f"t={t}: {fb_sum}")

#     # Print some sample values for inspection
#     print(f"\nSample log probabilities at t=0:")
#     print(f"Alpha[0,0] (entry state): {alpha[0,0]}")
#     print(f"Alpha[0,1] (first real state): {alpha[0,1]}")
#     print(f"Beta[0,1] (first real state): {beta[0,1]}")


# def test_fb_probabilities_advanced(hmm_model, feature_set):
#     test_features = feature_set[0]
#     emission_matrix = hmm_model.compute_emission_matrix(test_features)
#     alpha = hmm_model.forward(emission_matrix)
#     beta = hmm_model.backward(emission_matrix)

#     T = emission_matrix.shape[0]
#     middle_t = T // 2

#     # Compute posterior probabilities from alpha and beta
#     posterior = alpha[middle_t] + beta[middle_t]
#     posterior = posterior - np.logaddexp.reduce(posterior)

#     # Compute state probabilities from gamma, handling zeros
#     gamma = hmm_model.compute_gamma(alpha, beta)
#     gamma_middle = gamma[middle_t]

#     # Convert non-zero gamma values to log space
#     # Only look at real states (indices 1 to -1) to avoid entry/exit states
#     real_states_posterior = posterior[1:-1]
#     real_states_gamma = gamma_middle[1:-1]

#     print("\nAt timestep", middle_t)
#     print("Posterior probabilities:", np.exp(real_states_posterior))
#     print("Gamma probabilities:", real_states_gamma)

#     # Compare only where gamma is non-zero
#     non_zero_mask = real_states_gamma > 0
#     gamma_log = np.log(real_states_gamma[non_zero_mask])
#     posterior_masked = real_states_posterior[non_zero_mask]

#     assert np.allclose(
#         posterior_masked, gamma_log, atol=1e-5
#     ), "Posterior probabilities should match gamma values at timestep t"
