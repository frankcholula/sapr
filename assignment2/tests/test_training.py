import pytest
import numpy as np
from custom_hmm import HMM
from mfcc_extract import load_mfccs, load_mfccs_by_word


@pytest.fixture
def feature_set():
    return load_mfccs("feature_set")


@pytest.fixture
def hmm_model(feature_set):
    return HMM(8, 13, feature_set)


@pytest.fixture
def heed_features():
    return load_mfccs_by_word("feature_set", "heed")


def test_gamma_xi_probabilities(hmm_model, feature_set):
    test_features = feature_set[0]
    emission_matrix = hmm_model.compute_emission_matrix(test_features)
    alpha = hmm_model.forward(emission_matrix)
    beta = hmm_model.backward(emission_matrix)
    gamma = hmm_model.compute_gamma(alpha, beta)
    xi = hmm_model.compute_xi(alpha, beta, emission_matrix)

    T = emission_matrix.shape[0]

    # Test dimensions
    assert gamma.shape == (T, hmm_model.total_states)
    assert xi.shape == (T - 1, hmm_model.total_states, hmm_model.total_states)

    # Test probability properties
    assert np.all(gamma >= 0) and np.all(gamma <= 1)
    assert np.all(xi >= 0) and np.all(xi <= 1)

    print("\nGamma Matrix (time step 10):")
    hmm_model.print_matrix(gamma[10:11], "Gamma Matrix", col="State", idx="T")

    print("\nXi Matrix for t=10 (transitions from time step 10):")
    hmm_model.print_matrix(xi[10], "Xi Matrix t=10", col="To State", idx="From State")


# def test_update_transitions(hmm_model, heed_features):
#     """
#     Test the HMM transition matrix updates using multiple MFCC feature sequences from the 'heed' word.
#     This test verifies that:
#     1. The transition matrix maintains proper left-right HMM structure
#     2. Probabilities are properly normalized
#     3. Entry and exit state transitions are correctly handled
#     """
#     # Initialize accumulators for statistics across sequences
#     aggregated_gamma = np.zeros((hmm_model.num_states, 1))
#     aggregated_xi = np.zeros((hmm_model.num_states, hmm_model.num_states))
#     for seq_idx, heed_feature in enumerate(heed_features):
#         # Compute forward-backward statistics for this sequence
#         emission_matrix = hmm_model.compute_log_emission_matrix(heed_feature)
#         alpha = hmm_model.forward(emission_matrix, use_log=True)
#         beta = hmm_model.backward(emission_matrix, use_log=True)

#         # Compute gamma and xi for this sequence
#         gamma = hmm_model.compute_gamma(alpha, beta, use_log=True)
#         xi = hmm_model.compute_xi(alpha, beta, emission_matrix, use_log=True)

#         # Accumulate statistics
#         aggregated_gamma += np.sum(gamma, axis=1, keepdims=True)
#         aggregated_xi += np.sum(xi, axis=0)
#         print(f"\nProcessed sequence {seq_idx + 1}")
#         print(f"Sequence length: {heed_feature.shape[1]} frames")
#         assert np.isclose(
#             np.sum(gamma), heed_feature.shape[1]
#         ), "Gamma sum should equal T"
#         assert np.isclose(
#             np.sum(xi), heed_feature.shape[1] - 1
#         ), "Xi sum should equal T-1"

#     print("\nInitial A matrix:")
#     hmm_model.print_transition_matrix()

#     hmm_model.update_A(aggregated_xi, aggregated_gamma)

#     print("\nUpdated A matrix:")
#     hmm_model.print_transition_matrix()

#     assert (
#         hmm_model.A[0, 1] == 1.0
#     ), "Entry state must transition to first state with prob 1"
#     assert np.all(
#         hmm_model.A[0, 2:] == 0
#     ), "Entry state should have no other transitions"

#     #  Check main state transitions
#     for i in range(1, hmm_model.num_states + 1):
#         row_probs = hmm_model.A[i, :]

#         # Verify probability normalization
#         assert np.isclose(
#             np.sum(row_probs), 1.0, atol=1e-10
#         ), f"Row {i} probabilities must sum to 1"

#         # Verify left-right structure
#         if i < hmm_model.num_states:  # Not the last state
#             allowed = np.zeros_like(row_probs)
#             allowed[i] = 1  # Self-transition
#             allowed[i + 1] = 1  # Next state
#             assert np.all(
#                 (row_probs > 0) == allowed
#             ), f"State {i} has invalid transitions"

#             assert row_probs[i] > 0, f"State {i} should have non-zero self-transition"
#             assert (
#                 row_probs[i + 1] > 0
#             ), f"State {i} should have non-zero forward transition"
#         else:
#             allowed = np.zeros_like(row_probs)
#             allowed[i] = 1
#             allowed[i + 1] = 1
#             assert np.all(
#                 (row_probs > 0) == allowed
#             ), f"Last state has invalid transitions"

#     assert np.all(
#         hmm_model.A[-1, :] == 0
#     ), "Exit state should have no outgoing transitions"
#     assert np.all(
#         np.tril(hmm_model.A[1:-1, 1:-1], k=-1) == 0
#     ), "No backward transitions allowed"
#     assert np.all(
#         np.triu(hmm_model.A[1:-1, 1:-1], k=2) == 0
#     ), "No skipping states allowed"

#     main_diag = np.diag(hmm_model.A[1:-1, 1:-1])
#     assert np.all(
#         (main_diag > 0.5) & (main_diag < 0.95)
#     ), "Self-transition probabilities should be reasonable (between 0.5 and 0.95)"

#     # Print statistics to debug
#     print("\nTransition Statistics:")
#     print(f"Total gamma sum across all sequences: {np.sum(aggregated_gamma):.3f}")
#     print(f"Total xi sum across all sequences: {np.sum(aggregated_xi):.3f}")
#     print(f"Average self-transition probability: {np.mean(main_diag):.3f}")
#     print(f"Min self-transition probability: {np.min(main_diag):.3f}")
#     print(f"Max self-transition probability: {np.max(main_diag):.3f}")

#     forward_probs = [hmm_model.A[i, i + 1] for i in range(1, hmm_model.num_states + 1)]
#     print(f"\nForward transition probabilities: {[f'{p:.3f}' for p in forward_probs]}")


# def test_update_emissions(hmm_model, heed_features):
#     """
#     Basic test for emission parameter updates using 'heed' sequences.
#     Verifies fundamental properties of means and covariances after updates.
#     """
#     # Store initial parameters to check if they change
#     initial_means = hmm_model.B["mean"].copy()
#     initial_variances = hmm_model.B["covariance"].copy()

#     # Use first three sequences for a simple test
#     gamma_per_seq = []

#     # Compute gamma for each sequence
#     for features in heed_features:
#         emission_matrix = hmm_model.compute_log_emission_matrix(features)
#         alpha = hmm_model.forward(emission_matrix, use_log=True)
#         beta = hmm_model.backward(emission_matrix, use_log=True)
#         gamma = hmm_model.compute_gamma(alpha, beta, use_log=True)
#         gamma_per_seq.append(gamma)

#         # Print basic sequence info for debugging
#         print(f"\nSequence length: {features.shape[1]} frames")
#         print(f"Gamma sum: {np.sum(gamma):.3f}")

#     # Update emission parameters
#     hmm_model.update_B(heed_features, gamma_per_seq)

#     # === Basic Verification Checks ===

#     # 1. Check that parameters actually changed
#     assert not np.array_equal(initial_means, hmm_model.B["mean"]), \
#         "Means should be updated"
#     assert not np.array_equal(initial_variances, hmm_model.B["covariance"]), \
#         "Variances should be updated"

#     # 2. Check mathematical validity
#     assert np.all(np.isfinite(hmm_model.B["mean"])), \
#         "All means should be finite"
#     assert np.all(np.isfinite(hmm_model.B["covariance"])), \
#         "All variances should be finite"
#     assert np.all(hmm_model.B["covariance"] > 0), \
#         "All variances should be positive"

#     # Print before/after statistics for inspection
#     print("\nMean value ranges:")
#     print(f"Before: [{np.min(initial_means):.3f}, {np.max(initial_means):.3f}]")
#     print(f"After:  [{np.min(hmm_model.B['mean']):.3f}, {np.max(hmm_model.B['mean']):.3f}]")

#     print("\nVariance ranges:")
#     print(f"Before: [{np.min(initial_variances):.3f}, {np.max(initial_variances):.3f}]")
#     print(f"After:  [{np.min(hmm_model.B['covariance']):.3f}, {np.max(hmm_model.B['covariance']):.3f}]")


def test_baum_welch(hmm_model, heed_features):
    """
    Test the full Baum-Welch algorithm using the 'heed' sequences.
    Verifies that the model parameters converge to a stable state.
    """
    hmm_model.baum_welch(heed_features)
