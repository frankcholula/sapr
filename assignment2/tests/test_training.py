import pytest
import numpy as np
from hmm import HMM
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


def test_emission_matrix(hmm_model, feature_set):
    test_features = feature_set[0]
    B_probs = hmm_model.compute_log_emission_matrix(test_features)

    # Test shape
    assert B_probs.shape == (8, test_features.shape[1])

    # Test basic properties
    assert np.all(B_probs <= 0), "Log probabilities should be non-positive"
    assert np.all(
        np.isfinite(B_probs[B_probs != -np.inf])
    ), "Log probabilities should be finite where not -inf"

    # Since we initialized with global means, first frame probabilities should be similar
    first_frame_probs = B_probs[:, 0]
    print(f"\nFirst Frame Log Probabilities:\n{first_frame_probs}")
    prob_std = np.std(first_frame_probs)
    assert prob_std < 1e-10, "Initial log probabilities should be similar across states"
    hmm_model.print_matrix(
        B_probs, "Emission Matrix", col="T", idx="State", start_idx=1, start_col=1
    )


def test_fb_probabilities(hmm_model, feature_set):
    emission_matrix = hmm_model.compute_log_emission_matrix(feature_set[0])
    alpha = hmm_model.forward(emission_matrix, use_log=True)
    beta = hmm_model.backward(emission_matrix, use_log=True)
    T = emission_matrix.shape[1]
    # Test 1: First observation emission * first backward probability
    test1 = emission_matrix[0, 0] + beta[0, 0]

    # Test 2: Last transition to exit * last forward probability
    test2 = np.log(hmm_model.A[-2, -1]) + alpha[-2, T - 1]

    print(f"\nTest 1: {test1}")
    print(f"\nTest 2: {test2}")
    print(f"\nDifference: {abs(test1 - test2)}")
    hmm_model.print_matrix(
        alpha, "Alpha Matrix", col="T", idx="State", start_idx=0, start_col=1
    )
    hmm_model.print_matrix(
        beta, "Beta Matrix", col="T", idx="State", start_idx=0, start_col=1
    )


def test_gamma_xi_probabilities(hmm_model, feature_set):
    emission_matrix = hmm_model.compute_log_emission_matrix(feature_set[0])
    alpha = hmm_model.forward(emission_matrix, use_log=True)
    beta = hmm_model.backward(emission_matrix, use_log=True)
    gamma = hmm_model.compute_gamma(alpha, beta)
    xi = hmm_model.compute_xi(alpha, beta, emission_matrix)
    assert xi.shape == (
        emission_matrix.shape[1] - 1,
        hmm_model.num_states,
        hmm_model.num_states,
    )
    assert gamma.shape == (hmm_model.num_states, feature_set[0].shape[1])
    xi_summed = np.sum(xi, axis=2).T
    hmm_model.print_matrix(
        gamma, "Gamma Matrix", col="T", idx="State", start_idx=1, start_col=1
    )
    hmm_model.print_matrix(
        xi_summed, "Summed Xi Matrix", col="T", idx="State", start_idx=1, start_col=1
    )
    np.testing.assert_array_almost_equal(gamma[:, :-1], xi_summed)


def test_update_transitions(hmm_model, heed_features):
    """
    Test the HMM transition matrix updates using multiple MFCC feature sequences from the 'heed' word.
    This test verifies that:
    1. The transition matrix maintains proper left-right HMM structure
    2. Probabilities are properly normalized
    3. Entry and exit state transitions are correctly handled
    """
    # Initialize accumulators for statistics across sequences
    aggregated_gamma = np.zeros((hmm_model.num_states, 1))
    aggregated_xi = np.zeros((hmm_model.num_states, hmm_model.num_states))
    for seq_idx, heed_feature in enumerate(heed_features):
        # Compute forward-backward statistics for this sequence
        emission_matrix = hmm_model.compute_log_emission_matrix(heed_feature)
        alpha = hmm_model.forward(emission_matrix, use_log=True)
        beta = hmm_model.backward(emission_matrix, use_log=True)

        # Compute gamma and xi for this sequence
        gamma = hmm_model.compute_gamma(alpha, beta, use_log=True)
        xi = hmm_model.compute_xi(alpha, beta, emission_matrix, use_log=True)

        # Accumulate statistics
        aggregated_gamma += np.sum(gamma, axis=1, keepdims=True)
        aggregated_xi += np.sum(xi, axis=0)
        print(f"\nProcessed sequence {seq_idx + 1}")
        print(f"Sequence length: {heed_feature.shape[1]} frames")
        assert np.isclose(
            np.sum(gamma), heed_feature.shape[1]
        ), "Gamma sum should equal T"
        assert np.isclose(
            np.sum(xi), heed_feature.shape[1] - 1
        ), "Xi sum should equal T-1"

    print("\nInitial A matrix:")
    hmm_model.print_transition_matrix()

    hmm_model.update_A(aggregated_xi, aggregated_gamma)

    print("\nUpdated A matrix:")
    hmm_model.print_transition_matrix()

    assert (
        hmm_model.A[0, 1] == 1.0
    ), "Entry state must transition to first state with prob 1"
    assert np.all(
        hmm_model.A[0, 2:] == 0
    ), "Entry state should have no other transitions"

    #  Check main state transitions
    for i in range(1, hmm_model.num_states + 1):
        row_probs = hmm_model.A[i, :]

        # Verify probability normalization
        assert np.isclose(
            np.sum(row_probs), 1.0, atol=1e-10
        ), f"Row {i} probabilities must sum to 1"

        # Verify left-right structure
        if i < hmm_model.num_states:  # Not the last state
            allowed = np.zeros_like(row_probs)
            allowed[i] = 1  # Self-transition
            allowed[i + 1] = 1  # Next state
            assert np.all(
                (row_probs > 0) == allowed
            ), f"State {i} has invalid transitions"

            assert row_probs[i] > 0, f"State {i} should have non-zero self-transition"
            assert (
                row_probs[i + 1] > 0
            ), f"State {i} should have non-zero forward transition"
        else:
            allowed = np.zeros_like(row_probs)
            allowed[i] = 1
            allowed[i + 1] = 1
            assert np.all(
                (row_probs > 0) == allowed
            ), f"Last state has invalid transitions"

    assert np.all(
        hmm_model.A[-1, :] == 0
    ), "Exit state should have no outgoing transitions"
    assert np.all(
        np.tril(hmm_model.A[1:-1, 1:-1], k=-1) == 0
    ), "No backward transitions allowed"
    assert np.all(
        np.triu(hmm_model.A[1:-1, 1:-1], k=2) == 0
    ), "No skipping states allowed"

    main_diag = np.diag(hmm_model.A[1:-1, 1:-1])
    assert np.all(
        (main_diag > 0.5) & (main_diag < 0.95)
    ), "Self-transition probabilities should be reasonable (between 0.5 and 0.95)"

    # Print statistics to debug
    print("\nTransition Statistics:")
    print(f"Total gamma sum across all sequences: {np.sum(aggregated_gamma):.3f}")
    print(f"Total xi sum across all sequences: {np.sum(aggregated_xi):.3f}")
    print(f"Average self-transition probability: {np.mean(main_diag):.3f}")
    print(f"Min self-transition probability: {np.min(main_diag):.3f}")
    print(f"Max self-transition probability: {np.max(main_diag):.3f}")

    forward_probs = [hmm_model.A[i, i + 1] for i in range(1, hmm_model.num_states + 1)]
    print(f"\nForward transition probabilities: {[f'{p:.3f}' for p in forward_probs]}")


def test_update_emissions(hmm_model, heed_features):
    """
    Basic test for emission parameter updates using 'heed' sequences.
    Verifies fundamental properties of means and covariances after updates.
    """
    # Store initial parameters to check if they change
    initial_means = hmm_model.B["mean"].copy()
    initial_variances = hmm_model.B["covariance"].copy()
    
    # Use first three sequences for a simple test
    gamma_per_seq = []
    
    # Compute gamma for each sequence
    for features in heed_features:
        emission_matrix = hmm_model.compute_log_emission_matrix(features)
        alpha = hmm_model.forward(emission_matrix, use_log=True)
        beta = hmm_model.backward(emission_matrix, use_log=True)
        gamma = hmm_model.compute_gamma(alpha, beta, use_log=True)
        gamma_per_seq.append(gamma)
        
        # Print basic sequence info for debugging
        print(f"\nSequence length: {features.shape[1]} frames")
        print(f"Gamma sum: {np.sum(gamma):.3f}")
    
    # Update emission parameters
    hmm_model.update_B(heed_features, gamma_per_seq)
    
    # === Basic Verification Checks ===
    
    # 1. Check that parameters actually changed
    assert not np.array_equal(initial_means, hmm_model.B["mean"]), \
        "Means should be updated"
    assert not np.array_equal(initial_variances, hmm_model.B["covariance"]), \
        "Variances should be updated"
    
    # 2. Check mathematical validity
    assert np.all(np.isfinite(hmm_model.B["mean"])), \
        "All means should be finite"
    assert np.all(np.isfinite(hmm_model.B["covariance"])), \
        "All variances should be finite"
    assert np.all(hmm_model.B["covariance"] > 0), \
        "All variances should be positive"
    
    # Print before/after statistics for inspection
    print("\nMean value ranges:")
    print(f"Before: [{np.min(initial_means):.3f}, {np.max(initial_means):.3f}]")
    print(f"After:  [{np.min(hmm_model.B['mean']):.3f}, {np.max(hmm_model.B['mean']):.3f}]")
    
    print("\nVariance ranges:")
    print(f"Before: [{np.min(initial_variances):.3f}, {np.max(initial_variances):.3f}]")
    print(f"After:  [{np.min(hmm_model.B['covariance']):.3f}, {np.max(hmm_model.B['covariance']):.3f}]")


def test_baum_welch(hmm_model, heed_features):
    """
    Test the full Baum-Welch algorithm using the 'heed' sequences.
    Verifies that the model parameters converge to a stable state.
    """
    hmm_model.baum_welch(heed_features)