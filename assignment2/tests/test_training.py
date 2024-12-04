import pytest
import numpy as np
from custom_hmm import HMM
from mfcc_extract import load_mfccs, load_mfccs_by_word
import pandas as pd


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


def test_update_transitions(hmm_model, heed_features):
    """Test HMM transition matrix updates with multiple MFCC feature sequences."""
    # Accumulate statistics across all sequences
    aggregated_gamma = np.zeros(hmm_model.total_states)
    aggregated_xi = np.zeros((hmm_model.total_states, hmm_model.total_states))

    for features in heed_features:
        emission_matrix = hmm_model.compute_emission_matrix(features)
        alpha = hmm_model.forward(emission_matrix)
        beta = hmm_model.backward(emission_matrix)
        gamma = hmm_model.compute_gamma(alpha, beta)
        xi = hmm_model.compute_xi(alpha, beta, emission_matrix)

        # Sum over time
        aggregated_gamma += np.sum(gamma[:-1], axis=0)  # Exclude last frame
        aggregated_xi += np.sum(xi, axis=0)  # Sum over time

    print("\nDiagnostic Information:")
    print(f"Aggregated gamma shape: {aggregated_gamma.shape}")
    print(f"Aggregated xi shape: {aggregated_xi.shape}")
    print("\nAggregated gamma sums per state:")
    for i in range(hmm_model.total_states):
        print(f"State {i}: {aggregated_gamma[i]:.6f}")

    print("\nXi transition sums for first real state (state 1):")
    print(f"Sum of transitions from state 1: {np.sum(aggregated_xi[1, :]):.6f}")
    print(f"Self-loop (1->1): {aggregated_xi[1, 1]:.6f}")
    print(f"Forward (1->2): {aggregated_xi[1, 2]:.6f}")

    # Store initial A matrix
    initial_A = hmm_model.A.copy()
    print("\nInitial A matrix:")
    hmm_model.print_matrix(initial_A, "Initial Transition Matrix")

    # Update transition matrix
    hmm_model.update_A(aggregated_xi, aggregated_gamma)

    print("\nUpdated A matrix:")
    hmm_model.print_matrix(hmm_model.A, "Updated Transition Matrix")

    # Print row sums of updated matrix
    print("\nRow sums of updated transition matrix:")
    for i in range(hmm_model.total_states):
        row_sum = np.sum(hmm_model.A[i, :])
        print(f"State {i}: {row_sum:.10f}")

    # Basic structural tests
    assert (
        hmm_model.A[0, 1] == 1.0
    ), "Entry state must transition to first state with prob 1"
    assert np.all(
        hmm_model.A[0, [0, *range(2, hmm_model.total_states)]] == 0
    ), "Entry state should have no other transitions"
    assert hmm_model.A[-1, -1] == 1.0, "Exit state should have self-loop of 1"
    assert np.all(
        hmm_model.A[-1, :-1] == 0
    ), "Exit state should have no other transitions"

    # Check row sums and transitions
    for i in range(1, hmm_model.num_states + 1):
        row_sum = np.sum(hmm_model.A[i, :])
        print(f"\nState {i} transitions:")
        print(f"Self-loop (a_{i}{i}): {hmm_model.A[i, i]:.6f}")
        if i < hmm_model.num_states:
            print(f"Forward (a_{i}{i+1}): {hmm_model.A[i, i+1]:.6f}")
        print(f"Row sum: {row_sum:.10f}")

        assert np.isclose(row_sum, 1.0, atol=1e-10), f"Row {i} must sum to 1"


def test_update_emissions(hmm_model, heed_features):
    """Test HMM emission parameter updates with 'heed' sequences."""
    # Store initial parameters
    initial_means = hmm_model.B["mean"].copy()
    initial_covars = hmm_model.B["covariance"].copy()

    # Print initial parameters
    hmm_model.print_matrix(initial_means, "Initial Means", col="MFCC", idx="State")
    hmm_model.print_matrix(
        initial_covars, "Initial Covariances", col="MFCC", idx="State"
    )

    # Collect forward-backward statistics for all sequences
    gamma_per_seq = []
    for features in heed_features:
        emission_matrix = hmm_model.compute_emission_matrix(features)
        alpha = hmm_model.forward(emission_matrix)
        beta = hmm_model.backward(emission_matrix)
        gamma = hmm_model.compute_gamma(alpha, beta)
        gamma_per_seq.append(gamma)

    # Update emission parameters
    hmm_model.update_B(heed_features, gamma_per_seq)

    # Print updated parameters
    hmm_model.print_matrix(
        hmm_model.B["mean"], "Updated Means", col="MFCC", idx="State"
    )
    hmm_model.print_matrix(
        hmm_model.B["covariance"], "Updated Covariances", col="MFCC", idx="State"
    )

    # 1. Check dimensions
    assert hmm_model.B["mean"].shape == (
        10,
        13,
    ), "Mean dimensions should be (10 states, 13 observations)"
    assert hmm_model.B["covariance"].shape == (
        10,
        13,
    ), "Covariance dimensions should be (10 states, 13 observations)"

    # 2. Check entry/exit states remain zero (non-emitting)
    assert np.all(hmm_model.B["mean"][0] == 0), "Entry state means should be zero"
    assert np.all(hmm_model.B["mean"][-1] == 0), "Exit state means should be zero"
    assert np.all(
        hmm_model.B["covariance"][0] == 0
    ), "Entry state covariances should be zero"
    assert np.all(
        hmm_model.B["covariance"][-1] == 0
    ), "Exit state covariances should be zero"

    # 3. Check real states have been updated
    assert not np.all(
        hmm_model.B["mean"][1:-1] == 0
    ), "Real state means should be updated"
    assert not np.all(
        hmm_model.B["covariance"][1:-1] == 0
    ), "Real state covariances should be updated"

    # 4. Check mathematical validity for real states
    assert np.all(
        np.isfinite(hmm_model.B["mean"][1:-1])
    ), "All real state means should be finite"
    assert np.all(
        np.isfinite(hmm_model.B["covariance"][1:-1])
    ), "All real state covariances should be finite"
    assert np.all(
        hmm_model.B["covariance"][1:-1] > 0
    ), "All real state covariances should be positive"

    # 5. Check variance floor is applied to real states
    var_floor = hmm_model.var_floor_factor * np.mean(hmm_model.B["covariance"][1:-1])
    assert np.all(
        hmm_model.B["covariance"][1:-1] >= var_floor
    ), "All real state variances should be above the floor value"


def test_baum_welch(hmm_model, heed_features):
    """
    Test the full Baum-Welch algorithm using the 'heed' sequences.
    Verifies that the model parameters converge to a stable state.
    """
    hmm_model.baum_welch(heed_features)
