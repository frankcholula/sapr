import pytest
import numpy as np
from hmm import HMM
from mfcc_extract import load_mfccs


@pytest.fixture
def feature_set():
    return load_mfccs("feature_set")


@pytest.fixture
def hmm_model(feature_set):
    return HMM(8, 13, feature_set)


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
    hmm_model.print_matrix(B_probs, "Emission Matrix", col="T", idx="State", start_idx=1, start_col=1)


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
    hmm_model.print_matrix(alpha, "Alpha Matrix", col="T", idx="State", start_idx=0, start_col=1)
    hmm_model.print_matrix(beta, "Beta Matrix", col="T", idx="State", start_idx=0, start_col=1)


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
    hmm_model.print_matrix(gamma, "Gamma Matrix", col="T", idx="State", start_idx=1, start_col=1)
    hmm_model.print_matrix(xi_summed, "Summed Xi Matrix", col="T", idx="State", start_idx=1, start_col=1)
    np.testing.assert_array_almost_equal(gamma[:, :-1], xi_summed)


def test_update_transitions(hmm_model, feature_set):
    # Store initial probabilities
    initial_A = hmm_model.A.copy()
    initial_B = hmm_model.B.copy()

    # Calculate all necessary probabilities for one sequence
    emission_matrix = hmm_model.compute_log_emission_matrix(feature_set[0])
    alpha = hmm_model.forward(emission_matrix, use_log=True)
    beta = hmm_model.backward(emission_matrix, use_log=True)
    gamma = hmm_model.compute_gamma(alpha, beta)
    xi = hmm_model.compute_xi(alpha, beta, emission_matrix)
    
    hmm_model.update_A(xi, gamma)
    hmm_model.update_B(feature_set[0], gamma)
    
    # Check that A matrix actually changed
    assert not np.array_equal(initial_A, hmm_model.A), "Transition matrix should be updated"
    
    # Check structure is preserved between initial and updated A
    # Zero elements should stay zero (structure preserved)
    assert np.all((initial_A == 0) == (hmm_model.A == 0)), "Zero/non-zero structure should be preserved"
    hmm_model.print_matrix(initial_A, "Initial A Matrix", col="State", idx= "State", start_idx=0, start_col=0)
    hmm_model.print_matrix(hmm_model.A, "Updated A Matrix", col="State", idx= "State", start_idx=0, start_col=0)
    hmm_model.print_matrix(initial_B["mean"], "Initial B Means", col="State", idx="MFCC", start_idx=1, start_col=1)
    hmm_model.print_matrix(hmm_model.B["mean"], "Updated B Means", col="State", idx="MFCC", start_idx=1, start_col=1)
    hmm_model.print_matrix(initial_B["covariance"], "Initial B Variances", col="State", idx="MFCC", start_idx=1, start_col=1)
    hmm_model.print_matrix(hmm_model.B["covariance"], "Updated B Variances", col="State", idx="MFCC", start_idx=1, start_col=1)
