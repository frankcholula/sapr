import pytest
import numpy as np
from custom_hmm import HMM
from hmmlearn_hmm import HMMLearnModel
from mfcc_extract import load_mfccs
from train import pretty_print_matrix


@pytest.fixture
def feature_set():
    return load_mfccs("feature_set")


@pytest.fixture
def hmm_model(feature_set):
    return HMM(8, 13, feature_set)


def test_hmm_initialization(hmm_model):
    total_states = hmm_model.num_states + 2  # Including entry and exit states

    # Test dimensions
    assert hmm_model.B["mean"].shape == (
        total_states,
        13,
    ), "Mean matrix should be (total_states, num_obs)"
    assert hmm_model.B["covariance"].shape == (
        total_states,
        13,
        13,
    ), "Covariance matrix should be (total_states, num_obs, num_obs)"

    # Test variance floor on diagonal elements
    var_floor = hmm_model.var_floor_factor * np.mean(np.diag(hmm_model.global_covariance))
    for state in range(total_states):
        diag_elements = np.diag(hmm_model.B["covariance"][state])
        assert np.all(
            diag_elements >= var_floor
        ), f"State {state} diagonal variances should be >= variance floor"

    # Test covariance matrix properties
    for state in range(total_states):
        cov_matrix = hmm_model.B["covariance"][state]
        # Test symmetry
        assert np.allclose(
            cov_matrix, 
            cov_matrix.T
        ), f"Covariance matrix for state {state} should be symmetric"
        # Test positive semi-definiteness
        eigenvals = np.linalg.eigvals(cov_matrix)
        assert np.all(
            eigenvals >= -1e-10
        ), f"Covariance matrix for state {state} should be positive semi-definite"

    # Test initialization of means and covariances across states
    for i in range(1, total_states):
        np.testing.assert_array_almost_equal(
            hmm_model.B["mean"][0],
            hmm_model.B["mean"][i],
            err_msg="All state means should be identical at initialization",
        )
        np.testing.assert_array_almost_equal(
            hmm_model.B["covariance"][0],
            hmm_model.B["covariance"][i],
            err_msg="All state covariances should be identical at initialization",
        )

    # Test means and covariances match global statistics
    np.testing.assert_array_almost_equal(
        hmm_model.B["mean"][0],
        hmm_model.global_mean,
        err_msg="State means should match global mean at initialization",
    )
    np.testing.assert_array_almost_equal(
        hmm_model.B["covariance"][0],
        hmm_model.global_covariance,
        err_msg="State covariances should match global covariance at initialization",
    )

    assert hmm_model.pi.shape == (
        total_states,
    ), "Initial state probabilities should include entry/exit states"
    assert hmm_model.pi[0] == 1.0, "Entry state should have probability 1.0"
    assert np.all(
        hmm_model.pi[1:] == 0.0
    ), "All other states should have probability 0.0"

    compare_model = HMMLearnModel(8, "test")
    pretty_print_matrix(hmm_model.A)
    pretty_print_matrix(compare_model.initialize_transmat())


def test_global_mean_and_covariance(hmm_model, feature_set):
    # Test mean shape and values
    assert hmm_model.global_mean.shape == (13,)
    assert np.allclose(
        hmm_model.global_mean, 
        np.mean(np.concatenate(feature_set, axis=1), axis=1)
    ), "Global mean should be the mean of all features"
    
    # Test covariance shape and properties
    assert hmm_model.global_covariance.shape == (13, 13), "Global covariance should be (num_obs, num_obs)"
    assert np.allclose(
        hmm_model.global_covariance,
        hmm_model.global_covariance.T
    ), "Global covariance should be symmetric"
    
    # Test positive semi-definiteness
    eigenvals = np.linalg.eigvals(hmm_model.global_covariance)
    assert np.all(
        eigenvals >= -1e-10
    ), "Global covariance should be positive semi-definite"
    
    print("\nGlobal Statistics:")
    print(f"Mean shape: {hmm_model.global_mean.shape}")
    print(f"Covariance shape: {hmm_model.global_covariance.shape}")
    print(f"Mean:\n{hmm_model.global_mean}")
    print(f"Covariance diagonal:\n{np.diag(hmm_model.global_covariance)}")
