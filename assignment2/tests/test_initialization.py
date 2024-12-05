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


def test_hmm_shapes(hmm_model):
    """Test the shapes of initialized matrices"""
    total_states = hmm_model.num_states + 2

    assert hmm_model.B["mean"].shape == (
        total_states,
        13,
    ), "Mean matrix shape incorrect"
    assert hmm_model.B["covariance"].shape == (
        total_states,
        13,
        13,
    ), "Covariance matrix shape incorrect"
    assert hmm_model.global_mean.shape == (13,), "Global mean shape incorrect"
    assert hmm_model.global_covariance.shape == (
        13,
        13,
    ), "Global covariance shape incorrect"


def test_covariance_initialization(hmm_model):
    """Test that covariance matrices are properly initialized as diagonal"""
    # Check off-diagonal elements are zero
    for state in range(hmm_model.total_states):
        cov_matrix = hmm_model.B["covariance"][state]
        off_diag_mask = ~np.eye(13, dtype=bool)
        assert np.allclose(
            cov_matrix[off_diag_mask], 0
        ), f"State {state} has non-zero off-diagonal elements"

    # Check variance floor is applied
    var_floor = hmm_model.var_floor_factor * np.mean(
        np.diag(hmm_model.global_covariance)
    )
    for state in range(hmm_model.total_states):
        diag_elements = np.diag(hmm_model.B["covariance"][state])
        assert np.all(
            diag_elements >= var_floor
        ), f"State {state} has variances below floor"


def test_state_initialization(hmm_model):
    """Test that all states are initialized with same parameters"""
    for i in range(1, hmm_model.total_states):
        np.testing.assert_array_almost_equal(
            hmm_model.B["mean"][0],
            hmm_model.B["mean"][i],
            err_msg=f"State {i} mean differs from state 0",
        )
        np.testing.assert_array_almost_equal(
            hmm_model.B["covariance"][0],
            hmm_model.B["covariance"][i],
            err_msg=f"State {i} covariance differs from state 0",
        )


def test_initial_probabilities(hmm_model):
    """Test initial state probabilities"""
    assert hmm_model.pi[0] == 1.0, "Entry state should have probability 1.0"
    assert np.all(hmm_model.pi[1:] == 0.0), "Other states should have probability 0.0"


def print_debug_info(hmm_model):
    """Print debug information about the model's initialization"""
    print("\nGlobal Mean:", hmm_model.global_mean)
    print("\nGlobal Covariance Diagonal:", np.diag(hmm_model.global_covariance))
    print("\nFirst State Covariance Diagonal:", np.diag(hmm_model.B["covariance"][1]))

    # Check if any off-diagonal elements are non-zero
    off_diag_sum = np.sum(
        np.abs(
            hmm_model.global_covariance - np.diag(np.diag(hmm_model.global_covariance))
        )
    )
    print(f"\nSum of absolute off-diagonal elements: {off_diag_sum:.2e}")
