import pytest
import numpy as np
from custom_hmm import HMM
from mfcc_extract import load_mfccs


@pytest.fixture
def feature_set():
    return load_mfccs("feature_set")


@pytest.fixture
def hmm_model(feature_set):
    return HMM(8, 13, feature_set)


def test_hmm_initialization(hmm_model, feature_set):
    total_states = hmm_model.num_states + 2  # Including entry and exit states

    # Test dimensions
    assert hmm_model.B["mean"].shape == (
        total_states,
        13,
    ), "Mean matrix should be (total_states, num_obs)"
    assert hmm_model.B["covariance"].shape == (
        total_states,
        13,
    ), "Covariance matrix should be (total_states, num_obs)"

    # Test variance floor was applied
    var_floor = hmm_model.var_floor_factor * np.mean(hmm_model.variance)
    assert np.all(
        hmm_model.B["covariance"] >= var_floor
    ), "All variances should be >= variance floor"

    # Test that means and covariances are properly tiled
    # Since we initialize with global stats, all rows should be identical
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

    # Test pi initialization
    assert hmm_model.pi.shape == (
        total_states,
    ), "Initial state probabilities should include entry/exit states"
    assert hmm_model.pi[0] == 1.0, "Entry state should have probability 1.0"
    assert np.all(
        hmm_model.pi[1:] == 0.0
    ), "All other states should have probability 0.0"
