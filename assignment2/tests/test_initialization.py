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
    ), "Covariance matrix should be (total_states, num_obs)"

    var_floor = hmm_model.var_floor_factor * np.mean(hmm_model.global_variance)
    assert np.all(
        hmm_model.B["covariance"] >= var_floor
    ), "All variances should be >= variance floor"

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

    np.testing.assert_array_almost_equal(
        hmm_model.B["mean"][0],
        hmm_model.global_mean,  # Added test for global_mean
        err_msg="State means should match global mean at initialization",
    )
    np.testing.assert_array_almost_equal(
        hmm_model.B["covariance"][0],
        hmm_model.global_variance,  # Added test for global_variance
        err_msg="State covariances should match global variance at initialization",
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