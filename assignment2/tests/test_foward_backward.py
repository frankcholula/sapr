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


def test_emission_matrix(hmm_model, feature_set):
    test_features = feature_set[0]
    B_probs = hmm_model.compute_emission_matrix(test_features)
    assert B_probs.shape == (test_features.shape[1], hmm_model.total_states)
    assert np.all(B_probs <= 0), "Log probabilities should be non-positive"
    assert np.all(
        np.isfinite(B_probs[B_probs != -np.inf])
    ), "Log probabilities should be finite where not -inf"

    assert np.all(
        B_probs[:, 0] == -np.inf
    ), "Entry state should have -inf log probabilities"
    assert np.all(
        B_probs[:, -1] == -np.inf
    ), "Exit state should have -inf log probabilities"

    real_states_probs = B_probs[:, 1:-1]  # Exclude entry and exit states
    first_frame_probs = real_states_probs[0, :]
    print(f"\nFirst Frame Log Probabilities:\n{first_frame_probs}")
    prob_std = np.std(first_frame_probs)
    assert prob_std < 1e-10, "Initial log probabilities should be similar across states"
    hmm_model.print_matrix(
        B_probs, "Emission Matrix", col="State", idx="T", start_idx=1, start_col=1
    )
