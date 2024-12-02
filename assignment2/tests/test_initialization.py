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

def test_hmm_initialization(hmm_model, feature_set):
    """
    Test the initialization of HMM emission parameters.
    
    This test verifies that:
    1. The parameter matrices have correct dimensions
    2. The variance floor is properly applied for each state
    3. Initial parameters are properly replicated across states
    4. The parameters are finite and make statistical sense
    """
    # Test 1: Check dimensions of emission parameter matrices
    assert hmm_model.B["mean"].shape == (13, 8), \
        f"Mean matrix should be (num_obs, num_states), got shape {hmm_model.B['mean'].shape}"
    assert hmm_model.B["covariance"].shape == (13, 8), \
        f"Covariance matrix should be (num_obs, num_states), got shape {hmm_model.B['covariance'].shape}"
    
    # Test 2: Check variance floor application
    # We compute floor based on mean of variances across all states
    mean_variance = np.mean(hmm_model.B["covariance"])
    var_floor = 0.01 * mean_variance
    assert np.all(hmm_model.B["covariance"] >= var_floor), \
        f"All variances should be >= variance floor ({var_floor}). Min variance: {np.min(hmm_model.B['covariance'])}"
    
    # Test 3: Verify parameter tiling
    # At initialization, all states should have identical parameters
    for i in range(1, 8):
        np.testing.assert_array_almost_equal(
            hmm_model.B["mean"][:, 0], 
            hmm_model.B["mean"][:, i],
            err_msg=f"State {i} means differ from state 0: \nState 0: {hmm_model.B['mean'][:, 0]}\nState {i}: {hmm_model.B['mean'][:, i]}"
        )
        np.testing.assert_array_almost_equal(
            hmm_model.B["covariance"][:, 0],
            hmm_model.B["covariance"][:, i],
            err_msg=f"State {i} covariances differ from state 0: \nState 0: {hmm_model.B['covariance'][:, 0]}\nState {i}: {hmm_model.B['covariance'][:, i]}"
        )
    
    # Test 4: Check statistical validity
    assert np.all(np.isfinite(hmm_model.B["mean"])), \
        "Mean matrix contains non-finite values"
    assert np.all(np.isfinite(hmm_model.B["covariance"])), \
        "Covariance matrix contains non-finite values"
    assert np.all(hmm_model.B["covariance"] > 0), \
        "Covariance matrix contains non-positive values"
    
    # Print initialization statistics for debugging
    print("\nHMM Initialization Statistics:")
    print(f"Mean range: [{np.min(hmm_model.B['mean']):.3f}, {np.max(hmm_model.B['mean']):.3f}]")
    print(f"Variance range: [{np.min(hmm_model.B['covariance']):.3f}, {np.max(hmm_model.B['covariance']):.3f}]")
    print(f"Variance floor: {var_floor:.3f}")
