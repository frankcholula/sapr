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
   # Test dimensions
   assert hmm_model.B["mean"].shape == (13, 8), "Mean matrix should be (num_obs, num_states)"
   assert hmm_model.B["covariance"].shape == (13, 8), "Covariance matrix should be (num_obs, num_states)"
   
   # Test variance floor was applied
   var_floor = 0.01 * np.mean(hmm_model.variance)
   assert np.all(hmm_model.B["covariance"] >= var_floor), "All variances should be >= variance floor"
   
   # Test that means and covariances are properly tiled
   # Since we initialize with global stats, all columns should be identical
   for i in range(1, 8):
       np.testing.assert_array_almost_equal(
           hmm_model.B["mean"][:, 0], 
           hmm_model.B["mean"][:, i],
           err_msg="All state means should be identical at initialization"
       )
       np.testing.assert_array_almost_equal(
           hmm_model.B["covariance"][:, 0],
           hmm_model.B["covariance"][:, i],
           err_msg="All state covariances should be identical at initialization"
       )
