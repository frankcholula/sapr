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

def test_emission_probability(hmm_model, feature_set):
    # Take first sequence from feature_set
    test_features = feature_set[0]
    
    # Calculate emission probabilities
    B_probs = hmm_model.compute_emission_probability(test_features)
    
    # Test shape
    assert B_probs.shape == (8, test_features.shape[1])
    
    # Test basic properties
    assert np.all(B_probs >= 0), "Probabilities should be non-negative"
    assert np.all(np.isfinite(B_probs)), "Probabilities should be finite"
    
    # Since we initialized with global means, first frame probabilities should be similar
    first_frame_probs = B_probs[:, 0]
    print(f"\nFirst Frame Probabilities:\n{first_frame_probs}")
    prob_std = np.std(first_frame_probs)
    assert prob_std < 1e-10, "Initial probabilities should be similar across states"
