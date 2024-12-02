from hmm import HMM
from mfcc_extract import load_mfccs_by_word
import numpy as np

def test_hmm_initialization(word="heed", n_states=8):
    """
    Test the initialization of HMM parameters and compare with reference values.
    """
    print("Testing HMM initialization parameters\n")
    
    # Load features for the word
    features = load_mfccs_by_word("feature_set", word)
    
    # Calculate reference values directly
    total_frames = sum(f.shape[1] for f in features)
    avg_frames_per_state = total_frames / (len(features) * n_states)
    ref_aii = np.exp(-1 / (avg_frames_per_state - 1))
    ref_aij = 1 - ref_aii
    
    print("Reference Values:")
    print(f"Total frames: {total_frames}")
    print(f"Number of sequences: {len(features)}")
    print(f"Average frames per state: {avg_frames_per_state:.2f}")
    print(f"Calculated self-transition (aii): {ref_aii:.3f}")
    print(f"Calculated forward transition (aij): {ref_aij:.3f}")
    
    # Initialize your HMM
    print("\nYour HMM Initialization:")
    hmm = HMM(n_states, 13, features, model_name=word)
    
    # Get transition probabilities from your A matrix (excluding entry/exit states)
    your_aii = hmm.A[1, 1]  # First real state self-transition
    your_aij = hmm.A[1, 2]  # First real state forward transition
    
    print(f"Your self-transition (aii): {your_aii:.3f}")
    print(f"Your forward transition (aij): {your_aij:.3f}")
    
    # Compare global statistics
    print("\nComparing means and variances:")
    
    # Calculate reference global statistics
    features_concat = np.concatenate([f for f in features], axis=1)
    ref_means = np.mean(features_concat, axis=1)
    ref_vars = np.var(features_concat, axis=1)
    
    print("\nMeans comparison (first 3 dimensions):")
    print(f"Reference: {ref_means[:3]}")
    print(f"Your HMM:   {hmm.B['mean'][:3, 0]}")  # First state means
    
    print("\nVariances comparison (first 3 dimensions):")
    print(f"Reference: {ref_vars[:3]}")
    print(f"Your HMM:   {hmm.B['covariance'][:3, 0]}")  # First state variances
    
    # Calculate differences
    mean_diff = np.abs(ref_means - hmm.B['mean'][:, 0])
    var_diff = np.abs(ref_vars - hmm.B['covariance'][:, 0])
    
    print("\nDifferences:")
    print(f"Maximum mean difference: {np.max(mean_diff):.6f}")
    print(f"Maximum variance difference: {np.max(var_diff):.6f}")

if __name__ == "__main__":
    test_hmm_initialization()