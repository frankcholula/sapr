from hmmlearn import hmm
import numpy as np
from mfcc_extract import load_mfccs_by_word

def initialize_hmm_parameters(features_list, n_states=8):
    """
    Initialize HMM parameters using global statistics and proper transition probabilities
    based on average sequence duration.
    
    Args:
        features_list: List of MFCC feature matrices
        n_states: Number of HMM states
    """
    # Prepare data
    lengths = [f.shape[1] for f in features_list]
    X = np.concatenate([f.T for f in features_list], axis=0)
    
    # Calculate global statistics for emissions
    means = np.mean(X, axis=0)
    covars = np.var(X, axis=0)
    min_var = 1e-3  # Variance floor to prevent numerical issues
    covars = np.maximum(covars, min_var)
    
    # Calculate average sequence duration
    avg_duration = np.mean(lengths)
    frames_per_state = avg_duration / n_states
    
    # Calculate transition probabilities using the formula from assignment
    aii = np.exp(-1 / (frames_per_state - 1))  # Self-transition probability
    aij = 1 - aii  # Forward transition probability
    
    print(f"Average frames per state: {frames_per_state:.2f}")
    print(f"Calculated self-transition probability: {aii:.3f}")
    print(f"Calculated forward transition probability: {aij:.3f}")
    
    # Initialize the model
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=15,
        init_params="",
        params="stmc",
        verbose=True
    )
    
    # Set start probabilities (start in first state)
    startprob = np.zeros(n_states)
    startprob[0] = 1.0
    model.startprob_ = startprob
    
    # Set transition matrix using calculated probabilities
    transmat = np.zeros((n_states, n_states))
    for i in range(n_states-1):
        transmat[i, i] = aii
        transmat[i, i+1] = aij
    transmat[-1, -1] = 1.0
    model.transmat_ = transmat
    
    # Set emission parameters using global statistics
    model.means_ = np.tile(means, (n_states, 1))
    model.covars_ = np.tile(covars, (n_states, 1))
    
    return model, X, lengths

def train_word_hmm(word="heed", n_states=8):
    """Train HMM for a single word with proper initialization"""
    print(f"\nTraining HMM for word '{word}'...")
    
    # Load features
    features = load_mfccs_by_word("feature_set", word)
    
    # Initialize model with proper parameters
    model, X, lengths = initialize_hmm_parameters(features, n_states)
    
    # Train the model
    try:
        model.fit(X, lengths)
        log_likelihood = model.score(X, lengths)
        print(f"Training completed. Final log-likelihood: {log_likelihood:.2f}")
        return model, log_likelihood
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None, None


if __name__ == "__main__":
    model, score = train_word_hmm("heed")
