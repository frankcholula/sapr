from hmmlearn import hmm
import numpy as np
from scipy.special import logsumexp
from mfcc_extract import load_mfccs_by_word

def calculate_emission_probs(X, means, covars):
    """
    Calculate log emission probabilities for a Gaussian HMM.
    
    Args:
        X: Features of shape (T, n_features)
        means: Mean vectors of shape (n_states, n_features)
        covars: Covariance matrices of shape (n_states, n_features)
    
    Returns:
        Log emission probabilities of shape (T, n_states)
    """
    n_samples, n_features = X.shape
    n_states = means.shape[0]
    
    # Initialize output matrix
    log_emissions = np.zeros((n_samples, n_states))
    
    # Constants for Gaussian probability calculation
    const = -0.5 * (n_features * np.log(2 * np.pi))
    
    # Calculate for each state
    for j in range(n_states):
        # Get state-specific parameters
        state_mean = means[j]
        state_covar = covars[j]
        
        # Calculate log determinant term
        log_det = -0.5 * np.sum(np.log(state_covar))
        
        # Calculate Mahalanobis distance
        diff = X - state_mean
        mahalanobis = -0.5 * np.sum((diff ** 2) / state_covar, axis=1)
        
        # Combine terms
        log_emissions[:, j] = const + log_det + mahalanobis
    
    return log_emissions

def inspect_forward_pass(word="heed"):
    """
    Analyze emission probabilities and forward pass calculations.
    """
    # Load and prepare features
    features = load_mfccs_by_word("feature_set", word)
    X = features[0].T  # Shape becomes (T, 13)
    
    print("\nData Information:")
    print(f"Original feature shape: {features[0].shape}")
    print(f"Transformed feature shape: {X.shape}")
    
    # Model parameters
    n_states = 8
    n_features = X.shape[1]  # Should be 13
    
    # Calculate global statistics
    means = np.mean(X, axis=0)
    covars = np.var(X, axis=0)
    
    # Add variance floor
    min_covar = 1e-3
    covars = np.maximum(covars, min_covar)
    
    # Replicate for each state
    state_means = np.tile(means, (n_states, 1))
    state_covars = np.tile(covars, (n_states, 1))
    
    # Calculate emission probabilities
    log_emissions = calculate_emission_probs(X, state_means, state_covars)
    
    # Initialize transition matrix
    transmat = np.zeros((n_states, n_states))
    for i in range(n_states-1):
        transmat[i, i] = 0.8
        transmat[i, i+1] = 0.2
    transmat[-1, -1] = 1.0
    
    # Initialize start probabilities
    startprob = np.zeros(n_states)
    startprob[0] = 1.0
    
    # Forward pass calculation
    T = len(X)
    log_alpha = np.full((T, n_states), -np.inf)
    
    # Initialize first timestep
    log_alpha[0] = np.log(startprob) + log_emissions[0]
    
    # Forward recursion
    for t in range(1, T):
        for j in range(n_states):
            # Previous values + transitions
            prev_alpha = log_alpha[t-1] + np.log(transmat[:, j])
            log_alpha[t, j] = logsumexp(prev_alpha) + log_emissions[t, j]
    
    print("\nEmission Probability Statistics:")
    print(f"Min log emission: {log_emissions.min():.2f}")
    print(f"Max log emission: {log_emissions.max():.2f}")
    print(f"Mean log emission: {log_emissions.mean():.2f}")
    
    print("\nForward Variable Statistics:")
    valid_alphas = log_alpha[log_alpha > -np.inf]
    print(f"Min log alpha (excluding -inf): {valid_alphas.min():.2f}")
    print(f"Max log alpha: {log_alpha.max():.2f}")
    print(f"Mean log alpha (excluding -inf): {valid_alphas.mean():.2f}")
    
    print("\nDetailed Example (first 3 timesteps, first 3 states):")
    print("\nLog Emissions:")
    print(log_emissions[:3, :3])
    print("\nLog Alpha:")
    print(log_alpha[:3, :3])
    
    # Final log-likelihood
    log_likelihood = logsumexp(log_alpha[-1])
    print("\nSequence log-likelihood:", log_likelihood)
    
    return log_emissions, log_alpha, state_means, state_covars

if __name__ == "__main__":
    emissions, alphas, means, covars = inspect_forward_pass()