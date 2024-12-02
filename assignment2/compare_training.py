from hmm import HMM
from mfcc_extract import load_mfccs_by_word
import numpy as np
import pandas as pd

def debug_training_iteration(hmm_model, features, iteration):
    """Debug a single training iteration by examining intermediate values."""
    print(f"\nDEBUG: Iteration {iteration}")
    
    total_log_likelihood = 0
    all_gammas = []
    all_xis = []
    
    # Process each sequence
    for seq_idx, features_seq in enumerate(features):
        # Get emission probabilities
        log_B = hmm_model.compute_log_emission_matrix(features_seq)
        print(f"\nSequence {seq_idx + 1}")
        print(f"Log emission matrix range: [{np.min(log_B):.2f}, {np.max(log_B):.2f}]")
        
        # Forward-backward passes
        alpha = hmm_model.forward(log_B, use_log=True)
        beta = hmm_model.backward(log_B, use_log=True)
        
        # Check for numerical issues in alpha/beta
        print(f"Alpha range: [{np.min(alpha[alpha > -np.inf]):.2f}, {np.max(alpha):.2f}]")
        print(f"Beta range: [{np.min(beta[beta > -np.inf]):.2f}, {np.max(beta):.2f}]")
        
        # Calculate gamma and xi
        gamma = hmm_model.compute_gamma(alpha, beta, use_log=True)
        xi = hmm_model.compute_xi(alpha, beta, log_B, use_log=True)
        
        # Check gamma and xi
        print(f"Gamma range: [{np.min(gamma):.6f}, {np.max(gamma):.6f}]")
        print(f"Gamma sum per frame: [{np.min(np.sum(gamma, axis=0)):.6f}, {np.max(np.sum(gamma, axis=0)):.6f}]")
        if np.any(xi > 0):
            print(f"Xi range: [{np.min(xi[xi > 0]):.6f}, {np.max(xi):.6f}]")
        
        # Accumulate for parameter updates
        all_gammas.append(gamma)
        all_xis.append(xi)
        
        # Calculate sequence log-likelihood
        seq_log_likelihood = np.logaddexp.reduce(alpha[:, -1])
        total_log_likelihood += seq_log_likelihood
        print(f"Sequence log-likelihood: {seq_log_likelihood:.2f}")
    
    return total_log_likelihood, all_gammas, all_xis

def compare_training(word="heed", n_states=8, n_iter=15):
    """Compare training progress with reference implementation."""
    print(f"Training comparison for word '{word}'")
    
    # Load and prepare data
    features = load_mfccs_by_word("feature_set", word)
    
    # Initialize your HMM
    hmm_model = HMM(n_states, 13, features, model_name=word)
    
    # Track progress over iterations
    log_likelihoods = []
    
    for iteration in range(n_iter):
        # Debug current iteration
        total_log_likelihood, gammas, xis = debug_training_iteration(hmm_model, features, iteration + 1)
        log_likelihoods.append(total_log_likelihood)
        
        # Calculate improvement
        if iteration > 0:
            improvement = total_log_likelihood - log_likelihoods[-2]
            print(f"\nIteration {iteration + 1:2d}  {total_log_likelihood:14.8f}   {improvement:14.8f}")
        else:
            print(f"\nIteration {iteration + 1:2d}  {total_log_likelihood:14.8f}   {'nan':>14}")
        
        # Update parameters
        # Convert accumulated statistics to the format your update methods expect
        aggregated_gamma = np.sum(np.concatenate([g for g in gammas], axis=1), axis=1, keepdims=True)
        aggregated_xi = np.sum([np.sum(x, axis=0) for x in xis], axis=0)
        
        # Update parameters
        hmm_model.update_A(aggregated_xi, aggregated_gamma)
        hmm_model.update_B(features, gammas)
        
        # Check parameter ranges after updates
        print("\nParameter ranges after update:")
        print(f"A matrix (excluding entry/exit): [{np.min(hmm_model.A[1:-1, 1:-1]):.6f}, {np.max(hmm_model.A[1:-1, 1:-1]):.6f}]")
        print(f"Means: [{np.min(hmm_model.B['mean']):.6f}, {np.max(hmm_model.B['mean']):.6f}]")
        print(f"Covariances: [{np.min(hmm_model.B['covariance']):.6f}, {np.max(hmm_model.B['covariance']):.6f}]")
    
    # Print final comparison with reference
    print("\nTraining Summary:")
    print("Your implementation:")
    print(f"Starting log-likelihood: {log_likelihoods[0]:.2f}")
    print(f"Final log-likelihood: {log_likelihoods[-1]:.2f}")
    print(f"Total improvement: {log_likelihoods[-1] - log_likelihoods[0]:.2f}")
    
    print("\nReference implementation:")
    print("Starting log-likelihood: -96816.39")
    print("Final log-likelihood: -60768.75")
    print("Total improvement: 36047.64")

if __name__ == "__main__":
    compare_training()