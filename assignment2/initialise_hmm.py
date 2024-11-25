import numpy as np
import pandas as pd
from mfcc_extract import load_mfccs


def calculate_means(feature_set: list[np.ndarray]) -> np.ndarray:
    """
    Calculate global mean and covariance (diagonal) from MFCC files in a directory.
    """
    sum = np.zeros(13)
    count = 0
    for feature in feature_set:
        sum += np.sum(feature, axis=1)
        count += feature.shape[1]
    mean = sum / count
    return mean


def create_covariance_matrix(variance: np.ndarray) -> np.ndarray:
    """Create a diagonal covariance matrix from the variance vector."""
    return np.diag(variance)


def calculate_variance(feature_set: list[np.ndarray], mean: np.ndarray) -> np.ndarray:
    """Calculate variance of MFCC features across all frames"""
    ssd = np.zeros(13)
    count = 0
    for feature in feature_set:
        ssd += np.sum((feature - mean[:, np.newaxis]) ** 2, axis=1)
        count += feature.shape[1]
    variance = ssd / count
    return variance


def initialize_transitions(feature_set: list[np.ndarray], num_states: int) -> np.ndarray:
    total_frames = sum(feature.shape[1] for feature in feature_set)
    avg_frames_per_state = total_frames / (len(feature_set) * num_states)

    # self-loop probability    
    aii = np.exp(-1 / (avg_frames_per_state - 1))
    aij = 1 - aii 
    
    # Create transition matrix (including entry and exit states)
    total_states = num_states + 2
    A = np.zeros((total_states, total_states))
    
    # Entry state (index 0)
    A[0, 1] = 1.0
    
    for i in range(1, num_states + 1):
        A[i, i] = aii
        A[i, i + 1] = aij
    return A


def print_transition_matrix(A: np.ndarray, precision: int = 3) -> None:
    """
    Print the transition matrix using pandas DataFrame for better formatting.
    """
    n = A.shape[0]
    df = pd.DataFrame(
        A, 
        columns=[f'S{i}' for i in range(n)],
        index=[f'S{i}' for i in range(n)]
    )
    
    # Replace zeros with dots for cleaner visualization
    df = df.replace(0, '.')
    
    print("\nTransition Matrix:")
    print(df.round(precision))


# class HMM:
#     def __init__(self, num_states: int, num_)

if __name__ == "__main__":
    feature_set = load_mfccs("feature_set")
    mean = calculate_means(feature_set)
    variance = calculate_variance(feature_set, mean)
    cov_matrix = create_covariance_matrix(variance)
    trans_matrix = initialize_transitions(feature_set, 8)
    print_transition_matrix(trans_matrix)
