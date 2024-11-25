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


def initialize_transitions(
    feature_set: list[np.ndarray], num_states: int
) -> np.ndarray:
    # Calculate total frames from all audio files
    total_frames = sum(feature.shape[1] for feature in feature_set)

    # Calculate average frames per state
    avg_frames_per_state = total_frames / (len(feature_set) * num_states)

    # Calculate self-loop probability using the formula
    aii = np.exp(-1 / (avg_frames_per_state - 1))

    # Fill transition matrix
    A = np.zeros((num_states + 2, num_states + 2))
    for i in range(num_states + 2):
        if i == 0:
            A[i, i] = 0
            A[i, i + 1] = 1
        if 0 < i < num_states - 1 + 2:
            A[i, i] = aii
            A[i, i + 1] = 1 - aii
    return A


def print_transition_matrix(A: np.ndarray, precision: int = 3) -> None:
    """
    Print a prettified version of the transition matrix.
    """
    n = A.shape[0]
    print("\nTransition Matrix:")
    print("─" * (n * 8 + 1))

    # Print column headers
    print("    │", end="")
    for j in range(n):
        print(f"  S{j:d}  ", end="")
    print("\n" + "────┼" + "───────" * n)

    # Print matrix rows
    for i in range(n):
        print(f" S{i:d} │", end="")
        for j in range(n):
            val = A[i, j]
            if val == 0:
                print("  .   ", end="")
            else:
                print(f" {val:.{precision}f}", end="")
        print()

    print("─" * (n * 8 + 1))


if __name__ == "__main__":
    feature_set = load_mfccs("feature_set")
    mean = calculate_means(feature_set)
    variance = calculate_variance(feature_set, mean)
    cov_matrix = create_covariance_matrix(variance)
    trans_matrix = initialize_transitions(feature_set, 8)
    print_transition_matrix(trans_matrix)
