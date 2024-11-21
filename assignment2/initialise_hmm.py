import numpy as np
import pandas as pd
import os
from mfcc_extract import load_mfccs


def calculate_means(feature_set: list[np.ndarray]) -> np.ndarray:
    """
    Calculate global mean and covariance (diagonal) from MFCC files in a directory.

    Parameters:
        directory_path (str): Path to the directory containing MFCC files (.npy).

    Returns:
        mean (np.ndarray): Global mean (13-dimensional).
        covariance (np.ndarray): Diagonal covariance matrix (13x13).
    """
    sum = np.zeros(13)
    count = 0
    for feature in feature_set:
        sum += np.sum(feature, axis=1)
        count += feature.shape[1]
    mean = sum / count
    return mean

def calculate_variance(feature_set: list[np.ndarray]) -> np.ndarray:
    calculate_means(feature_set)
    ssd = np.zeros(13)
    counts = 0
    for feature in feature_set:
        print(len(feature))
        # for window in feature:
            # print(window.shape)
        # for window in feature[1]:
            # ssd += (window - mean) ** 2
            # counts += 1
    # variance = ssd / counts
    # return variance
feature_set = load_mfccs("feature_set")
print(feature_set[0].shape)
mean = calculate_means(feature_set)
variance = calculate_variance(feature_set)

# global_mean, global_variance = calculate_global_stats("sapr-main/assignment2/feature_set")

# print(f"Global mean: {global_mean}")
# print(f"Global variance: {global_variance}")

# #---------------------------------

# class HMM:
#     def __init__(self, num_states, num_features, global_mean, global_variance):
#         """
#         Initialize a prototype HMM with flat-start parameters.

#         Parameters:
#             num_states: Number of states in the HMM.
#             num_features: Number of MFCC coefficients.
#             global_mean: Global mean vector for MFCC features.
#             global_variance: Global variance vector for MFCC features.
#         """
#         self.num_states = num_states
#         self.num_features = num_features

#         # Initialize transition probabilities (uniform flat start)
#         self.A = np.full((num_states, num_states), 1 / num_states)

#         # Initialize state probabilities (uniform flat start)
#         self.pi = np.full(num_states, 1 / num_states)

#         # Initialize Gaussian emission probabilities (global mean and variance)
#         self.mean = np.tile(global_mean, (num_states, 1))
#         self.variance = np.tile(global_variance, (num_states, 1))

# num_states = 8  # Number of states for each HMM
# num_features = 13  # Dimensionality of MFCCs
# word_hmm = HMM(num_states, num_features, global_mean, global_variance)

# print(f"Transition probabilities:\n{word_hmm.A}")
# print(f"Emission mean for each state:\n{word_hmm.mean}")
# print(f"Emission variance for each state:\n{word_hmm.variance}")

# #---------------------------------------

# def initialize_hmms(vocabulary, num_states, num_features, global_mean, global_variance):
#     """
#     Initialize HMMs for a list of words in the vocabulary.

#     Parameters:
#         vocabulary: List of word labels.
#         num_states: Number of states for each HMM.
#         num_features: Number of MFCC coefficients.
#         global_mean: Global mean vector for MFCC features.
#         global_variance: Global variance vector for MFCC features.

#     Returns:
#         hmms: Dictionary of word -> HMM.
#     """
#     hmms = {}
#     for word in vocabulary:
#         hmms[word] = HMM(num_states, num_features, global_mean, global_variance)
#     return hmms


# vocabulary = ["heed", "hid", "head", "had", "hard", "hud", "hod", "hoard", "hood", "who'd", "heard"]
# hmms = initialize_hmms(vocabulary, num_states=5, num_features=13, global_mean=global_mean, global_variance=global_variance)

# for word, hmm in hmms.items():
#     print(f"HMM for word '{word}':")
#     print(f"Transition probabilities:\n{hmm.A}")
