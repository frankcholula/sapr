import numpy as np
import librosa
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from mfcc_extract import extract_mfcc

# Viterbi Algorithm for decoding the most likely sequence of hidden states
def viterbi_algorithm(observations, A, B, pi):
    N = A.shape[0]  # Number of states
    T = observations.shape[0]  # Number of observations
    log_A = np.log(A)  # Logarithm of transition probabilities
    log_B = np.log(B)  # Logarithm of emission probabilities

    # Initialize dynamic programming tables
    alpha = np.zeros((N, T))  # Holds the max log probability for each state at each time step
    backpointer = np.zeros((N, T), dtype=int)  # Backpointer for the most likely state sequence

    # Initialization step
    alpha[:, 0] = np.log(pi) + log_B[:, observations[0]]  # First observation

    # Recursion step
    for t in range(1, T):
        for i in range(N):
            trans_probs = alpha[:, t-1] + log_A[:, i]  # Transition probabilities from previous state
            best_prev_state = np.argmax(trans_probs)  # Best previous state
            alpha[i, t] = trans_probs[best_prev_state] + log_B[i, observations[t]]
            backpointer[i, t] = best_prev_state

    # Termination step: find the best final state
    best_last_state = np.argmax(alpha[:, T-1])

    # Backtrack to find the most likely state sequence
    state_sequence = np.zeros(T, dtype=int)
    state_sequence[T-1] = best_last_state
    for t in range(T-2, -1, -1):
        state_sequence[t] = backpointer[state_sequence[t+1], t+1]

    return state_sequence, alpha[state_sequence, range(T)].sum()  # Return the best sequence and its log-likelihood

# Function to evaluate the model on a test set
def evaluate_model(test_files, model_params, n_mfcc=13):
    # model_params is a dictionary with 'A', 'B', and 'pi' for each word model
    true_labels = []
    predicted_labels = []
    
    for test_file in test_files:
        word = test_file.split('.')[0]  # Assume filename format 'word.wav'
        true_labels.append(word)

        # Extract MFCC features from the audio file
        features = extract_mfcc(test_file, n_mfcc=n_mfcc)
        observations = quantize_features(features, model_params['means'], model_params['covariances'])

        # Run Viterbi algorithm for each word model
        best_log_likelihood = -np.inf
        predicted_label = None
        for label, params in model_params.items():
            A, B, pi = params['A'], params['B'], params['pi']
            _, log_likelihood = viterbi_algorithm(observations, A, B, pi)
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                predicted_label = label

        predicted_labels.append(predicted_label)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model_params.keys(), yticklabels=model_params.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Calculate accuracy
    accuracy = np.sum(np.array(true_labels) == np.array(predicted_labels)) / len(true_labels) * 100
    print(f"Recognition Accuracy: {accuracy:.2f}%")
    return accuracy, cm

# Quantize MFCC features to the nearest Gaussian mean based on model parameters
def quantize_features(features, means, covariances):
    quantized_features = []
    for frame in features:
        distances = [np.linalg.norm(frame - mean) for mean in means]
        quantized_features.append(np.argmin(distances))  # Assign the frame to the nearest mean
    return np.array(quantized_features)
    

if __name__ == "__main__":
    test_files = []  # Replace with actual test file paths
    model_params = {}

    # Run evaluation
    evaluate_model(test_files, model_params)
