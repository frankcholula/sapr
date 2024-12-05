import numpy as np
import logging
import pandas as pd
import pickle
import csv
from pathlib import Path
#import hmm as init
from mfcc_extract import load_mfccs_by_word
from mfcc_extract import load_mfccs
from custom_hmm import HMM



def multivariate_gaussian(x, mean, covariance):
    """
    Calculate the multivariate Gaussian probability density for a given observation.
    
    """
    K = len(mean)  # Dimensionality
    diff = x - mean
    # Compute probability density
    #print(covariance.shape)
    #print(diff.shape)
    #print(diff)
    try:
        prob = (1 / np.sqrt((2 * np.pi) ** K * np.linalg.det(covariance))) * \
               np.exp(-(np.linalg.solve(covariance, diff).T.dot(diff)) / 2)
    except np.linalg.LinAlgError as e:
        #print("Error inverting covariance matrix:", e)
        prob = 0.0
    return prob


def forward_algorithm(observations, model):
    """
    Forward procedure to calculate forward likelihoods (alpha).
    """
    A = model["A"]
    num_states = A.shape[0] - 2
    T = observations.shape[1]
    alpha = np.zeros((T, num_states+2))
    pi = A[0,1:-1]
    # Initialize
    means = model["means"]
    covariances = model["covariances"]
    scale_factor = 1e24
    alpha[0, 1] = A[0,1] * multivariate_gaussian(observations[:, 0], means[1], covariances[1])
    # Recursion
    for t in range(1, T):
        alpha[t,0] = 0
        for j in range(1, num_states+1):
            #print(A[1:-1, j])
            # alpha[t, j] = np.sum(alpha[t - 1, :] * A[1:-1, j]) * multivariate_gaussian(
            #     observations[:, t], means[j - 1], covariances[j - 1]
            # )
            for k in range(1,num_states+1):
                alpha[t,j] += alpha[t-1,k] * A[k,j]
            
            alpha[t,j] *= multivariate_gaussian(observations[:, t], means[j], covariances[j])*scale_factor
        alpha[t,-1] = alpha[t-1,-2] * A[-2,-1]
            #print(alpha[t,j])

    return alpha


def backward_algorithm(observations, model):
    """
    Backward procedure to calculate backward likelihoods (beta).
    """
    A = model["A"]
    num_states = A.shape[0] - 2
    T = observations.shape[1]
    beta = np.zeros((T, num_states+2))
    means = model["means"]
    covariances = model["covariances"]
    scale_factor = 1e24
    # Initialize
    beta[T-1,:] = np.transpose(A[:,-1])
    beta[T-1,-1] = 0
    #print(beta[T-1,:])
    # Recursion
    for t in range(T - 2, -1, -1):
        beta[t,-1] = 0
        for i in range(0, num_states+1):

            beta[t,-1] = 0
            for j in range(1,num_states+1):
                beta[t,i] += A[i,j] * beta[t+1,j] *\
                    multivariate_gaussian(observations[:, t+1], means[j], covariances[j])*scale_factor
                
            #print(beta[t,i])

    return beta

def compute_gamma(alpha, beta, Pr):
    """
    Computes gamma (state occupancy probabilities).

    """
    gamma = (alpha * beta) / Pr
    return gamma

def compute_xi(observations, alpha, beta, model, Pr):
    """
    Computes xi (state transition probabilities).

   
    """
    A = model["A"]
    means = model["means"]
    covariances = model["covariances"]
    T = observations.shape[1]
    num_states = A.shape[0] - 2
    #A_accum = np.zeros_like(A)

    xi = np.zeros((T, num_states+2, num_states+2))
    for t in range(T - 1):
        xi[t, 0, 1] = (alpha[t, 1] * A[0, 1] * \
                                multivariate_gaussian(observations[:, t+1], means[1], covariances[1]) * \
                                beta[t+1, 1])/Pr

        for i in range(1,num_states+1):
                #Self loop
                xi[t, i, i] = (alpha[t, i] * A[i, i] * \
                                multivariate_gaussian(observations[:, t+1], means[i], covariances[i])* \
                                beta[t+1, i])/Pr
                # Forward transition
                xi[t, i, i+1] = (alpha[t, i] * A[i, i+1] * \
                                multivariate_gaussian(observations[:, t+1], means[i+1], covariances[i+1])* \
                                beta[t+1, i+1])/Pr
        #Last real state exit
        xi[t, -2, -1] = (alpha[t-1, -2] * A[-2, -1] * \
                                multivariate_gaussian(observations[:, t], means[-1], covariances[-1])* \
                                beta[t+1, -1])/Pr

        # Exit state self-loop
        xi[t,-1,-1] = (alpha[t-1, -1] * A[-1, -1] * \
                                multivariate_gaussian(observations[:, t], means[-1], covariances[-1])* \
                                beta[t+1, -1])/Pr

        xi[t, :, :] #/= np.sum(xi[t, :, :])  # Normalize
    #print_matrix(xi)
    # Accumulate transition matrix updates(Xij)
    #print_matrix(xi[-1])
    #exit()
        # for i in range(num_states+2):
        #     for j in range(num_states+2):
        #         A_accum[i, j] += np.sum(xi[:, i, j])
    return xi
   
def update_transition_matrix(A_accum, gamma_sum_accum, num_states):
    """
    Updates the transition matrix using accumulated values.

    """
    A_accum[0,1] = 1
    for i in range(1,num_states + 1):
        A_accum[i, :] /= gamma_sum_accum[i]
        A_accum[i, :] *= 1e24
    A_accum[-2,-1] = 1 - A_accum[-2,-2]
    A_accum[-1,-1] = 1
    return A_accum

def update_means_and_covariances(observations_list, gamma_per_seq, num_states, means, covariances, covariance_floor):
    """
    Updates means and covariances using accumulated values.

    
    """
    means_accum = np.zeros_like(means)
    covariances_accum = np.zeros_like(covariances)
    state_occupancy = np.zeros(num_states + 2)
    
    #compute new means
    for features, gamma in zip(observations_list, gamma_per_seq):
            # Sum weighted observations for each state (exclude entry/exit)
            for j in range(1, num_states + 1):
                means_accum[j] += np.sum(gamma[:, j : j + 1] * features.T, axis=0)
                state_occupancy[j] += np.sum(gamma[:, j])

    # Normalize means by state occupancy
    for j in range(1, num_states + 1):
        if state_occupancy[j] > 0:
            means_accum[j] /= state_occupancy[j]

    # Second pass: compute new variances
    for features, gamma in zip(observations_list, gamma_per_seq):
        for j in range(1, num_states + 1):  # Iterate over each state
            T, D = features.shape[1], features.shape[0]  # T: time_steps, D: features
            for t in range(T):  # Loop over time steps
                diff = features[:, t] - means[j]  # Difference vector (D,)
                outer_product = np.outer(diff, diff)  # Outer product (D, D)
                covariances_accum[j] += gamma[t, j] * outer_product  # Weighted and accumulate


    # Normalize variances and apply floor
    for j in range(1, num_states + 1):
        if state_occupancy[j] > 0:
            covariances_accum[j] /= state_occupancy[j]

    # Apply variance floor
    var_floor = 0.001 * covariance_floor
    for i in range(1, num_states + 1):
        for j in range(13):
            covariances_accum[i][j,j] = np.maximum(covariances_accum[i][j,j], var_floor[j,j])

    # Update model parameters
    return means_accum, covariances_accum

def baum_welch(observations_list, model):
    """
    Performs Baum-Welch re-estimation on the HMM parameters.

    
    """
    A = model["A"]
    means = model["means"]
    covariances = model["covariances"]
    num_states = A.shape[0] - 2
    covariance_floor = model["covariance_floor"]

    # Initialize accumulators
    A_accum = np.zeros_like(A)
    gamma_sum_accum = np.zeros(num_states + 2)
    gamma_per_seq = []
    i = 0
    for observations in observations_list:
        T = observations.shape[1]

        # Forward and backward probabilities
        alpha = forward_algorithm(observations, model)
        beta = backward_algorithm(observations, model)
        Pr = model["A"][-2, -1] * alpha[-1, -2]
        #print(np.sum(alpha[-1, :]))
        #print(model["A"][-2, -1] * alpha[-1, -2])

        # Compute gamma and xi
        gamma = compute_gamma(alpha, beta, Pr)
        # Store gamma for B updates
        gamma_per_seq.append(gamma)

        xi = compute_xi(observations, alpha, beta, model, Pr)
        #xi2 = compute_xi2(observations, alpha, beta, model, Pr)
        #xi = compute_xifrank(observations, alpha, beta, model, Pr)
        #print(xi)
        #i += 1
        #if i == 20:
            #print_matrix(np.sum(xi, axis=0))
            #print_matrix(np.sum(xi2, axis=0))
        #
        # Accumulate transition matrix updates
        A_accum += np.sum(xi, axis=0)
        # i+= 1
        # if i == 20:
        #     print_matrix(A_accum); exit()
        gamma_sum_accum += np.sum(gamma, axis=0)

    # Update transition matrix
    #print_matrix(A_accum)
    A = update_transition_matrix(A_accum, gamma_sum_accum, num_states)
    #print_matrix(A); exit()
    #print_matrix(A); exit()
    # Update means and covariances
    means, covariances = update_means_and_covariances(
        observations_list, gamma_per_seq, num_states, means, covariances, covariance_floor,
    )

    # Return the updated model
    new_model = {"A": A, "means": means, "covariances": covariances, "covariance_floor":covariance_floor}
    return new_model



def viterbi_algorithm(observations, model):
    """
    Viterbi algorithm to find the most likely sequence of states given the observations.
    
    """
    A = model["A"]
    means = model["means"]
    covariances = model["covariances"]
    num_states = A.shape[0] - 2
    T = observations.shape[1]  # Number of time steps
    delta = np.zeros((T, num_states+2))  # Maximum cumulative likelihoods
    psi = np.zeros((T, num_states+2), dtype=int)  # Backpointer table
    scale_factor = 1e21

    # Initialize
    delta[0, 1] = A[0,1] * multivariate_gaussian(observations[:, 0], means[1], covariances[1]) #* scale_factor
    psi[0, :] = 0

    # Recursion
    for t in range(1, T):
        delta[t,0] = 0
        for j in range(1,num_states+1):
            max_value = -np.inf
            max_state = -1
            for i in range(1,num_states+1):
                value = delta[t - 1, i] * A[i, j]
                if value > max_value:
                    max_value = value
                    max_state = i
            delta[t, j] = max_value * multivariate_gaussian(observations[:, t], means[j], covariances[j]) * scale_factor
            psi[t, j] = max_state

    # Finalize
    P_star = delta[-1, -2]*A[-2, -1]
    #X_star = np.zeros(T, dtype=int)
    #X_star[-1] = np.argmax(delta[-1, :])
    

    # Trace back
    #for t in range(T - 2, -1, -1):
        #X_star[t] = psi[t + 1, X_star[t + 1]]

    return  P_star

def print_matrix(matrix):
    """
    Prints a 2D matrix so that it is aligned
    """
    # Determine the maximum width of any element for alignment
    col_widths = [max(len(str(item)) for item in col) for col in zip(*matrix)]
    
    # Print each row, aligning the columns
    for row in matrix:
        formatted_row = " | ".join(f"{str(item):<{col_widths[i]}}" for i, item in enumerate(row))
        print(formatted_row)
    print("\n")

def evaluate_model(features, models, vocabs):
    """
    Evaluate models on the dataset and compute accuracy and average likelihood.

    """
    total_log_likelihood = 0
    correct_predictions = 0
    total_predictions = 0
    
    for word in vocabs:
        for single_observation in features[word]:
            log_probs = {}
            
            # Score each model for the observation
            for candidate_word in vocabs:
                log_prob= viterbi_algorithm(single_observation, models[candidate_word])
                log_probs[candidate_word] = log_prob
            
            # Find the best-scoring model
            predicted_word = max(log_probs, key=log_probs.get)
            total_log_likelihood += log_probs[word]  # Add true model's log likelihood
            
            # Check if the prediction is correct
            if predicted_word == word:
                correct_predictions += 1
            total_predictions += 1
    
    # Compute metrics
    accuracy = correct_predictions / total_predictions * 100
    avg_log_likelihood = total_log_likelihood / total_predictions
    
    return accuracy, avg_log_likelihood


def train_hmm():
    vocabs = [
        "heed",
        "hid",
        "head",
        "had",
        "hard",
        "hud",
        "hod",
        "hoard",
        "hood",
        "whod",
        "heard",
    ]
    feature_set = load_mfccs("feature_set")
    features = {word: load_mfccs_by_word("feature_set", word) for word in vocabs}
    test_features = {word: load_mfccs_by_word("eval_feature_set", word) for word in vocabs}
    # total_features_length = sum(len(features[word]) for word in vocabs)
    # assert total_features_length == len(feature_set)
    hmms = {word: HMM(8, 13, feature_set, model_name=word) for word in vocabs}
    num_states = hmms[vocabs[0]].num_states
    total_states = hmms[vocabs[0]].total_states
    
    # getting models from HMM class initialization
    hmmmodels = {}
    for word in vocabs:
        
        model = {
            "A": hmms[word].A,
            "means": hmms[word].B["mean"],  # num_states x 13
            "covariances": np.array([np.diag(hmms[word].B["covariance"][0])] * total_states),  # num_states x 13
            "covariance_floor":np.diag(hmms[word].B["covariance"][0])
        }
        hmmmodels[word] = model


    # for epoch in range(15):
    #     print(f"Epoch {epoch + 1}")
    #     word = "heed"
    #     print(f"Training HMM for word: {word}")
    #     hmmmodels[word] = baum_welch(features[word], hmmmodels[word])
    #     print_matrix(hmmmodels[word]["A"])
    #     print(hmmmodels[word]["means"])
    #     print_matrix(hmmmodels[word]["covariances"][3])
    # exit()

    # CSV file to save accuracy and log-likelihood
    csv_file = "training_metrics.csv"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["Epoch", "Train Accuracy", "Train Avg Likelihood", "Test Accuracy", "Test Avg Likelihood"])
        #Training loop for Baum-Welch
        for epoch in range(15):  # Train for 10 epochs or adjust as needed
            print(f"Epoch {epoch + 1}")
            
            # Train each word's HMM
            for word in vocabs:
                print(f"Training HMM for word: {word}")
                hmmmodels[word] = baum_welch(features[word], hmmmodels[word])
            
            # Evaluate models on training set
            train_accuracy, train_avg_likelihood = evaluate_model(features, hmmmodels, vocabs)
            
            # Evaluate models on test set
            test_accuracy, test_avg_likelihood = evaluate_model(test_features, hmmmodels, vocabs)
    
            
            # Print metrics for this epoch
            print(f"Epoch {epoch + 1} - Train Accuracy: {train_accuracy:.2f}%, Train Avg Likelihood: {train_avg_likelihood:.2f}")
            print(f"Epoch {epoch + 1} - Test Accuracy: {test_accuracy:.2f}%, Test Avg Likelihood: {test_avg_likelihood:.2f}")
            
            # Save metrics to CSV
            writer.writerow([epoch + 1, train_accuracy, train_avg_likelihood, test_accuracy, test_avg_likelihood])
    
    print(f"Training metrics saved to {csv_file}")
    return hmmmodels









    #print(mat["A"].shape for mat in hmmmodels["heed"])
    #print(hmmmodels["heed"]["covariances"][0])
    model = hmmmodels["heed"]
    testing_feature = features["heed"]
    observations = testing_feature[7]

    
    #Test forward and backward implementation
    #print(model.A)
    #print(model.B["mean"].shape)
    #print(model.B["covariance"].shape)
    #new_alpha = forward_algorithm(observations,model)
    #new_beta = backward_algorithm(observations,model)
    #print(new_alpha)
    #print(new_beta)
    # Test for forward and backward compatible
    #print(multivariate_gaussian(observations[:, 0], model["means"][1], model["covariances"][1]) * new_beta[0, 1])
    #print(model["A"][-2, -1] * new_alpha[-1, -2])
    
    new_model = baum_welch(testing_feature,model)

    print(viterbi_algorithm(testing_feature[0],new_model))


    # print("\n baum welch ran without errors \n")
    # print_matrix(model["A"])
    # print_matrix(new_model["A"])

    # print(model["means"][4])
    # print(new_model["means"][4])

    # new_model2 = baum_welch(testing_feature,new_model)
    # print("\n baum welch ran without errors \n")
    # print_matrix(new_model2["A"])
    # print(new_model2["means"][4])
    # for cov in new_model2["covariances"]:
    #     print_matrix(cov)
    #print_matrix(model["A"])
    #print_matrix(hmms["heed"].A)

    #print(viterbi_algorithm(observations,model.A,model.B))
    #HMM.print_parameters(new_model)
    
    # for word, hmm in hmms.items():
    #     hmm.baum_welch(features[word], 2)


    #for word in vocabs:
     #   print(f"Training HMM for {word}...")
      #  hmms[word] = baum_welch(features[word], hmms[word])
      #  print(f"Completed training for {word}")



if __name__ == "__main__":
    models = train_hmm()
    save_path = Path("trained_models")
    save_path.mkdir(parents=True, exist_ok=True)

    # Save each model to a separate file
    for word, model in models.items():
        file_path = save_path / f"{word}_hmm.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved model for word '{word}' to {file_path}")
#     