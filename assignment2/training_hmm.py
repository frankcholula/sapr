import numpy as np
import initialise_hmm as init
from mfcc_extract import load_mfcc_class
from mfcc_extract import load_mfccs



def multivariate_gaussian(x, mean, covariance):
    """
    Calculate the multivariate Gaussian probability density for a given observation.
    
    Parameters:
        x (np.ndarray): Observation vector (K-dimensional).
        mean (np.ndarray): Mean vector (K-dimensional).
        covariance (np.ndarray): Diagonal covariance vector (K-dimensional).

    Returns:
        float: Probability density of the observation.
    """
    K = len(mean)  # Dimensionality
    diff = x - mean
    # Compute probability density
    #print(covariance.shape)
    #print(diff.shape)
    prob = (1 / np.sqrt((2 * np.pi) ** K * np.linalg.det(covariance))) * \
            np.exp(-(np.linalg.solve(covariance, diff).T.dot(diff)) / 2)
            #np.exp(-0.5 * np.sum((diff ** 2) / covariance))
    return prob



def forward_algorithm(observations, A, means, covariances):
    """
    Forward procedure to calculate forward likelihoods (alpha).
    """
    num_states = A.shape[0] - 2
    T = observations.shape[1]
    alpha = np.zeros((T, num_states))
    pi = A[0,1:-1]
    #print(pi)
    # Initialize
    scale_factor = 1e24
    alpha[0, :] = pi * multivariate_gaussian(observations[:, 0], means[0], covariances[0])
    # Recursion
    for t in range(1, T):
        for j in range(0, num_states):
            #print(A[1:-1, j])
            # alpha[t, j] = np.sum(alpha[t - 1, :] * A[1:-1, j]) * multivariate_gaussian(
            #     observations[:, t], means[j - 1], covariances[j - 1]
            # )
            for k in range(0,num_states):
                alpha[t,j] += alpha[t-1,k] * A[k+1,j+1]
            alpha[t,j] *= multivariate_gaussian(observations[:, t], means[j], covariances[j])*scale_factor
            #print(alpha[t,j])

    return alpha


def backward_algorithm(observations, A, means, covariances):
    """
    Backward procedure to calculate backward likelihoods (beta).
    """
    num_states = A.shape[0] - 2
    T = observations.shape[1]
    beta = np.zeros((T, num_states))
    scale_factor = 1e24
    # Initialize
    beta[T-1,:] = np.transpose(A[1:-1,-1])
    #print(beta[T-1,:])
    # Recursion
    for t in range(T - 2, -1, -1):
        for i in range(0, num_states):
            # beta[t, i] = np.sum(
            #     beta[t + 1, :] * A[i, :] * multivariate_gaussian(
            #         observations[:, t + 1], means, covariances
            #     )
            # )
            for j in range(0,num_states):
                beta[t,i] += A[i+1,j+1] * beta[t+1,j] * multivariate_gaussian(observations[:, t+1], means[j], covariances[j])*scale_factor
                
            #print(beta[t,i])

    return beta


def baum_welch(observations_list, model, max_iters=10):
    """
    Performs Baum-Welch re-estimation on the HMM parameters.

      Parameters:
        observations_list: list of np.ndarray
            List of observation sequences (each array is features x time).
        model: dict
            A dictionary containing HMM parameters:
                - "A": Transition matrix
                - "means": Means of Gaussian emissions
                - "covariances": Covariance matrices of Gaussian emissions
        max_iters: int
            Maximum number of iterations for re-estimation.

    Returns:
        updated_model: dict
            Updated HMM parameters after re-estimation.
    """
    A = model["A"]
    means = model["means"]
    covariances = model["covariances"]

    num_states = A.shape[0] - 2
    K = observations_list[0].shape[0]  # Dimensionality of feature space

    for iteration in range(max_iters):
        # Initialize accumulators for re-estimation
        A_accum = np.zeros_like(A)
        means_accum = np.zeros_like(means)
        covariances_accum = np.zeros_like(covariances)
        gamma_sum_accum = np.zeros(num_states)

        for observations in observations_list:
            T = observations.shape[1]  # Number of time steps

            # Forward and backward probabilities
            alpha = forward_algorithm(observations, A, means, covariances)
            beta = backward_algorithm(observations, A, means, covariances)

            # Calculate gamma (state occupancy probabilities)
            gamma = np.zeros((T, num_states))
            for t in range(T):
                gamma[t, :] = alpha[t, :] * beta[t, :]
                gamma[t, :] /= np.sum(gamma[t, :])  # Normalize

            # Accumulate gamma sums for re-estimation
            gamma_sum_accum += np.sum(gamma, axis=0)

            # Calculate xi (state transition probabilities)
            xi = np.zeros((T - 1, num_states, num_states))
            for t in range(T - 1):
                for i in range(num_states):
                    for j in range(num_states):
                        xi[t, i, j] = alpha[t-1, i] * A[i + 1, j + 1] * \
                                      multivariate_gaussian(observations[:, t], means[j], covariances[j]) * \
                                      beta[t, j]
                print(np.sum(xi[t, :, :]))
                xi[t, :, :] /= np.sum(xi[t, :, :])  # Normalize

            # Accumulate transition matrix updates
            for i in range(num_states):
                for j in range(num_states):
                    A_accum[i + 1, j + 1] += np.sum(xi[:, i, j])

            # Accumulate mean and covariance updates
            for j in range(num_states):
                gamma_sum = np.sum(gamma[:, j])
                weighted_sum = np.sum(gamma[:, j][:, np.newaxis] * observations.T, axis=0)
                means_accum[j] += weighted_sum
                diff = observations.T - means[j]
                covariances_accum[j] += np.sum(
                    gamma[:, j][:, np.newaxis, np.newaxis] *
                    np.einsum('ij,ik->ijk', diff, diff),
                    axis=0
                )

        # Normalize transition matrix
        for i in range(num_states):
            A_accum[i + 1, 1:-1] /= np.sum(A_accum[i + 1, 1:-1])

        # Normalize means and covariances
        for j in range(num_states):
            means[j] = means_accum[j] / gamma_sum_accum[j]
            covariances[j] = covariances_accum[j] / gamma_sum_accum[j]

        A = A_accum  # Update transition matrix

    # Return the updated model
    updated_model = {
        "A": A,
        "means": means,
        "covariances": covariances
    }
    return updated_model





if __name__ == "__main__":
    TRAINING_FOLDER = "feature_set"
    class_num = 1
    num_states = 8
    class_one_trial = load_mfcc_class(TRAINING_FOLDER, class_num)
    
    feature_set = load_mfccs(TRAINING_FOLDER)
   

    global_mean = init.calculate_means(feature_set)
    global_covariance_matrix = init.create_covariance_matrix(init.calculate_variance(feature_set,global_mean))
    self_loop_prob = init.intialize_transition_prob(feature_set, num_states)
    A = init.initialize_transitions(feature_set, num_states)

    model = {
        "A": A.copy(),
        "means": np.tile(global_mean, (num_states, 1)),  # num_states x 13
        "covariances": np.array([global_covariance_matrix] * num_states),  # num_states x 13
    }
    observations = feature_set[0]
    #print(model["A"])
    #print(model["means"].shape)
    #print(model["covariances"][0].shape)
    new_alpha = forward_algorithm(observations,model["A"],model["means"],model["covariances"])
    new_beta = backward_algorithm(observations,model["A"],model["means"],model["covariances"])
    # print(new_beta[:, 0])
    # print(new_beta[:, 7])

    # TODO: use this as a test case to check forward and backward are working. 
    print(multivariate_gaussian(observations[:, 0], model["means"][0], model["covariances"][0]) * new_beta[0, 0])
    print(model["A"][-2, -1] * new_alpha[-1, 7])
    
    #print(new_a)
    new_model = baum_welch(class_one_trial,model,1)
    # print(new_model["A"])
    # print(new_model["means"])
    # print(new_model["covariances"])
# observations, A, means, covariances, pi)
# states = [0, 1, 0, 2, 1]  # Hidden states
# observations = feature_set[0]  # Observed sequence (indices into emission matrix)
# start_prob = [1, 0, 0, 0, 0, 0, 0, 0] # Initial probabilities
# emission_prob = [0, 0, 0, 0, 0, 0, 0, 1-self_loop_prob]  # Emission matrix
# trans_prob = A  # Transition matrix

# new_a = forward_algorithm(observations,states,start_prob,trans_prob,emission_prob)




# start_prob, trans_prob, emission_prob = baum_welch(
#     observations, states, start_prob, trans_prob, emission_prob, num_iterations=10
# )

# print("Updated Start Probabilities:", start_prob)
# print("Updated Transition Probabilities:", trans_prob)
# print("Updated Emission Probabilities:", emission_prob)