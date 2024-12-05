import numpy as np
import pandas as pd
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)


class HMM:
    def __init__(
        self,
        num_states: int,
        num_obs: int,
        feature_set: list[np.ndarray] = None,
        model_name: str = None,
        var_floor_factor: float = 0.001,
    ):
        assert num_states > 0, "Number of states must be greater than 0."
        assert num_obs > 0, "Number of observations must be greater than 0."
        self.model_name = model_name
        self.num_states = num_states
        self.num_obs = num_obs
        self.var_floor_factor = var_floor_factor
        self.total_states = num_states + 2

        self.pi = np.zeros(self.total_states)
        self.pi[0] = 1.0

        if feature_set is not None:
            assert all(
                feature.shape[0] == num_obs for feature in feature_set
            ), "All features must have the same dimension as the number of observations."
            self.init_parameters(feature_set)

    def init_parameters(self, feature_set: list[np.ndarray]) -> None:
        self.global_mean = self.calculate_means(feature_set)

        # Calculate full covariance then zero out off-diagonal elements
        self.global_covariance = self.calculate_covariance(
            feature_set, self.global_mean
        )
        self.global_covariance *= np.eye(self.num_obs)  # Zero out off-diagonal elements

        # Apply variance floor to diagonal
        var_floor = self.var_floor_factor * np.mean(np.diag(self.global_covariance))
        np.fill_diagonal(
            self.global_covariance,
            np.maximum(np.diag(self.global_covariance), var_floor),
        )

        self.A = self.initialize_transitions(feature_set, self.num_states)

        # Initialize B using global statistics
        means = np.tile(self.global_mean, (self.total_states, 1))

        # Initialize covariance matrices for each state
        covars = np.zeros((self.total_states, self.num_obs, self.num_obs))
        for i in range(self.total_states):
            covars[i] = self.global_covariance.copy()

        self.B = {"mean": means, "covariance": covars}

        assert self.B["mean"].shape == (self.total_states, self.num_obs)
        assert self.B["covariance"].shape == (
            self.total_states,
            self.num_obs,
            self.num_obs,
        )

    def calculate_means(self, feature_set: list[np.ndarray]) -> np.ndarray:
        """
        Calculate global mean and covariance (diagonal) from MFCC files in a directory.
        """
        sum = np.zeros(self.num_obs)
        count = 0
        for feature in feature_set:
            sum += np.sum(feature, axis=1)
            count += feature.shape[1]
        mean = sum / count
        return mean

    def calculate_covariance(
        self, feature_set: list[np.ndarray], mean: np.ndarray
    ) -> np.ndarray:
        """Calculate full covariance matrix of MFCC features across all frames"""
        covariance = np.zeros((self.num_obs, self.num_obs))
        count = 0
        for feature in feature_set:
            centered = feature - mean[:, np.newaxis]
            covariance += centered @ centered.T
            count += feature.shape[1]
        return covariance / count

    def initialize_transitions(
        self, feature_set: list[np.ndarray], num_states: int
    ) -> np.ndarray:
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

        A[-1, -1] = 1.0  # set exit state self-loop for easier computation
        return A

    def print_parameters(self):
        print("HMM Parameters:")
        print(f"\nN (states): {self.num_states}")
        print(f"\nM (observation dim): {self.num_obs}")
        print(f"\nπ (initial state distribution): {self.pi.round(3)}")
        print("\nA (transition matrix):")
        self.print_matrix(self.A, "Transition Matrix", col="To", idx="From")
        print("\nB (emission parameters):")
        self.print_emission_parameters()

    def print_emission_parameters(self, precision: int = 3) -> None:
        print("\nMeans (each column is a state, each row is an MFCC coefficient):")
        means_df = pd.DataFrame(
            self.B["mean"],
            columns=[f"State {i+1}" for i in range(self.num_states)],
            index=[f"MFCC {i+1}" for i in range(self.num_obs)],
        )
        print(means_df.round(precision))

        # Print covariances
        print("\nVariances (each column is a state, each row is an MFCC coefficient):")
        cov_df = pd.DataFrame(
            self.B["covariance"],
            columns=[f"State {i+1}" for i in range(self.num_states)],
            index=[f"MFCC {i+1}" for i in range(self.num_obs)],
        )
        print(cov_df.round(precision))

    def compute_emission_matrix(self, features: np.ndarray) -> np.ndarray:
        """
        Compute log emission probabilities for each state and time.
        Returns a TxN matrix where T is number of frames and N is number of states.
        """
        T = features.shape[1]
        log_emission_matrix = np.full((T, self.total_states), -np.inf)

        # Skip entry (0) and exit (-1) states as they're non-emitting
        for j in range(1, self.total_states - 1):
            diff = features - self.B["mean"][j, :, np.newaxis]
            try:
                # Handle full covariance calculation
                inv_cov = np.linalg.inv(self.B["covariance"][j])
                sign, logdet = np.linalg.slogdet(self.B["covariance"][j])
                assert (
                    sign > 0
                ), f"Non-positive definite covariance matrix for state {j}"

                # Compute for all time steps at once using matrix operations
                log_emission_matrix[:, j] = -0.5 * (
                    self.num_obs * np.log(2 * np.pi)
                    + logdet
                    + np.sum(diff.T @ inv_cov @ diff, axis=1)
                )
            except np.linalg.LinAlgError:
                logging.warning(
                    f"Singular covariance matrix for state {j}, using diagonal approximation"
                )
                diag_cov = np.diag(np.diag(self.B["covariance"][j]))
                log_emission_matrix[:, j] = -0.5 * np.sum(
                    diff**2 / np.diag(diag_cov)[:, np.newaxis]
                    + np.log(2 * np.pi * np.diag(diag_cov))[:, np.newaxis],
                    axis=0,
                )

        return log_emission_matrix

    def forward(self, emission_matrix: np.ndarray) -> tuple[np.ndarray, float]:
        T = emission_matrix.shape[0]
        alpha = np.full((T, self.total_states), -np.inf)

        # Initialize
        alpha[0, 0] = 0
        alpha[0, 1] = np.log(self.A[0, 1]) + emission_matrix[0, 1]

        # Forward pass
        for t in range(1, T):
            alpha[t, 0] = -np.inf
            for j in range(1, self.total_states):
                if j == 1:
                    alpha[t, j] = (
                        np.logaddexp(
                            alpha[t - 1, 0] + np.log(self.A[0, 1]),
                            alpha[t - 1, 1] + np.log(self.A[1, 1]),
                        )
                        + emission_matrix[t, j]
                    )
                elif j < self.total_states - 1:
                    alpha[t, j] = (
                        np.logaddexp(
                            alpha[t - 1, j - 1] + np.log(self.A[j - 1, j]),
                            alpha[t - 1, j] + np.log(self.A[j, j]),
                        )
                        + emission_matrix[t, j]
                    )
                else:
                    alpha[t, j] = alpha[t - 1, j - 1] + np.log(self.A[j - 1, j])

        # Global scale factor
        scale_factor = np.max(alpha)
        alpha -= scale_factor

        return alpha, scale_factor

    def backward(self, emission_matrix: np.ndarray, scale_factor: float) -> np.ndarray:
        T = emission_matrix.shape[0]
        beta = np.full((T, self.total_states), -np.inf)

        beta[-1, -1] = 0

        for t in range(T - 2, -1, -1):
            for i in range(self.total_states - 1):
                if i == 0:
                    beta[t, i] = (
                        np.log(self.A[i, 1])
                        + emission_matrix[t + 1, 1]
                        + beta[t + 1, 1]
                    )
                elif i < self.total_states - 2:
                    beta[t, i] = np.logaddexp(
                        np.log(self.A[i, i])
                        + emission_matrix[t + 1, i]
                        + beta[t + 1, i],
                        np.log(self.A[i, i + 1])
                        + emission_matrix[t + 1, i + 1]
                        + beta[t + 1, i + 1],
                    )
                else:
                    beta[t, i] = np.logaddexp(
                        np.log(self.A[i, i])
                        + emission_matrix[t + 1, i]
                        + beta[t + 1, i],
                        np.log(self.A[i, i + 1]) + beta[t + 1, i + 1],
                    )

        beta[:-1] -= scale_factor

        return beta

    def compute_gamma(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        Compute state occupation likelihood γ(t,j) in log space.
        """
        log_gamma = alpha + beta
        log_norm = np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        log_gamma = log_gamma - log_norm
        gamma = np.exp(log_gamma)

        return gamma

    def compute_xi(
        self, alpha: np.ndarray, beta: np.ndarray, emission_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the probability of transitioning between states at each time step.
        """
        T = alpha.shape[0]
        # Initialize with zeros instead of -inf
        xi = np.zeros((T - 1, self.total_states, self.total_states))
        log_likelihood = np.logaddexp.reduce(alpha[-1])

        for t in range(T - 1):
            # Entry state only transitions to first real state
            xi[t, 0, 1] = np.exp(
                alpha[t, 0]
                + np.log(self.A[0, 1])
                + emission_matrix[t + 1, 1]
                + beta[t + 1, 1]
                - log_likelihood
            )

            # Real states can only self-loop or go to next state
            for i in range(1, self.total_states - 1):
                if self.A[i, i] > 0:  # Self-loop
                    xi[t, i, i] = np.exp(
                        alpha[t, i]
                        + np.log(self.A[i, i])
                        + emission_matrix[t + 1, i]
                        + beta[t + 1, i]
                        - log_likelihood
                    )

                if i < self.total_states - 2:  # Forward transition
                    xi[t, i, i + 1] = np.exp(
                        alpha[t, i]
                        + np.log(self.A[i, i + 1])
                        + emission_matrix[t + 1, i + 1]
                        + beta[t + 1, i + 1]
                        - log_likelihood
                    )

            # Last real state to exit
            xi[t, -2, -1] = np.exp(
                alpha[t, -2]
                + np.log(self.A[-2, -1])
                + emission_matrix[t + 1, -1]
                + beta[t + 1, -1]
                - log_likelihood
            )

            # Exit state self-loop
            xi[t, -1, -1] = np.exp(
                alpha[t, -1]
                + np.log(self.A[-1, -1])
                + emission_matrix[t + 1, -1]
                + beta[t + 1, -1]
                - log_likelihood
            )

            # Normalize
            if np.sum(xi[t]) > 0:
                xi[t] /= np.sum(xi[t])

        return xi

    def print_matrix(
        self,
        matrix: np.ndarray,
        title: str,
        col="T",
        idx="State",
        start_idx=0,
        start_col=0,
    ) -> None:
        """
        Prints a given matrix with a formatted title, column headers, and row indices.

        Parameters:
            matrix (np.ndarray): The matrix to be printed.
            title (str): The title to display above the matrix.
        """
        if matrix.ndim == 2:
            print(f"\n{title}:")
            df = pd.DataFrame(
                matrix,
                columns=[f"{col} {i + start_col}" for i in range(matrix.shape[1])],
                index=[f"{idx} {i + start_idx}" for i in range(matrix.shape[0])],
            )
            print(df)
        else:
            logging.warning("Method only supports 2D matrices.")

    def update_A(self, aggregated_xi, aggregated_gamma) -> None:
        """
        Update transition probability matrix A using accumulated statistics.
        """
        self.A[0, 1] = 1.0

        # Update transitions for real states
        for i in range(1, self.total_states - 1):
            if aggregated_gamma[i] > 0:
                self.A[i, i] = aggregated_xi[i, i] / aggregated_gamma[i]
                self.A[i, i + 1] = 1.0 - self.A[i, i]

        # Exit state always self-loops
        self.A[-1, -1] = 1.0

    # def update_B(
    #     self, features_list: list[np.ndarray], gamma_per_seq: list[np.ndarray]
    # ) -> None:
    #     state_means = np.zeros((self.total_states, self.num_obs))
    #     state_covars = np.zeros((self.total_states, self.num_obs, self.num_obs))
    #     state_occupancy = np.zeros(self.total_states)

    #     # Update means first
    #     for features, gamma in zip(features_list, gamma_per_seq):
    #         for j in range(1, self.total_states - 1):
    #             state_means[j] += np.sum(gamma[:, j : j + 1] * features.T, axis=0)
    #             state_occupancy[j] += np.sum(gamma[:, j])

    #     for j in range(1, self.total_states - 1):
    #         if state_occupancy[j] > 0:
    #             state_means[j] /= state_occupancy[j]

    #     # Update covariance matrices
    #     for features, gamma in zip(features_list, gamma_per_seq):
    #         for j in range(1, self.total_states - 1):
    #             diff = features.T - state_means[j]
    #             for t in range(features.shape[1]):
    #                 state_covars[j] += gamma[t, j] * np.outer(diff[t], diff[t])

    #     # Normalize and apply variance floor
    #     var_floor = self.var_floor_factor * np.mean(np.diagonal(self.global_covariance))
    #     for j in range(1, self.total_states - 1):
    #         if state_occupancy[j] > 0:
    #             state_covars[j] /= state_occupancy[j]
    #             # Ensure symmetry
    #             state_covars[j] = (state_covars[j] + state_covars[j].T) / 2
    #             # Apply floor to diagonal
    #             diag_indices = np.diag_indices(self.num_obs)
    #             state_covars[j][diag_indices] = np.maximum(
    #                 state_covars[j][diag_indices], var_floor
    #             )

    #     self.B["mean"] = state_means
    #     self.B["covariance"] = state_covars

    # def baum_welch(
    #     self, features_list: list[np.ndarray], max_iter: int = 15, tol: float = 1e-4
    # ):
    #     """
    #     Train the HMM using the Baum-Welch algorithm on multiple sequences.
    #     Returns the log likelihood values for each iteration to monitor convergence.
    #     """
    #     print(f"\nTraining `{self.model_name}` HMM using Baum-Welch algorithm...")
    #     prev_log_likelihood = float("-inf")
    #     log_likelihood_history = []

    #     for iteration in range(max_iter):
    #         total_log_likelihood = 0

    #         # Initialize accumulators for transition updates
    #         aggregated_gamma = np.zeros(self.total_states)
    #         aggregated_xi = np.zeros((self.total_states, self.total_states))
    #         gamma_per_seq = []

    #         # E-Step across all sequences
    #         for features in features_list:
    #             # Forward-backward calculations
    #             emission_matrix = self.compute_emission_matrix(features)
    #             alpha, scale_factors = self.forward(emission_matrix)
    #             beta = self.backward(emission_matrix, scale_factors)
    #             gamma = self.compute_gamma(alpha, beta)
    #             xi = self.compute_xi(alpha, beta, emission_matrix)

    #             # Store gamma for B updates
    #             gamma_per_seq.append(gamma)

    #             # Accumulate statistics for A updates
    #             aggregated_gamma += np.sum(gamma[:-1], axis=0)  # Sum over time
    #             aggregated_xi += np.sum(xi, axis=0)  # Sum over time

    #             # Compute sequence log-likelihood
    #             seq_log_likelihood = np.logaddexp.reduce(alpha[-1])
    #             total_log_likelihood += seq_log_likelihood

    #         # Store log likelihood
    #         log_likelihood_history.append(total_log_likelihood)

    #         print(
    #             f"Iteration {iteration + 1}, Log-Likelihood: {total_log_likelihood:.2f}"
    #         )

    #         # Check convergence
    #         if abs(total_log_likelihood - prev_log_likelihood) < tol:
    #             print(f"Converged after {iteration + 1} iterations!")
    #             break

    #         prev_log_likelihood = total_log_likelihood

    #         # M-Step: Update model parameters
    #         self.update_A(aggregated_xi, aggregated_gamma)
    #         self.update_B(features_list, gamma_per_seq)

    #     print("Training complete!")
    #     return log_likelihood_history

    # def decode(self, features: np.ndarray) -> Tuple[List[int], float]:
    #     """
    #     Decode the most likely state sequence using the Viterbi algorithm.
    #     """
    #     T = features.shape[0]
    #     emission_matrix = self.compute_emission_matrix(features)

    #     V = np.full((T, self.total_states), -np.inf)
    #     backpointer = np.zeros((T, self.total_states), dtype=int)

    #     V[0, 0] = 0
    #     V[0, 1] = np.log(self.A[0, 1]) + emission_matrix[0, 1]

    #     for t in range(1, T):
    #         for j in range(1, self.total_states):
    #             if j == 1:
    #                 prev_states = [1]
    #                 if t == 1:
    #                     prev_states.append(0)
    #             elif j == self.total_states - 1:
    #                 if t >= self.num_states:
    #                     prev_states = [j - 1, j]
    #                 else:
    #                     continue
    #             else:
    #                 prev_states = [j - 1, j]

    #             best_score = -np.inf
    #             best_prev = None

    #             for i in prev_states:
    #                 score = V[t - 1, i] + np.log(self.A[i, j])
    #                 if score > best_score:
    #                     best_score = score
    #                     best_prev = i

    #             if best_prev is not None:
    #                 if j != 0 and j != self.total_states - 1:
    #                     V[t, j] = best_score + emission_matrix[t, j]
    #                 else:
    #                     V[t, j] = best_score
    #                 backpointer[t, j] = best_prev

    #     path = []
    #     curr_state = self.total_states - 1

    #     for t in range(T - 1, -1, -1):
    #         path.append(curr_state)
    #         curr_state = backpointer[t, curr_state]

    #     path.reverse()

    #     return float(V[T - 1, -1]), path
