import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


class HMM:
    def __init__(
        self,
        num_states: int,
        num_obs: int,
        feature_set: list[np.ndarray] = None,
        model_name: str = None,
    ):
        assert num_states > 0, "Number of states must be greater than 0."
        assert num_obs > 0, "Number of observations must be greater than 0."
        self.model_name = model_name
        self.num_states = num_states
        self.num_obs = num_obs

        self.pi = np.zeros(num_states + 2)
        self.pi[0] = 1.0

        if feature_set is not None:
            assert all(
                feature.shape[0] == num_obs for feature in feature_set
            ), "All features must have the same dimension as the number of observations."
            self.init_parameters(feature_set)

    def init_parameters(self, feature_set: list[np.ndarray]) -> None:
        global_mean = self.calculate_means(feature_set)
        global_variance = self.calculate_variance(feature_set, global_mean)
        # Add variance floor
        var_floor = 0.01 * np.mean(global_variance)
        global_variance = np.maximum(global_variance, var_floor)

        # Initialize A matrix
        self.A = self.initialize_transitions(feature_set, self.num_states)
        self.B = {
            "mean": np.tile(global_mean[:, np.newaxis], (1, self.num_states)),
            "covariance": np.tile(global_variance[:, np.newaxis], (1, self.num_states)),
        }
        assert self.B["mean"].shape == (self.num_obs, self.num_states)
        assert self.B["covariance"].shape == (self.num_obs, self.num_states)

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

    def calculate_variance(
        self, feature_set: list[np.ndarray], mean: np.ndarray
    ) -> np.ndarray:
        """Calculate variance of MFCC features across all frames"""
        ssd = np.zeros(self.num_obs)
        count = 0
        for feature in feature_set:
            ssd += np.sum((feature - mean[:, np.newaxis]) ** 2, axis=1)
            count += feature.shape[1]
        variance = ssd / count
        return variance

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
        return A

    def print_parameters(self):
        print("HMM Parameters:")
        print(f"\nN (states): {self.num_states}")
        print(f"\nM (observation dim): {self.num_obs}")

        print(f"\nπ (initial state distribution): {self.pi.round(3)}")

        print("\nA (transition matrix):")
        self.print_matrix(self.A, "Transition Matrix", col="State", idx="State")

        print("\nB (emission parameters):")
        self.print_matrix(self.B["mean"], "Mean", idx="Feature", start_idx=1)

    def compute_emission_matrix(self, features: np.ndarray) -> np.ndarray:
        """
        Compute the emission matrix B using a 13-dimensional multivariate Gaussian.
        """
        T = features.shape[1]
        emission_matrix = np.zeros((self.num_states, T))

        for j in range(self.num_states):
            diff = features - self.B["mean"][:, j : j + 1]
            mahalanobis_squared = diff**2 / self.B["covariance"][:, j : j + 1]
            exponent = -0.5 * np.sum(mahalanobis_squared, axis=0)
            determinant = np.prod(self.B["covariance"][:, j])
            normalization = np.sqrt((2 * np.pi) ** self.num_obs * determinant)
            emission_matrix[j] = np.exp(exponent) / normalization
        return emission_matrix

    def compute_emission_matrix(self, features: np.ndarray) -> np.ndarray:
        """
        For a multivariate Gaussian, the log probability is:
        log(p(x)) = -0.5 * (d*log(2π) + log(|Σ|) + (x-μ)ᵀΣ⁻¹(x-μ))

        Args:
            features: Matrix of shape (num_features, T) containing MFCC features

        Returns:
            log_emission_matrix: Matrix of shape (num_states, T) containing log probabilities
        """
        T = features.shape[1]
        log_emission_matrix = np.zeros((self.num_states, T))
        const_term = -0.5 * self.num_obs * np.log(2 * np.pi)

        for j in range(self.num_states):
            # Get state-specific parameters
            state_mean = self.B["mean"][:, j]
            state_covariance = self.B["covariance"][:, j]

            # Compute log determinant term: -0.5 * log(|Σ|)
            # For diagonal covariance, this is sum of logs
            log_det = -0.5 * np.sum(np.log(state_covariance))

            # Compute Mahalanobis distance term: -0.5 * (x-μ)ᵀΣ⁻¹(x-μ)
            # For diagonal covariance, this simplifies to element-wise operations
            diff = (
                features - state_mean[:, np.newaxis]
            )  # Broadcasting to match features shape
            mahalanobis_squared = diff**2 / state_covariance[:, np.newaxis]
            mahalanobis_term = -0.5 * np.sum(mahalanobis_squared, axis=0)

            # Combine all terms for this state
            log_emission_matrix[j] = const_term + log_det + mahalanobis_term

        return log_emission_matrix

    def forward(self, emission_matrix: np.ndarray) -> np.ndarray:
        """
        Compute forward probabilities α(t,j) including entry and exit states.

        Args:
            emission_matrix: Emission probabilities of shape (num_states, T)
                            Only includes real states (not entry/exit).
        Returns:
            alpha: Forward probabilities of shape (num_states + 2, T)
                Includes entry state (0) and exit state (num_states + 1).
        """
        T = emission_matrix.shape[1]  # Number of time steps

        alpha = np.full((self.num_states + 2, T), -np.inf)

        # Initialize t=0
        alpha[0, 0] = -np.inf
        alpha[1, 0] = np.log(self.A[0, 1]) + emission_matrix[0, 0]

        # Forward recursion
        for t in range(1, T):
            alpha[0, t] = -np.inf  # Entry state always -inf after t=0

            for j in range(1, self.num_states + 1):
                from_prev = alpha[j - 1, t - 1] + np.log(self.A[j - 1, j])
                self_loop = alpha[j, t - 1] + np.log(self.A[j, j])
                alpha[j, t] = (
                    np.logaddexp(from_prev, self_loop) + emission_matrix[j - 1, t]
                )

            # Exit state (num_states + 1) - can only come from last real state
            alpha[-1, t] = alpha[-2, t - 1] + np.log(self.A[-2, -1])

        return alpha

    def backward(self, emission_matrix: np.ndarray) -> np.ndarray:
        """
        Compute backward probabilities β(t,j) including entry and exit states.

        Args:
            emission_matrix: Emission probabilities of shape (num_states, T)
                            Only includes real states (not entry/exit).
        Returns:
            beta: Backward probabilities of shape (num_states + 2, T)
                Includes entry state (0) and exit state (num_states + 1).
        """
        T = emission_matrix.shape[1]  # Number of time steps

        beta = np.full((self.num_states + 2, T), -np.inf)

        # Initialize t=T-1
        for j in range(self.num_states + 1):  # states 0 to 8
            if self.A[j, -1] == 0:
                beta[j, T - 1] = -np.inf  # -inf if no transition to exit state
            else:
                beta[j, T - 1] = np.log(self.A[j, -1])
        beta[-1, T - 1] = -np.inf  # Exit state stays at -inf

        # Backward recursion
        for t in range(T - 2, -1, -1):  # Go from T-2 down to 0
            beta[-1, t] = -np.inf  # Exit state stays at -inf
            for j in range(self.num_states + 1):
                if j == self.num_states:  # State 8
                    if self.A[j, j] == 0:
                        beta[j, t] = -np.inf  # -inf if no self-loop
                    else:
                        beta[j, t] = (
                            beta[j, t + 1]
                            + np.log(self.A[j, j])
                            + emission_matrix[j - 1, t + 1]
                        )
                else:
                    if self.A[j, j] == 0:
                        self_loop = -np.inf
                    else:
                        self_loop = (
                            beta[j, t + 1]
                            + np.log(self.A[j, j])
                            + emission_matrix[j, t + 1]
                        )
                    if self.A[j, j + 1] == 0:
                        to_next = -np.inf
                    else:
                        to_next = (
                            beta[j + 1, t + 1]
                            + np.log(self.A[j, j + 1])
                            + emission_matrix[j, t + 1]
                        )
                    beta[j, t] = np.logaddexp(self_loop, to_next)

        return beta

    def compute_gamma(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        Compute state occupation likelihood γ(t,j) for each state j and time t.

        Args:
            alpha: Forward probabilities of shape (num_states + 2, T)
            beta: Backward probabilities of shape (num_states + 2, T)
        Returns:
            gamma: State occupation likelihoods of shape (num_states, T)
                  Only includes real states (not entry/exit)
        """
        # Sum alpha and beta in log space
        log_gamma = alpha + beta
        # Compute normalization term for each time step
        log_norm = np.logaddexp.reduce(log_gamma, axis=0)
        # Normalize (subtract in log space = divide in normal space)
        log_gamma = log_gamma - log_norm
        # Convert back to normal space and return only real states
        gamma = np.exp(log_gamma[1:-1])
        return gamma

    def compute_xi(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        emission_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate the probability of transitioning between states at each time step."""
        T = alpha.shape[1]
        xi = np.zeros((T - 1, self.num_states, self.num_states))

        # Helper function to compute transition probability
        def compute_transition_prob(t, i, j):
            """Calculate probability of transitioning from state i to j at time t.
            xi(t,i,j) = [alpha(t,i) * a_ij * b_j(O_t+1) * beta(t+1,j)] / P(O|lambda)
            """
            prob = (
                alpha[i + 1, t]  # Being in state i
                + np.log(self.A[i + 1, j + 1])  # Transitioning to state j
                + emission_matrix[j, t + 1]  # Observing next symbol in state j
                + beta[j + 1, t + 1]  # Future observations from state j
                - np.logaddexp.reduce(alpha[:, -1])
            )  # Normalize by total probability
            return np.exp(prob)

        # Calculate probabilities for each time step
        for t in range(T - 1):
            for i in range(self.num_states):
                # Only compute allowed transitions in left-right HMM:
                # 1. Self-transition (stay in same state)
                xi[t, i, i] = compute_transition_prob(t, i, i)

                # 2. Forward transition (move to next state)
                if i < self.num_states - 1:
                    xi[t, i, i + 1] = compute_transition_prob(t, i, i + 1)

            # Normalize probabilities at each time step
            if np.sum(xi[t]) > 0:  # Avoid division by zero
                xi[t] = xi[t] / np.sum(xi[t])

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

    def update_A(self, aggregated_xi: np.ndarray, aggregated_gamma: np.ndarray) -> None:
        """
        Update transition probability matrix A using accumulated statistics from all sequences.

        Args:
            aggregated_xi: Sum of transition counts across all sequences, shape (num_states, num_states)
            aggregated_gamma: Sum of state occupation counts across all sequences, shape (num_states, 1)
        """
        # Entry state always transitions to first state with probability 1
        self.A[0, 1] = 1.0

        # Update transitions for real states (states 1 to N)
        for i in range(self.num_states):
            total_transitions_from_i = aggregated_gamma[i, 0]

            if total_transitions_from_i > 0:
                # Self-transition probability
                self.A[i + 1, i + 1] = aggregated_xi[i, i] / total_transitions_from_i

                # Forward transition probability (if not the last state)
                if i < self.num_states - 1:
                    self.A[i + 1, i + 2] = (
                        aggregated_xi[i, i + 1] / total_transitions_from_i
                    )

                # For the last real state, also update transition to exit state
                if i == self.num_states - 1:
                    # The probability of transitioning to exit state is what remains
                    # after self-transition probability is assigned
                    self.A[i + 1, i + 2] = 1.0 - self.A[i + 1, i + 1]

    def update_B(
        self, training_features: list[np.ndarray], gamma_per_seq: list[np.ndarray]
    ) -> None:
        """
        Update emission parameters (means and covariances) for each state using
        the weighted statistics from all training sequences.

        Args:
            training_features: List of feature matrices, each of shape (num_features, T)
            gamma_per_seq: List of gamma matrices, each of shape (num_states, T),
                        containing state occupation probabilities
        """
        # Initialize accumulators for means
        weighted_sum_features = np.zeros((self.num_obs, self.num_states))
        weighted_sum_gamma = np.zeros((self.num_states, 1))

        # First pass: Compute new means for each state
        for features, gamma in zip(training_features, gamma_per_seq):
            # features: (num_obs, T), gamma: (num_states, T)
            # weighted_sum_features will be (num_obs, num_states)
            weighted_sum_features += np.dot(features, gamma.T)
            weighted_sum_gamma += np.sum(gamma, axis=1, keepdims=True)

        # Avoid division by zero by adding small epsilon
        eps = 1e-10
        # Update means - shape: (num_obs, num_states)
        self.B["mean"] = weighted_sum_features / (weighted_sum_gamma.T + eps)

        # Initialize variance accumulator
        weighted_sq_diff_sum = np.zeros((self.num_obs, self.num_states))

        # Second pass: Compute new variances using updated means
        for features, gamma in zip(training_features, gamma_per_seq):
            for j in range(self.num_states):
                # Compute squared differences from state-specific mean
                diff = features - self.B["mean"][:, j : j + 1]
                sq_diff = diff**2

                # Weight the squared differences by gamma and sum
                weighted_sq_diff_sum[:, j] += np.sum(gamma[j] * sq_diff, axis=1)

        # Update variances with normalization and flooring
        self.B["covariance"] = weighted_sq_diff_sum / (weighted_sum_gamma.T + eps)

        # Apply variance flooring to prevent numerical issues
        # Use a state-specific floor based on the mean variance for that state
        for j in range(self.num_states):
            var_floor = 0.01 * np.mean(self.B["covariance"][:, j])
            self.B["covariance"][:, j] = np.maximum(
                self.B["covariance"][:, j], var_floor
            )

        # Print statistics for debugging
        print("\nEmission parameter update statistics:")
        print(f"Number of sequences processed: {len(training_features)}")
        print(f"Average gamma sum per state: {np.mean(weighted_sum_gamma):.3f}")
        print(
            f"Mean range: [{np.min(self.B['mean']):.3f}, {np.max(self.B['mean']):.3f}]"
        )
        print(
            f"Variance range: [{np.min(self.B['covariance']):.3f}, {np.max(self.B['covariance']):.3f}]"
        )

        # Additional checks for numerical stability
        if not np.all(np.isfinite(self.B["mean"])):
            print("Warning: Non-finite values detected in means!")
        if not np.all(np.isfinite(self.B["covariance"])):
            print("Warning: Non-finite values detected in variances!")
        if np.any(self.B["covariance"] <= 0):
            print("Warning: Non-positive variances detected!")

    def baum_welch(
        self, features_list: list[np.ndarray], max_iter: int = 15, tol: float = 1e-4
    ):
        """
        Train the HMM using the Baum-Welch algorithm on multiple sequences.
        Returns the log likelihood values for each iteration to monitor convergence.

        Args:
            features_list: List of observed feature matrices, each of shape (num_features, T).
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance for log-likelihood improvement.

        Returns:
            list[float]: Log likelihood values for each iteration
        """
        print(f"\nTraining `{self.model_name}` HMM using Baum-Welch algorithm...")
        prev_log_likelihood = float("-inf")

        # Create list to store log likelihood for each iteration
        log_likelihood_history = []

        for iteration in range(max_iter):
            total_log_likelihood = 0

            # Initialize accumulators for transition updates
            aggregated_gamma = np.zeros((self.num_states, 1))
            aggregated_xi = np.zeros((self.num_states, self.num_states))

            # Store gamma values for each sequence
            gamma_per_seq = []

            # E-Step across all sequences
            for seq_idx, features in enumerate(features_list):
                # Compute forward-backward statistics
                log_B = self.compute_emission_matrix(features)
                alpha = self.forward(log_B)
                beta = self.backward(log_B)
                gamma = self.compute_gamma(alpha, beta)
                xi = self.compute_xi(alpha, beta, log_B)

                gamma_per_seq.append(gamma)

                # Accumulate statistics for transition updates
                aggregated_gamma += np.sum(gamma, axis=1, keepdims=True)
                aggregated_xi += np.sum(xi, axis=0)

                # Compute log-likelihood for this sequence
                log_likelihood = np.logaddexp.reduce(alpha[:, -1])
                total_log_likelihood += log_likelihood

                if len(features_list) > 10 and seq_idx % 10 == 0:
                    print(f"Processed sequence {seq_idx + 1}/{len(features_list)}...")

            # Store the log likelihood for this iteration
            log_likelihood_history.append(total_log_likelihood)

            print(
                f"Iteration {iteration + 1}, Total Log-Likelihood: {total_log_likelihood}"
            )

            # Check for convergence
            if abs(total_log_likelihood - prev_log_likelihood) < tol:
                print(f"Converged after {iteration + 1} iterations!")
                break

            prev_log_likelihood = total_log_likelihood

            # M-Step: Update model parameters
            self.update_A(aggregated_xi, aggregated_gamma)
            self.update_B(features_list, gamma_per_seq)

        print("Baum-Welch training completed.")

        # Print summary of likelihood changes
        total_improvement = log_likelihood_history[-1] - log_likelihood_history[0]
        print("\nTraining summary:")
        print(f"Initial log-likelihood: {log_likelihood_history[0]:.2f}")
        print(f"Final log-likelihood: {log_likelihood_history[-1]:.2f}")
        print(f"Total improvement: {total_improvement:.2f}")
        print(log_likelihood_history)
        return log_likelihood_history
