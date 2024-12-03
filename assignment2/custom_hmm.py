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
        self.mean = self.calculate_means(feature_set)
        self.variance = self.calculate_variance(feature_set, self.mean)
        # Add variance floor
        var_floor = 0.01 * np.mean(self.variance)
        self.variance = np.maximum(self.variance, var_floor)

        self.A = self.initialize_transitions(feature_set, self.num_states)
        self.B = {
            "mean": np.tile(self.mean[:, np.newaxis], (1, self.num_states)),
            "covariance": np.tile(self.variance[:, np.newaxis], (1, self.num_states)),
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
        self.print_transition_matrix()

        print("\nB (emission parameters):")
        self.print_emission_parameters()

    def print_transition_matrix(self, precision: int = 3) -> None:
        """
        Print the transition matrix using pandas DataFrame for better formatting.
        """
        n = self.A.shape[0]
        df = pd.DataFrame(
            self.A,
            columns=[f"S{i}" for i in range(n)],
            index=[f"S{i}" for i in range(n)],
        )

        # Replace zeros with dots for cleaner visualization
        df = df.replace(0, ".")
        print(df.round(precision))

    def print_emission_parameters(self, precision: int = 3) -> None:
        # Print means
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

    def compute_log_emission_matrix(self, features: np.ndarray) -> np.ndarray:
        emission_matrix = self.compute_emission_matrix(features)
        return np.log(emission_matrix)

    def forward(self, emission_matrix: np.ndarray, use_log=True) -> np.ndarray:
        """
        Compute forward probabilities α(t,j) including entry and exit states.

        Args:
            emission_matrix: Emission probabilities of shape (num_states, T)
                            Only includes real states (not entry/exit).
            use_log: Whether to compute probabilities in log space.
        Returns:
            alpha: Forward probabilities of shape (num_states + 2, T)
                Includes entry state (0) and exit state (num_states + 1).
        """
        T = emission_matrix.shape[1]  # Number of time steps

        if use_log:
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

        else:
            alpha = np.zeros((self.num_states + 2, T))

            # Initialize t=0
            alpha[0, 0] = 0
            alpha[1, 0] = self.A[0, 1] * emission_matrix[0, 0]

            # Forward recursion
            for t in range(1, T):
                alpha[0, t] = 0  # Entry state always 0 after t=0

                for j in range(1, self.num_states + 1):
                    from_prev = alpha[j - 1, t - 1] * self.A[j - 1, j]
                    self_loop = alpha[j, t - 1] * self.A[j, j]
                    alpha[j, t] = (from_prev + self_loop) * emission_matrix[j - 1, t]

                # Exit state (num_states + 1) - can only come from last real state

        return alpha

    def backward(self, emission_matrix: np.ndarray, use_log=True) -> np.ndarray:
        """
        Compute backward probabilities β(t,j) including entry and exit states.

        Args:
            emission_matrix: Emission probabilities of shape (num_states, T)
                            Only includes real states (not entry/exit).
            use_log: Whether to compute probabilities in log space.
        Returns:
            beta: Backward probabilities of shape (num_states + 2, T)
                Includes entry state (0) and exit state (num_states + 1).
        """
        T = emission_matrix.shape[1]  # Number of time steps

        if use_log:
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

        else:
            beta = np.zeros((self.num_states + 2, T))

            # Initialize t=T-1
            for j in range(self.num_states + 1):  # states 0 to 8
                beta[j, T - 1] = self.A[j, -1]
            beta[-1, T - 1] = 0  # Exit state stays at 0

            # Backward recursion
            for t in range(T - 2, -1, -1):  # Go from T-2 down to 0
                beta[-1, t] = 0  # Exit state stays at 0
                for j in range(self.num_states + 1):
                    if j == self.num_states:  # State 8
                        beta[j, t] = (
                            beta[j, t + 1]
                            * self.A[j, j]
                            * emission_matrix[j - 1, t + 1]
                        )
                    else:
                        self_loop = (
                            beta[j, t + 1] * self.A[j, j] * emission_matrix[j, t + 1]
                        )
                        to_next = (
                            beta[j + 1, t + 1]
                            * self.A[j, j + 1]
                            * emission_matrix[j, t + 1]
                        )
                        beta[j, t] = self_loop + to_next

        return beta

    def compute_gamma(
        self, alpha: np.ndarray, beta: np.ndarray, use_log=True
    ) -> np.ndarray:
        """
        Compute state occupation likelihood γ(t,j) for each state j and time t.

        Args:
            alpha: Forward probabilities of shape (num_states + 2, T)
            beta: Backward probabilities of shape (num_states + 2, T)
            use_log: Whether probabilities are in log space

        Returns:
            gamma: State occupation likelihoods of shape (num_states, T)
                  Only includes real states (not entry/exit)
        """
        if use_log:
            # Sum alpha and beta in log space
            log_gamma = alpha + beta
            # Compute normalization term for each time step
            log_norm = np.logaddexp.reduce(log_gamma, axis=0)
            # Normalize (subtract in log space = divide in normal space)
            log_gamma = log_gamma - log_norm
            # Convert back to normal space and return only real states
            gamma = np.exp(log_gamma[1:-1])
        else:
            gamma = alpha * beta
            gamma = gamma / np.sum(gamma, axis=0)
            gamma = gamma[1:-1]
        return gamma

    def compute_xi(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        emission_matrix: np.ndarray,
        use_log=True,
    ) -> np.ndarray:
        """Calculate the probability of transitioning between states at each time step."""
        T = alpha.shape[1]
        xi = np.zeros((T - 1, self.num_states, self.num_states))

        # Helper function to compute transition probability
        def compute_transition_prob(t, i, j):
            """Calculate probability of transitioning from state i to j at time t.
            xi(t,i,j) = [alpha(t,i) * a_ij * b_j(O_t+1) * beta(t+1,j)] / P(O|lambda)
            """
            if use_log:
                prob = (
                    alpha[i + 1, t]  # Being in state i
                    + np.log(self.A[i + 1, j + 1])  # Transitioning to state j
                    + emission_matrix[j, t + 1]  # Observing next symbol in state j
                    + beta[j + 1, t + 1]  # Future observations from state j
                    - np.logaddexp.reduce(alpha[:, -1])
                )  # Normalize by total probability
                return np.exp(prob)
            else:
                return (
                    alpha[i + 1, t]
                    * self.A[i + 1, j + 1]
                    * emission_matrix[j, t + 1]
                    * beta[j + 1, t + 1]
                ) / np.sum(alpha[:, -1])

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
        Update emission parameters (means and covariances) using multiple training sequences.

        Args:
            training_features: List of feature matrices, one per training sequence
            gamma_per_seq: List of gamma matrices corresponding to each training sequence
        """
        weighted_sum_features = np.zeros((self.num_obs, self.num_states))
        weighted_sum_gamma = np.zeros((self.num_states, 1))

        # First pass: Compute new means
        for features, gamma in zip(training_features, gamma_per_seq):
            weighted_sum_features += np.dot(features, gamma.T)
            weighted_sum_gamma += np.sum(gamma, axis=1, keepdims=True)

        # Update means
        self.B["mean"] = weighted_sum_features / (weighted_sum_gamma.T)

        # Initialize variance accumulator
        updated_variances = np.zeros_like(self.B["covariance"])

        # Second pass: Compute new variances using updated means
        for features, gamma in zip(training_features, gamma_per_seq):
            for j in range(self.num_states):
                diff = features - self.B["mean"][:, j : j + 1]
                weighted_sq_diff = np.sum(gamma[j] * (diff**2), axis=1)
                updated_variances[:, j] += weighted_sq_diff

        # Normalize variances by total gamma counts
        updated_variances /= weighted_sum_gamma.T

        # Apply variance flooring to prevent numerical issues
        var_floor = 0.01 * np.mean(updated_variances)
        self.B["covariance"] = np.maximum(updated_variances, var_floor)

        # Print some statistics for debugging
        print("\nEmission parameter update statistics:")
        print(f"Number of sequences processed: {len(training_features)}")
        print(f"Average gamma sum per state: {np.mean(weighted_sum_gamma):.3f}")
        print(
            f"Mean range: [{np.min(self.B['mean']):.3f}, {np.max(self.B['mean']):.3f}]"
        )
        print(
            f"Variance range: [{np.min(self.B['covariance']):.3f}, {np.max(self.B['covariance']):.3f}]"
        )

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
                log_B = self.compute_log_emission_matrix(features)
                alpha = self.forward(log_B, use_log=True)
                beta = self.backward(log_B, use_log=True)
                gamma = self.compute_gamma(alpha, beta, use_log=True)
                xi = self.compute_xi(alpha, beta, log_B, use_log=True)

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
