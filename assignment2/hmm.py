import numpy as np
import pandas as pd
from mfcc_extract import load_mfccs
import logging

logging.basicConfig(level=logging.INFO)


class HMM:
    def __init__(
        self, num_states: int, num_obs: int, feature_set: list[np.ndarray] = None
    ):
        assert num_states > 0, "Number of states must be greater than 0."
        assert num_obs > 0, "Number of observations must be greater than 0."

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
                alpha[-1, t] = alpha[-2, t - 1] * self.A[-2, -1]

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

    def update_A(self, xi: np.ndarray, gamma: np.ndarray) -> None:
        """
        Update transition probability matrix A using Baum-Welch formula:
        aᵢⱼ = (sum over t of ξ(t,i,j)) / (sum over t of γ(t,i))
        """
        # Entry state transitions to first state with probability 1
        self.A[0, 1] = 1.0

        # Update transitions for real states
        for i in range(self.num_states):
            expected_transitions_from_i = np.sum(gamma[i, :-1])  # Exclude last time step

            if expected_transitions_from_i > 0:
                # Self-transition and forward-transition
                if i < self.num_states - 1:
                    expected_self_transitions = np.sum(xi[:, i, i])
                    expected_forward_transitions = np.sum(xi[:, i, i+1])
                    self.A[i+1, i+1] = expected_self_transitions / expected_transitions_from_i
                    self.A[i+1, i+2] = expected_forward_transitions / expected_transitions_from_i

                # Transition to exit state
                if i == self.num_states - 1:  # For the last real state (State 8 in your case)
                    expected_exit_transitions = gamma[i, -1]  # Probability of being in state i at the last time step
                    total_transitions_from_i = expected_transitions_from_i + expected_exit_transitions
                    self.A[i+1, i+1] = (np.sum(xi[:, i, i]) / total_transitions_from_i)  # Normalize self-transition
                    self.A[i+1, i+2] = (expected_exit_transitions / total_transitions_from_i)  # Normalize exit transition
    def update_B(self, features: np.ndarray, gamma: np.ndarray) -> None:
        """
        Update emission probability matrix B (Gaussian parameters for each state).
        
        Args:
            features: Observed feature matrix of shape (num_features, T).
            gamma: State occupation probabilities of shape (N, T).
        """
        num_features, T = features.shape  # F, T
        N = self.num_states  # Number of states

        # Initialize updated means and variances
        updated_means = np.zeros((num_features, N))
        updated_variances = np.zeros((num_features, N))

        # Update mean and variance for each state
        for j in range(N):
            # Weighted sum for mean
            gamma_sum = np.sum(gamma[j])  # Sum of gamma over all time steps for state j
            if gamma_sum > 0:
                updated_means[:, j] = np.sum(gamma[j] * features, axis=1) / gamma_sum

                # Weighted sum for variance
                deviations = features - updated_means[:, j:j+1]  # Broadcast mean across time
                updated_variances[:, j] = np.sum(gamma[j] * (deviations ** 2), axis=1) / gamma_sum
            else:
                # Handle empty states (rare in well-trained models)
                updated_means[:, j] = np.zeros(num_features)
                updated_variances[:, j] = np.ones(num_features)  # Prevent division by zero

        # Update model parameters
        self.B["mean"] = updated_means
        self.B["covariance"] = updated_variances

    
    def print_matrix(
        self, matrix: np.ndarray, title: str, col="T", idx="State", start_idx=0, start_col=0
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
    
    def baum_welch(self, features_list: list[np.ndarray], max_iter: int = 100, tol: float = 1e-4):
        """
        Train the HMM using the Baum-Welch algorithm on multiple sequences.

        Args:
            features_list: List of observed feature matrices, each of shape (num_features, T).
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance for log-likelihood improvement.
        """
        prev_log_likelihood = float('-inf')  # Initialize log-likelihood

        for iteration in range(max_iter):
            total_log_likelihood = 0  # Sum log-likelihood over all sequences

            # Aggregated gamma and xi over all sequences
            aggregated_gamma = None
            aggregated_xi = None

            for features in features_list:
                T = features.shape[1]

                # === E-Step for a single sequence ===
                log_B = self.compute_log_emission_matrix(features)  # Log emission probabilities
                alpha = self.forward(log_B, use_log=True)
                beta = self.backward(log_B, use_log=True)
                gamma = self.compute_gamma(alpha, beta, use_log=True)
                xi = self.compute_xi(alpha, beta, log_B, use_log=True)

                # Accumulate gamma and xi
                if aggregated_gamma is None:
                    aggregated_gamma = gamma
                    aggregated_xi = xi
                else:
                    aggregated_gamma += gamma
                    aggregated_xi += xi

                # Compute log-likelihood for the current sequence
                log_likelihood = np.logaddexp.reduce(alpha[:, -1])  # Log-sum-exp for sequence
                total_log_likelihood += log_likelihood

            print(f"Iteration {iteration}, Total Log-Likelihood: {total_log_likelihood}")

            # Check for convergence
            if abs(total_log_likelihood - prev_log_likelihood) < tol:
                print("Converged!")
                break
            prev_log_likelihood = total_log_likelihood

            # === M-Step ===
            self.update_A(aggregated_xi, aggregated_gamma)  # Update transition probabilities
            self.update_B(np.hstack(features_list), aggregated_gamma)  # Update emission probabilities

        print("Baum-Welch training completed.")
