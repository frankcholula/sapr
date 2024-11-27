import numpy as np
import pandas as pd
from mfcc_extract import load_mfccs


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
                            Only includes real states (not entry/exit)
        Returns:
            alpha: Forward probabilities of shape (num_states + 2, T)
                    Includes entry state (0) and exit state (num_states + 1)
        """
        T = emission_matrix.shape[1]  # Number of time steps
        if use_log:
            alpha = np.full((self.num_states + 2, T), -np.inf)
        else:
            alpha = np.zeros((self.num_states + 2, T))

        # Initialize t=0
        if use_log:
            alpha[0, 0] = -np.inf
            alpha[1, 0] = np.log(self.A[0, 1]) + emission_matrix[0, 0]
        else:
            # We start in entry state (0) and can only transition to first real state (1)
            # Entry state probability is 0 since we don't start there at t=0
            alpha[0, 0] = 0
            # First real state gets probability from entry transition * emission
            alpha[1, 0] = self.A[0, 1] * emission_matrix[0, 0]
            # All other states have 0 probability at t=0
            # (already handled by numpy zeros initialization)

        # Forward recursion
        for t in range(1, T):
            # Entry state (0) - always 0 probability after t=0
            if use_log:
                alpha[0, t] = -np.inf
            else:
                alpha[0, t] = 0

            # Real states (1 to 8)
            for j in range(1, self.num_states + 1):
                if use_log:
                    # Log domain: use log sum exp for adding probabilities
                    from_prev = alpha[j - 1, t - 1] + np.log(self.A[j - 1, j])
                    self_loop = alpha[j, t - 1] + np.log(self.A[j, j])
                    # logaddexp handles adding probabilities in log domain
                    alpha[j, t] = (
                        np.logaddexp(from_prev, self_loop) + emission_matrix[j - 1, t]
                    )
                else:
                    # Two possibilities:
                    # 1. Come from previous state j-1 through forward transition
                    # 2. Stay in same state j through self-loop
                    from_prev = alpha[j - 1, t - 1] * self.A[j - 1, j]
                    self_loop = alpha[j, t - 1] * self.A[j, j]
                    # Total probability = (prev + self_loop) * emission
                    # Note: emission_matrix[j-1] because emission matrix is 0-based
                    alpha[j, t] = (from_prev + self_loop) * emission_matrix[j - 1, t]

            # Exit state (9) - can only come from last real state (8)
            if use_log:
                alpha[-1, t] = alpha[-2, t - 1] + np.log(self.A[-2, -1])
            else:
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
                beta[j, T-1] = np.log(self.A[j, -1])
            beta[-1, T-1] = -np.inf  # Exit state stays at -inf

            # Backward recursion
            for t in range(T-2, -1, -1):  # Go from T-2 down to 0
                beta[-1, t] = -np.inf  # Exit state stays at -inf
                for j in range(self.num_states + 1):
                    if j == self.num_states:  # State 8
                        beta[j, t] = beta[j, t+1] + np.log(self.A[j, j]) + emission_matrix[j-1, t+1]
                    else:
                        self_loop = beta[j, t+1] + np.log(self.A[j, j]) + emission_matrix[j, t+1]
                        to_next = beta[j+1, t+1] + np.log(self.A[j, j+1]) + emission_matrix[j, t+1]
                        beta[j, t] = np.logaddexp(self_loop, to_next)

        else:
            beta = np.zeros((self.num_states + 2, T))

            # Initialize t=T-1
            for j in range(self.num_states + 1):  # states 0 to 8
                beta[j, T-1] = self.A[j, -1]
            beta[-1, T-1] = 0  # Exit state stays at 0

            # Backward recursion
            for t in range(T-2, -1, -1):  # Go from T-2 down to 0
                beta[-1, t] = 0  # Exit state stays at 0
                for j in range(self.num_states + 1):
                    if j == self.num_states:  # State 8
                        beta[j, t] = beta[j, t+1] * self.A[j, j] * emission_matrix[j-1, t+1]
                    else:
                        self_loop = beta[j, t+1] * self.A[j, j] * emission_matrix[j, t+1]
                        to_next = beta[j+1, t+1] * self.A[j, j+1] * emission_matrix[j, t+1]
                        beta[j, t] = self_loop + to_next

        return beta
    def print_matrix(self, matrix: np.ndarray, title: str) -> None:
        """
        Prints a given matrix with a formatted title, column headers, and row indices.
        
        Parameters:
            matrix (np.ndarray): The matrix to be printed.
            title (str): The title to display above the matrix.
        """
        print(f"\n{title}:")
        df = pd.DataFrame(
            matrix,
            columns=[f"T{i+1}" for i in range(matrix.shape[1])],
            index=[f"State {i+1}" for i in range(matrix.shape[0])],
        )
        print(df)


if __name__ == "__main__":
    feature_set = load_mfccs("feature_set")
    hmm = HMM(8, 13, feature_set)
    log_emission_matrix = hmm.compute_log_emission_matrix(feature_set[0])
    hmm.print_matrix(log_emission_matrix, "Log Emission Matrix")
    hmm.print_transition_matrix()
    alpha_log = hmm.forward(log_emission_matrix, use_log=True)
    beta_log = hmm.backward(log_emission_matrix, use_log=True)
    hmm.print_matrix(alpha_log, "Log Forward Matrix")
    hmm.print_matrix(beta_log, "Log Backward Matrix")
