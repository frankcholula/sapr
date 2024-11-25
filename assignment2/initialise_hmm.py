import numpy as np
import pandas as pd
from mfcc_extract import load_mfccs


class HMM:
    def __init__(self, num_states: int, num_obs: int, feature_set: list[np.ndarray]=None):
        self.num_states = num_states
        self.num_obs = num_obs
        
        self.pi = np.zeros(num_states + 2)
        self.pi[0] = 1.0

        if feature_set is not None:
            self.init_parameters(feature_set)

    def init_parameters(self, feature_set: list[np.ndarray]) -> None:
        self.mean = self.calculate_means(feature_set)
        self.variance = self.calculate_variance(feature_set, self.mean)
        self.A = self.initialize_transitions(feature_set, self.num_states)      
        self.B = {
            'mean': self.mean,
            'covariance': self.create_covariance_matrix(self.variance)
        }  

    
    def calculate_means(self, feature_set: list[np.ndarray]) -> np.ndarray:
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


    def calculate_variance(self, feature_set: list[np.ndarray], mean: np.ndarray) -> np.ndarray:
        """Calculate variance of MFCC features across all frames"""
        ssd = np.zeros(13)
        count = 0
        for feature in feature_set:
            ssd += np.sum((feature - mean[:, np.newaxis]) ** 2, axis=1)
            count += feature.shape[1]
        variance = ssd / count
        return variance

    def create_covariance_matrix(self, variance: np.ndarray) -> np.ndarray:
        """Create a diagonal covariance matrix from the variance vector."""
        return np.diag(variance)


    def initialize_transitions(self, feature_set: list[np.ndarray], num_states: int) -> np.ndarray:
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
        print(f"N (states): {self.num_states}")
        print(f"M (observation dim): {self.num_obs}")
        
        print("π (initial state distribution):")
        print(self.pi.round(3))
        
        print("A (transition matrix):")
        self.print_transition_matrix()
        
        print("B (emission parameters):")
        print(f"Mean shape: {self.B['mean'].shape}")
        print(f"Covariance shape: {self.B['covariance'].shape}")
        
    def print_transition_matrix(self, precision: int = 3) -> None:
        """
        Print the transition matrix using pandas DataFrame for better formatting.
        """
        n = self.A.shape[0]
        df = pd.DataFrame(
            self.A, 
            columns=[f'S{i}' for i in range(n)],
            index=[f'S{i}' for i in range(n)]
        )
        
        # Replace zeros with dots for cleaner visualization
        df = df.replace(0, '.')
        print(df.round(precision))





if __name__ == "__main__":
    feature_set = load_mfccs("feature_set")
    hmm = HMM(8, 13, feature_set)
    hmm.print_parameters()
