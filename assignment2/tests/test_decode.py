import pytest
import numpy as np
from custom_hmm import HMM
from mfcc_extract import load_mfccs, load_mfccs_by_word


@pytest.fixture
def feature_set():
    return load_mfccs("feature_set")


@pytest.fixture
def hmm_model(feature_set):
    return HMM(8, 13, feature_set)


@pytest.fixture
def heed_features():
    return load_mfccs_by_word("feature_set", "heed")


@pytest.fixture
def trained_heed_model(hmm_model, heed_features):
    hmm_model.baum_welch(heed_features, max_iter=7)
    return hmm_model


def test_viterbi_matrix(trained_heed_model, heed_features):
    """Test to visualize the Viterbi matrix with detailed diagnostics."""
    # Get first test sequence - keep original shape (13, T)
    test_features = heed_features[0]
    T = test_features.shape[1]  # Number of frames

    # Compute emission matrix - this will give us (T, total_states)
    emission_matrix = trained_heed_model.compute_emission_matrix(test_features)
    print(f"\nEmission matrix shape: {emission_matrix.shape}")

    # Initialize Viterbi matrix (T, total_states)
    V = np.full((T, trained_heed_model.total_states), -np.inf)

    # Initialize first time step
    V[0, 0] = 0  # Start in entry state
    V[0, 1] = (
        np.log(trained_heed_model.A[0, 1]) + emission_matrix[0, 1]
    )  # Transition to first state

    # Print initial values
    print("\nInitial step calculations:")
    print(f"Entry state (t=0): {V[0,0]:.2f}")
    print(
        f"First state (t=0): log(A[0,1])={np.log(trained_heed_model.A[0,1]):.2f} + emission={emission_matrix[0,1]:.2f} = {V[0,1]:.2f}"
    )

    # Forward recursion
    for t in range(1, T):
        # Entry state is always -inf after t=0
        V[t, 0] = -np.inf

        # Handle real states (1 to num_states)
        for j in range(1, trained_heed_model.num_states + 1):
            self_loop = -np.inf
            from_prev = -np.inf

            # Self-loop if available
            if trained_heed_model.A[j, j] > 0:
                self_loop = V[t - 1, j] + np.log(trained_heed_model.A[j, j])

            # From previous state
            if j > 1 and trained_heed_model.A[j - 1, j] > 0:
                from_prev = V[t - 1, j - 1] + np.log(trained_heed_model.A[j - 1, j])
            elif j == 1:  # Special case for first real state
                from_prev = V[t - 1, 0] + np.log(trained_heed_model.A[0, 1])

            # Take maximum and add emission probability
            V[t, j] = np.max([self_loop, from_prev]) + emission_matrix[t, j]

        # Handle exit state transitions
        if (
            t >= trained_heed_model.num_states
        ):  # Allow exit after going through enough states
            last_to_exit = V[t - 1, trained_heed_model.num_states] + np.log(
                trained_heed_model.A[trained_heed_model.num_states, -1]
            )
            exit_self_loop = V[t - 1, -1] + np.log(trained_heed_model.A[-1, -1])
            V[t, -1] = np.max([last_to_exit, exit_self_loop])

    # Print Viterbi matrix
    print("\nViterbi Matrix (showing first 10 frames, log probabilities):")
    print(
        "Time\t"
        + "\t".join(
            [
                (
                    f"S{i}"
                    if i != 0 and i != trained_heed_model.total_states - 1
                    else ("Entry" if i == 0 else "Exit")
                )
                for i in range(trained_heed_model.total_states)
            ]
        )
    )

    for t in range(min(10, T)):
        print(
            f"t={t}\t"
            + "\t".join(
                f"{V[t,j]:.1f}" if V[t, j] != -np.inf else "-inf"
                for j in range(trained_heed_model.total_states)
            )
        )

    # Find best path
    path = []
    current_state = trained_heed_model.total_states - 1  # Start from exit state
    current_prob = V[T - 1, current_state]

    print(f"\nFinal log probability: {current_prob:.2f}")
    print("\nBest state sequence:")
    for t in range(T - 1, -1, -1):
        path.append(current_state)
        # Find best previous state
        best_prev_state = 0
        best_prev_prob = -np.inf
        for i in range(trained_heed_model.total_states):
            if trained_heed_model.A[i, current_state] > 0:
                prev_prob = V[t - 1, i] + np.log(trained_heed_model.A[i, current_state])
                if prev_prob > best_prev_prob:
                    best_prev_prob = prev_prob
                    best_prev_state = i
        current_state = best_prev_state

    path = path[::-1]  # Reverse to get correct order
    print(
        " ".join(
            [
                (
                    f"S{s}"
                    if s != 0 and s != trained_heed_model.total_states - 1
                    else ("Entry" if s == 0 else "Exit")
                )
                for s in path[:20]
            ]
        )
    )
