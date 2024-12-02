import numpy as np
from hmm import HMM

def test_emission_parameter_stability():
    """Test numerical stability of emission parameter updates during training."""
    # Initialize with small synthetic dataset
    num_states = 8
    num_obs = 13
    num_frames = 100
    
    # Create a simple feature set for initialization
    np.random.seed(42)
    init_feature = np.random.randn(num_obs, num_frames)
    feature_set = [init_feature]
    
    # Initialize HMM
    hmm = HMM(num_states, num_obs, feature_set)
    
    # Create training data with two distinct clusters
    training_features = [
        np.random.randn(num_obs, num_frames) * 2 + 5,  # Cluster 1
        np.random.randn(num_obs, num_frames) * 2 - 5   # Cluster 2
    ]
    
    # Create corresponding gamma values that encourage states to specialize
    gamma_sequences = []
    for seq_idx, _ in enumerate(training_features):
        gamma = np.zeros((num_states, num_frames))
        # Assign frames roughly equally across first half of states for first sequence
        # and second half of states for second sequence
        frames_per_state = num_frames // (num_states // 2)
        start_state = seq_idx * (num_states // 2)
        end_state = start_state + (num_states // 2)
        
        for s in range(start_state, end_state):
            start_idx = (s - start_state) * frames_per_state
            end_idx = start_idx + frames_per_state
            if end_idx > num_frames:
                end_idx = num_frames
            gamma[s, start_idx:end_idx] = 1.0
        gamma_sequences.append(gamma)
    
    # Test stability over multiple updates
    num_updates = 5
    
    print("\nInitial statistics:")
    print(f"Mean range: [{np.min(hmm.B['mean'])}, {np.max(hmm.B['mean'])}]")
    print(f"Variance range: [{np.min(hmm.B['covariance'])}, {np.max(hmm.B['covariance'])}]")
    
    for i in range(num_updates):
        hmm.update_B(training_features, gamma_sequences)
        
        # Verify basic statistical properties
        assert np.all(np.isfinite(hmm.B['mean'])), f"Non-finite values in means at iteration {i}"
        assert np.all(np.isfinite(hmm.B['covariance'])), f"Non-finite values in variances at iteration {i}"
        assert np.all(hmm.B['covariance'] > 0), f"Non-positive variances at iteration {i}"
        
        # Check for reasonable bounds
        assert np.all(np.abs(hmm.B['mean']) < 1000), f"Means too large at iteration {i}"
        assert np.all(hmm.B['covariance'] < 50000), f"Variances too large at iteration {i}"
        
        print(f"\nUpdate {i+1} statistics:")
        print(f"Mean range: [{np.min(hmm.B['mean'])}, {np.max(hmm.B['mean'])}]")
        print(f"Variance range: [{np.min(hmm.B['covariance'])}, {np.max(hmm.B['covariance'])}]")
        
        # Verify variance floor is maintained for each state
        for j in range(num_states):
            var_floor = 0.01 * np.mean(hmm.B['covariance'][:, j])
            assert np.all(hmm.B['covariance'][:, j] >= var_floor), f"Variance floor violated for state {j} at iteration {i}"

def test_emission_update_correctness():
    """Test correctness of emission parameter updates with simple cases."""
    num_states = 2
    num_obs = 2
    
    # Initialize with zero features
    init_feature = np.zeros((num_obs, 4))
    hmm = HMM(num_states, num_obs, [init_feature])
    
    # Create a simple feature set with two clear clusters
    features = [np.array([
        [0, 0, 10, 10],  # First dimension
        [0, 0, 10, 10]   # Second dimension
    ])]
    
    gamma = [np.array([
        [1, 1, 0, 0],    # State 1 owns first two frames
        [0, 0, 1, 1]     # State 2 owns last two frames
    ])]
    
    print("\nTesting with simple clustered data:")
    print("Initial parameters:")
    print(f"Means:\n{hmm.B['mean']}")
    print(f"Variances:\n{hmm.B['covariance']}")
    
    hmm.update_B(features, gamma)
    
    print("\nUpdated parameters:")
    print(f"Means:\n{hmm.B['mean']}")
    print(f"Variances:\n{hmm.B['covariance']}")
    
    # After update, each state should specialize to its cluster
    expected_means = np.array([
        [0, 10],  # First dimension: state 1 mean = 0, state 2 mean = 10
        [0, 10]   # Second dimension: state 1 mean = 0, state 2 mean = 10
    ])
    
    np.testing.assert_array_almost_equal(
        hmm.B['mean'],
        expected_means,
        decimal=1,
        err_msg=f"Incorrect means. Expected\n{expected_means}\n, got\n{hmm.B['mean']}"
    )
    
    # Variances should be very small since points in each state are identical
    # We'll just check they're below a small threshold
    assert np.all(hmm.B['covariance'] < 1), \
        f"Variances too large. Expected small values, got\n{hmm.B['covariance']}"

def test_emission_numerical_stability():
    """Test numerical stability with extreme values."""
    num_states = 8
    num_obs = 13
    num_frames = 50
    
    # Initialize with reasonable values
    init_feature = np.random.randn(num_obs, num_frames)
    hmm = HMM(num_states, num_obs, [init_feature])
    
    # Test Case 1: Very small values
    small_features = [np.random.randn(num_obs, num_frames) * 0.001]
    small_gamma = [np.ones((num_states, num_frames)) * 1e-10]
    
    print("\nTesting with very small values:")
    print(f"Feature range: [{np.min(small_features[0])}, {np.max(small_features[0])}]")
    print(f"Gamma range: [{np.min(small_gamma[0])}, {np.max(small_gamma[0])}]")
    
    hmm.update_B(small_features, small_gamma)
    
    assert np.all(np.isfinite(hmm.B['mean'])), "Means became non-finite with small values"
    assert np.all(np.isfinite(hmm.B['covariance'])), "Variances became non-finite with small values"
    assert np.all(hmm.B['covariance'] > 0), "Non-positive variances with small values"
    
    # Test Case 2: Very large values
    large_features = [np.random.randn(num_obs, num_frames) * 1e3]
    large_gamma = [np.random.dirichlet([1] * num_states, num_frames).T]
    
    print("\nTesting with very large values:")
    print(f"Feature range: [{np.min(large_features[0])}, {np.max(large_features[0])}]")
    print(f"Gamma range: [{np.min(large_gamma[0])}, {np.max(large_gamma[0])}]")
    
    hmm.update_B(large_features, large_gamma)
    
    assert np.all(np.isfinite(hmm.B['mean'])), "Means became non-finite with large values"
    assert np.all(np.isfinite(hmm.B['covariance'])), "Variances became non-finite with large values"
    assert np.all(hmm.B['covariance'] > 0), "Non-positive variances with large values"
