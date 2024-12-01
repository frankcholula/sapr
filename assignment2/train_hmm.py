from hmm import HMM
from mfcc_extract import load_mfccs, load_mfccs_by_word
from matplotlib import pyplot as plt

def plot_log_likelihood(log_likelihoods, title=None, save_path=None):
    """
    Create a plot showing how log likelihood changes during HMM training.
    
    Args:
        log_likelihoods (list): List of log likelihood values from training
        title (str, optional): Custom title for the plot
        save_path (str, optional): If provided, save plot to this path
    """
    # Create figure with reasonable size
    plt.figure(figsize=(10, 6))
    
    # Plot log likelihoods
    iterations = range(len(log_likelihoods))
    plt.plot(iterations, log_likelihoods, 'b-', linewidth=2, label='Log Likelihood')
    
    # Add markers to more easily see individual points
    plt.plot(iterations, log_likelihoods, 'bo', markersize=4)
    
    # Calculate improvement
    total_improvement = log_likelihoods[-1] - log_likelihoods[0]
    
    # Set labels and title
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Log Likelihood', fontsize=12)
    plt.title(title or f'HMM Training Progress\nTotal Improvement: {total_improvement:.2f}', 
              fontsize=14)
    
    # Add grid for easier reading
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()


def train_hmm():
    vocabs = [
        "heed",
        # "hid",
        # "head",
        # "had",
        # "hard",
        # "hud",
        # "hod",
        # "hoard",
        # "hood",
        # "whod",
        # "heard",
    ]

    feature_set = load_mfccs("feature_set")
    features = {word: load_mfccs_by_word("feature_set", word) for word in vocabs}
    # total_features_length = sum(len(features[word]) for word in vocabs)
    # assert total_features_length == len(feature_set)

    hmms = {word: HMM(8, 13, feature_set, model_name=word) for word in vocabs}

    
    for word, hmm in hmms.items():
        log_likelihoods = hmm.baum_welch(features[word], 100)
        plot_log_likelihood(log_likelihoods)



if __name__ == "__main__":
    train_hmm()
