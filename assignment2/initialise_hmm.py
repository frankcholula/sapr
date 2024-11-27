from mfcc_extract import load_mfccs
from hmm import HMM

if __name__ == "__main__":
    feature_set = load_mfccs("feature_set")
    hmm = HMM(8, 13, feature_set)
    hmm.print_parameters()
