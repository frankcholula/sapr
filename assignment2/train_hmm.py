from hmm import HMM
from mfcc_extract import load_mfccs, load_mfccs_by_word
import numpy as np

if __name__ == "__main__":
    feature_set = load_mfccs("feature_set")
    hood_features= load_mfccs_by_word("feature_set", "hood")
    heed_features = load_mfccs_by_word("feature_set", "heed")
    hood_hmm = HMM(8, 13, feature_set)
    heed_hmm = HMM(8, 13, feature_set)
    hood_hmm.baum_welch(hood_features, 10)
    heed_hmm.baum_welch(heed_features, 10)