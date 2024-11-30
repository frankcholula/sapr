from hmm import HMM
from mfcc_extract import load_mfccs, load_mfccs_by_word


def train_hmm():
    vocabs = [
        "heed",
        "hid",
        "head",
        "had",
        "hard",
        "hud",
        "hod",
        "hoard",
        "hood",
        "whod",
        "heard",
    ]

    feature_set = load_mfccs("feature_set")
    features = {word: load_mfccs_by_word("feature_set", word) for word in vocabs}
    total_features_length = sum(len(features[word]) for word in vocabs)
    assert total_features_length == len(feature_set)

    hmms = {word: HMM(8, 13, feature_set, model_name=word) for word in vocabs}
    for word, hmm in hmms.items():
        hmm.baum_welch(features[word], 10)


if __name__ == "__main__":
    train_hmm()
