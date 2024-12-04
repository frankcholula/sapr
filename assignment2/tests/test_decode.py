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


def test_viterbi_decode(trained_heed_model, heed_features):
    trained_heed_model.decode(heed_features[0])