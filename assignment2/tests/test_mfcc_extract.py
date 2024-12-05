# import pytest
# import numpy as np
# import os
# import soundfile as sf
# import tempfile
# import shutil
# from mfcc_extract import extract_mfcc, load_mfcc


# @pytest.fixture
# def test_dir():
#     tmp_dir = tempfile.mkdtemp()
#     yield tmp_dir
#     shutil.rmtree(tmp_dir)


# @pytest.fixture
# def test_wav(test_dir):
#     # Create a test audio file of a 440 Hz sine wave
#     sample_rate = 22050
#     duration = 1.0  # 1 second
#     t = np.linspace(0, duration, int(sample_rate * duration))
#     audio_data = np.sin(2 * np.pi * 440 * t)

#     # Save test audio file in temp directory
#     wav_path = os.path.join(test_dir, "test.wav")
#     sf.write(wav_path, audio_data, sample_rate)
#     return wav_path


# def test_extract_mfcc(test_wav):
#     mfcc = extract_mfcc(test_wav)
#     assert mfcc.shape[0] == 13, "Should have 13 MFCC coefficients"
#     assert mfcc.shape[1] > 0, "Should have at least one frame"


# def test_save_and_load_mfcc(test_dir, test_wav):
#     # Create paths in temp directory
#     test_npy = os.path.join(test_dir, "test.npy")
#     # Extract and save
#     mfcc = extract_mfcc(test_wav)
#     np.save(test_npy, mfcc)
#     # Load and compare
#     loaded_mfcc = load_mfcc(test_npy)
#     np.testing.assert_array_equal(mfcc, loaded_mfcc)
