import unittest
import numpy as np
import os
import soundfile as sf
import tempfile
import shutil

from mfcc_extract import extract_mfcc, load_mfcc

class TestMFCCExtraction(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test audio file of a 440 Hz sine wave
        self.sample_rate = 22050
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        self.audio_data = np.sin(2 * np.pi * 440 * t)
        
        # Save test audio file in temp directory
        self.test_wav = os.path.join(self.test_dir, 'test.wav')
        sf.write(self.test_wav, self.audio_data, self.sample_rate)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_extract_mfcc(self):
        mfcc = extract_mfcc(self.test_wav)
        self.assertEqual(mfcc.shape[0], 13, "Should have 13 MFCC coefficients")        
        self.assertTrue(mfcc.shape[1] > 0, "Should have at least one frame")

    def test_save_and_load_mfcc(self):
        test_npy = os.path.join(self.test_dir, 'test.npy')
        mfcc = extract_mfcc(self.test_wav)
        np.save(test_npy, mfcc)        
        loaded_mfcc = load_mfcc(test_npy)
        np.testing.assert_array_equal(mfcc, loaded_mfcc)

if __name__ == '__main__':
    unittest.main()
