import librosa
import numpy as np
import os
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)


def extract_mfcc(audio_path: str) -> np.ndarray:
    try:
        y, sr = librosa.load(audio_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc
    except Exception as e:
        logging.error(f"Error processing {audio_path}: {str(e)}")
        raise


def extract_mfccs(input_folder: str, output_folder: str) -> str:
    logging.debug(f"Extracting MFCCs from {input_folder} to {output_folder}...")
    os.makedirs(output_folder, exist_ok=True)

    processed_files = 0
    for file in os.listdir(input_folder):
        if file.endswith(".mp3"):
            try:
                input_path = os.path.join(input_folder, file)
                output_file = os.path.join(output_folder, file.replace(".mp3", ".npy"))

                mfcc = extract_mfcc(input_path)
                np.save(output_file, mfcc)

                processed_files += 1
                logging.debug(f"Processed {file} ({processed_files} files done)")

            except Exception as e:
                logging.error(f"Failed to process {file}: {str(e)}")
                continue

    logging.info(f"Completed processing {processed_files} files")
    return output_folder


def load_mfcc(file_path: str) -> np.ndarray:
    try:
        return np.load(file_path)
    except Exception as e:
        logging.error(f"Failed to load MFCC from {file_path}: {str(e)}")
        raise

def plot_histogram(data: np.ndarray):
    plt.hist(data.flatten(), bins=50)
    plt.show()


if __name__ == "__main__":
    # TRAINING_FOLDER = "dev_set"
    # extract_mfccs(TRAINING_FOLDER, "feature_set")
    test_mfcc = load_mfcc("feature_set/sp01a_w01_heed.npy")
    plot_histogram(test_mfcc)
