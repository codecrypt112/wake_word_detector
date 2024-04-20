import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Parameters
SAMPLES_DIR = "samples"
WAKE_WORD_DIR = os.path.join(SAMPLES_DIR, "j")
NON_WAKE_WORD_DIR = os.path.join(SAMPLES_DIR, "non-j")
OUTPUT_DIR = "preprocessed_data"
SAMPLE_RATE = 16000
DURATION = 1  # Duration of audio samples in seconds
NUM_MFCC = 13  # Number of MFCC coefficients
N_FFT = 2048
HOP_LENGTH = 512
NUM_CLASSES = 2  # Wake word and non-wake word
RANDOM_STATE = 42

def extract_mfcc(audio, num_mfcc=13, n_fft=2048, hop_length=512):
    """Extract Mel-frequency cepstral coefficients (MFCCs) from audio."""
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs


def preprocess_data():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Prepare labels and features
    labels = []
    features = []

    # Process wake word samples
    for filename in os.listdir(WAKE_WORD_DIR):
        if filename.endswith(".wav"):
            filepath = os.path.join(WAKE_WORD_DIR, filename)
            audio, _ = librosa.load(filepath, sr=SAMPLE_RATE)
            mfccs = extract_mfcc(audio, num_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
            features.append(mfccs.T)  # Transpose to match shape (time_steps, num_mfcc)
            labels.append(1)  # Wake word label

    # Process non-wake word samples
    for filename in os.listdir(NON_WAKE_WORD_DIR):
        if filename.endswith(".wav"):
            filepath = os.path.join(NON_WAKE_WORD_DIR, filename)
            audio, _ = librosa.load(filepath, sr=SAMPLE_RATE)
            mfccs = extract_mfcc(audio, num_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
            features.append(mfccs.T)  # Transpose to match shape (time_steps, num_mfcc)
            labels.append(0)  # Non-wake word label

    # Convert labels to categorical
    label_encoder = LabelEncoder()
    encoded_labels = to_categorical(label_encoder.fit_transform(labels))

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(np.array(features), encoded_labels, test_size=0.2, random_state=RANDOM_STATE)

    # Save preprocessed data
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print("Preprocessing completed. Data saved in 'preprocessed_data' directory.")

if __name__ == "__main__":
    preprocess_data()


if __name__ == "__main__":
    preprocess_data()
