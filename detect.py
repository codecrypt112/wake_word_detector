import os
import pyaudio
import numpy as np
import tensorflow as tf
import librosa
from tensorflow.keras.models import load_model

# Parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
MODEL_PATH = "model.h5"
THRESHOLD = 0.5  # Threshold for wake word detection
SAMPLE_DURATION = 1  # Duration of each sample for prediction (in seconds)
ACTIVITY_THRESHOLD = 0.1  # Threshold for audio activity detection

# Load the trained model
model = load_model(MODEL_PATH)

# Function to preprocess audio data
def preprocess_audio_data(audio_data):
    # Convert audio data to floating-point and extract MFCC features
    audio_data_float = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    mfccs = librosa.feature.mfcc(y=audio_data_float, sr=RATE, n_mfcc=13)

    # Pad or truncate mfccs to match the expected input shape
    padded_mfccs = np.zeros((1, 31766))
    mfccs_length = mfccs.shape[1]
    padded_mfccs[0, :mfccs_length] = mfccs[0]

    return padded_mfccs

# Function to detect audio activity
def detect_activity(audio_data):
    energy = np.sum(audio_data.astype(np.float32) ** 2) / float(len(audio_data))
    return energy > ACTIVITY_THRESHOLD

# Real-time detection function
def real_time_detection():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* Listening for wake word...")

    detected = False  # Flag to track wake word detection

    while not detected:  # Loop until wake word is detected
        data = stream.read(int(RATE * SAMPLE_DURATION))
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Check for audio activity
        if detect_activity(audio_data):
            # Preprocess the audio data
            preprocessed_data = preprocess_audio_data(audio_data)

            # Make prediction
            prediction = model.predict(preprocessed_data)
            print("Prediction:", prediction)  # Debug print
            if prediction[0][0] > THRESHOLD:
                print("Wake word detected!")
                detected = True  # Set flag to exit loop

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    real_time_detection()
