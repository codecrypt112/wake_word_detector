import os
import time
import pyaudio
import wave

# Parameters
SAMPLES_DIR = "samples"
WAKE_WORD = "j"
NON_WAKE_WORD = "non_j"
NUM_SAMPLES_PER_CLASS = 50
RECORD_SECONDS = 2  # Adjust as needed
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

def record_audio(filename, num):
    """Record audio and save to file."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Recording..." + str(num))

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save audio to file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def main():
    if not os.path.exists(SAMPLES_DIR):
        os.makedirs(SAMPLES_DIR)

    # Record wake word samples
    print("Recording wake word samples...")
    for i in range(NUM_SAMPLES_PER_CLASS):
        filename = os.path.join(SAMPLES_DIR, f"{WAKE_WORD}_{i+1}.wav")
        record_audio(filename, i)
        time.sleep(1)  # Pause between recordings

    # Record non-wake word samples
    print("Recording non-wake word samples...")
    for i in range(NUM_SAMPLES_PER_CLASS):
        filename = os.path.join(SAMPLES_DIR, f"{NON_WAKE_WORD}_{i+1}.wav")
        record_audio(filename, i)
        time.sleep(1)  # Pause between recordings

    print("All recordings completed.")

if __name__ == "__main__":
    main()
