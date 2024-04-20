import wave
import numpy as np
import pyaudio
from noisereduce import reduce_noise
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def record_audio(file_name, duration=5, channels=1, sample_rate=44100, chunk_size=1024):
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open stream
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    print("Recording...")

    frames = []

    # Record audio for the specified duration
    for i in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a WAV file
    wf = wave.open(file_name, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Convert binary data to numpy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    # Perform noise reduction
    denoised_audio_data = reduce_noise(audio_data, sample_rate)

    # Save the denoised audio to a WAV file
    denoised_wf = wave.open("denoised_" + file_name, 'wb')
    denoised_wf.setnchannels(channels)
    denoised_wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    denoised_wf.setframerate(sample_rate)
    denoised_wf.writeframes(denoised_audio_data.tobytes())
    denoised_wf.close()


def visualize_audio(file_name):
    print("Visualizing audio...")
    # Open WAV file
    with wave.open(file_name, 'rb') as wf:
        # Get audio parameters
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        duration = num_frames / float(sample_rate)

        # Read audio data
        audio_data = np.frombuffer(wf.readframes(num_frames), dtype=np.int16)

    # Time axis for waveform and spectrogram
    time = np.linspace(0, duration, num_frames)

    # Plot waveform
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(time, audio_data, color='b')
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot spectrogram
    plt.subplot(3, 1, 2)
    f, t, Sxx = spectrogram(audio_data, fs=sample_rate)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Power Spectral Density (dB/Hz)')

    # Plot frequency spectrum
    plt.subplot(3, 1, 3)
    plt.magnitude_spectrum(audio_data, Fs=sample_rate, scale='dB', color='r')
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_name = "recorded_audio.wav"
    duration = 5  # Duration of recording in seconds
    record_audio(file_name, duration)
    visualize_audio(file_name)
