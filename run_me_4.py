
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, lfilter
import os
import sys

def bandpass_filter_fft(signal, fs, center_freq, bandwidth):
    N = len(signal)
    freqs = fftfreq(N, d=1/fs)
    spectrum = fft(signal)
    mask = (np.abs(freqs - center_freq) < bandwidth / 2)
    filtered_spectrum = np.zeros_like(spectrum)
    filtered_spectrum[mask] = spectrum[mask]
    return np.real(ifft(filtered_spectrum))

def demodulate(signal, fs, carrier_freq):
    t = np.arange(len(signal)) / fs
    return signal * 2 * np.cos(2 * np.pi * carrier_freq * t)

def lowpass(signal, fs, cutoff=2000):
    b, a = butter(4, cutoff / (fs / 2), btype='low')
    return lfilter(b, a, signal)

def save_audio(filename, signal, fs):
    signal /= np.max(np.abs(signal) + 1e-12)
    wavfile.write(filename, fs, (signal * 32767).astype(np.int16))
    print(f"Saved: {filename}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_me_4.py <filename.wav>")
        sys.exit(1)

    input_file = sys.argv[1]
    fs, data = wavfile.read(input_file)
    data = data.astype(float)

    # Define frequency bands and names
    bands = [
        ("lead", 3500),
        ("bass", 7500),
        ("harmony", 11000),
        ("rhythm", 16000)
    ]
    bandwidth = 3000

    # Create output folder
    base = os.path.splitext(os.path.basename(input_file))[0]
    out_dir = base + "_tracks"
    os.makedirs(out_dir, exist_ok=True)

    combined = np.zeros_like(data)

    for name, freq in bands:
        band = bandpass_filter_fft(data, fs, freq, bandwidth)
        baseband = demodulate(band, fs, freq)
        audio = lowpass(baseband, fs)
        combined += audio
        save_audio(os.path.join(out_dir, f"{name}.wav"), audio, fs)

    # Save combined harmony track
    combined /= np.max(np.abs(combined) + 1e-12)
    save_audio(os.path.join(out_dir, "combined_harmony.wav"), combined, fs)

if __name__ == "__main__":
    main()
