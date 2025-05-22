
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import hilbert, butter, lfilter

def ssb_demodulate(iq_signal, fs, sideband='usb', audio_freq_shift=3000):
    t = np.arange(len(iq_signal)) / fs
    shift = np.exp(-1j * 2 * np.pi * audio_freq_shift * t)
    shifted = iq_signal * shift

    if sideband == 'lsb':
        shifted = np.conj(shifted)

    analytic = hilbert(np.real(shifted))
    audio = np.real(analytic)

    b, a = butter(5, 3000 / (fs / 2), btype='low')
    filtered_audio = lfilter(b, a, audio)
    filtered_audio /= np.max(np.abs(filtered_audio) + 1e-12)
    return filtered_audio

def main():
    input_file = "SDRuno_20200912_004330Z_7150kHz.wav"
    output_file = "ssb_audio.wav"

    fs, data = wavfile.read(input_file)
    I = data[:, 0].astype(np.float32)
    Q = data[:, 1].astype(np.float32)
    iq_signal = I + 1j * Q

    segment_duration = 5  # seconds
    N = int(fs * segment_duration)
    iq_segment = iq_signal[:N]

    freqs = fftfreq(N, d=1/fs)
    spectrum = fft(iq_segment)

    target_freq = 14880  # Hz
    bandwidth = 3000    # Hz
    mask = (freqs > target_freq - bandwidth/2) & (freqs < target_freq + bandwidth/2)
    filtered_spectrum = np.zeros_like(spectrum)
    filtered_spectrum[mask] = spectrum[mask]
    iq_filtered = ifft(filtered_spectrum)

    audio = ssb_demodulate(iq_filtered, fs, sideband='usb', audio_freq_shift=target_freq)
    decimation_factor = fs // 8000
    audio_downsampled = audio[::decimation_factor]

    wavfile.write(output_file, 8000, (audio_downsampled * 32767).astype(np.int16))
    print(f"Saved demodulated audio to {output_file}")

if __name__ == "__main__":
    main()
