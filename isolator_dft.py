
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq

# Load the synthetic signal
fs, data = wavfile.read("synthetic_radio_twinkle.wav")
data = data.astype(float)

# Apply FFT
N = len(data)
freqs = fftfreq(N, d=1/fs)
spectrum = fft(data)

# Create mask to preserve melody frequencies only
mask = np.zeros(N, dtype=bool)
for f in [440, 523.25, 659.25]:
    mask |= (np.abs(freqs - f) < 5)

# Apply mask
filtered_spectrum = np.zeros_like(spectrum)
filtered_spectrum[mask] = spectrum[mask]

# Inverse FFT to reconstruct
filtered_signal = np.real(ifft(filtered_spectrum))
filtered_signal /= np.max(np.abs(filtered_signal))  # Normalize

# Save result
wavfile.write("isolated_melody.wav", fs, (filtered_signal * 32767).astype(np.int16))
print("Saved: isolated_melody.wav")
