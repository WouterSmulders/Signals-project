import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, medfilt
import matplotlib.pyplot as plt
from collections import Counter

MALE_MIN, MALE_MAX = 85, 180
FEMALE_MIN, FEMALE_MAX = 165, 255
OVERLAP_MIN, OVERLAP_MAX = 165, 180

def highpass_filter(data, fs, cutoff=70, order=4):
    """
    Apply a high-pass filter to the audio signal.
    Parameters:
        data: numpy array, input audio signal
        fs: int, sample rate
        cutoff: float, cutoff frequency in Hz
        order: int, filter order
    Output:
        filtered: numpy array, filtered audio signal
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered = lfilter(b, a, data)
    return filtered

def find_fundamental(frame, fs, fmin=80, fmax=300):
    """
    Estimate the fundamental frequency in a frame using the DFT.
    Parameters:
        frame: numpy array, audio frame
        fs: int, sample rate
        fmin: float, minimum frequency to consider
        fmax: float, maximum frequency to consider
    Output:
        fundamental_freq: float, estimated fundamental frequency in Hz
    """
    N = len(frame)
    if np.all(frame == 0):
        return 0
    fft = np.fft.rfft(frame * np.hanning(N))
    freqs = np.fft.rfftfreq(N, 1/fs)
    idx_range = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    magnitudes = np.abs(fft)[idx_range]
    if len(magnitudes) == 0:
        return 0
    idx_max = np.argmax(magnitudes)
    fundamental_freq = freqs[idx_range][idx_max]
    return fundamental_freq

def classify_register(freq):
    """
    Classify the given frequency as male, female, overlap, or other.
    Parameters:
        freq: float, frequency in Hz
    Output:
        str: register classification label
    """
    if MALE_MIN <= freq < OVERLAP_MIN:
        return 'male'
    elif OVERLAP_MIN <= freq <= OVERLAP_MAX:
        return 'overlap'
    elif OVERLAP_MAX < freq <= FEMALE_MAX:
        return 'female'
    else:
        return 'other'

def classify_audio_dft(filename, frame_duration=0.10):
    """
    Classify an audio file as male or female using DFT-based pitch detection.
    Parameters:
        filename: str, path to the .wav file
        frame_duration: float, length of each frame in seconds
    Outputs:
        result: str, classification by consecutive-frames method
        vote_result: str, classification by majority vote
        fundamentals: list of float, fundamental frequencies per frame
        classifications: list of str, register labels per frame
    """
    fs, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data / np.max(np.abs(data))
    data = highpass_filter(data, fs) 

    frame_size = int(fs * frame_duration)
    num_frames = len(data) // frame_size

    fundamentals = []
    classifications = []

    for i in range(num_frames):
        frame = data[i * frame_size : (i+1) * frame_size]
        freq = find_fundamental(frame, fs)
        fundamentals.append(freq)
        classifications.append(classify_register(freq))

    # Median filter to smooth the detected frequencies
    fundamentals = medfilt(fundamentals, kernel_size=5)

    # Classify by consecutive frames
    result = 'undecided'
    for i in range(len(classifications) - 4):
        five = classifications[i:i+5]
        if all(x == 'male' for x in five):
            result = 'male'
            break
        elif all(x == 'female' for x in five):
            result = 'female'
            break
        elif all(x == 'overlap' for x in five):
            continue
    if result == 'undecided':
        for reg in classifications:
            if reg == 'male':
                result = 'male'
                break
            elif reg == 'female':
                result = 'female'
                break

    print(f"Consecutive-frames classification: {result}")

    # Majority vote classification
    count = Counter(classifications)
    # Only count 'male' and 'female' votes for majority
    num_male = count['male']
    num_female = count['female']
    if num_male > num_female:
        vote_result = 'male'
    elif num_female > num_male:
        vote_result = 'female'
    else:
        vote_result = 'undecided'
    print(f"Majority-vote classification: {vote_result}")

    return result, vote_result, fundamentals, classifications


if __name__ == "__main__":
    """
    Example: Classify and plot the fundamental frequency for a sample file.
    """
    filename = 'Male7.wav'
    result, vote_result, fundamentals, classifications = classify_audio_dft(filename)
    plt.figure(figsize=(10, 4))
    plt.plot(fundamentals, marker='o', label='Estimated Fundamental (Hz)')
    plt.axhspan(MALE_MIN, MALE_MAX, color='blue', alpha=0.1, label='Male')
    plt.axhspan(FEMALE_MIN, FEMALE_MAX, color='red', alpha=0.1, label='Female')
    plt.axhspan(OVERLAP_MIN, OVERLAP_MAX, color='purple', alpha=0.2, label='Overlap')
    plt.xlabel('Frame')
    plt.ylabel('Estimated Fundamental Frequency (Hz)')
    plt.title('Fundamental Frequency Per Frame (DFT-based, Noise-Reduced)')
    plt.legend()
    plt.tight_layout()
    plt.show()
