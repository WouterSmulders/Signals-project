from pydub import AudioSegment
import numpy as np
import os

def audiosegment_to_np(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
    return samples.astype(np.float32)

def np_to_audiosegment(samples, sample_width, frame_rate, channels):
    samples = np.int16(samples)
    if channels == 2:
        samples = samples.flatten()
    return AudioSegment(
        samples.tobytes(),
        frame_rate=frame_rate,
        sample_width=sample_width,
        channels=channels
    )

def invert_phase(audio_np):
    return -audio_np

def isolate_vocals(original_path, instrumental_path, output_path='isolated_vocals.wav'):
    # Load audio files
    original = AudioSegment.from_file(original_path)
    instrumental = AudioSegment.from_file(instrumental_path)

    # Ensure same length
    min_len = min(len(original), len(instrumental))
    original = original[:min_len]
    instrumental = instrumental[:min_len]

    # Convert to NumPy
    original_np = audiosegment_to_np(original)
    instrumental_np = audiosegment_to_np(instrumental)

    # Invert instrumental and add to original
    inverted_instrumental = invert_phase(instrumental_np)
    vocals_np = original_np + inverted_instrumental

    # Normalize output
    vocals_np /= np.max(np.abs(vocals_np)) * 1.1

    # Convert back to audio
    vocals_audio = np_to_audiosegment(
        vocals_np,
        sample_width=original.sample_width,
        frame_rate=original.frame_rate,
        channels=original.channels
    )

    # Export
    vocals_audio.export(output_path, format="wav")
    print(f"Saved isolated vocals to {output_path}")

# Example usage
isolate_vocals("example1.mp3", "harmony.wav")