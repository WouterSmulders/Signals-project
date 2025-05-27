Signals and Systems Project: Vocal Gender Classification Using DFT

Author: [Your Name]
Date: [Submission Date]

Overview:
---------
This project implements a simple, explainable algorithm for classifying isolated vocal audio recordings as either male or female based on frequency analysis using the Discrete Fourier Transform (DFT). The classification leverages frame-based pitch detection and two decision strategies: "consecutive-frames" and "majority-vote" methods.

How It Works:
-------------
1. The program loads a .wav audio file containing a vocal recording.
2. The audio is split into short frames (default: 0.10 seconds).
3. For each frame, the DFT (via FFT) is computed, and the most prominent frequency within the vocal range is selected as the fundamental frequency.
4. The sequence of estimated fundamentals is median-filtered to smooth out noise.
5. Each frame is classified as "male", "female", "overlap", or "other" based on vocal frequency ranges.
6. Two classification strategies are used:
   - "Consecutive-frames": If 5 consecutive frames are labeled as male or female, that label is chosen.
   - "Majority-vote": The most frequent label ("male" or "female") among all frames is chosen.
7. Results and frequency plots are displayed for analysis.

How to Use:
-----------
1. Make sure you have Python 3.x installed with the following libraries:
   - numpy
   - scipy
   - matplotlib

   You can install these with:
       pip install numpy scipy matplotlib

2. Place your .wav vocal file in the same directory as the script.

3. Edit the script's main section to set the correct filename, e.g.:
       filename = 'YourAudioFile.wav'

4. Run the script:
       python classify.py

5. The terminal will display both the consecutive-frames and majority-vote classification results. A plot will show the detected fundamental frequencies per frame.

Files:
------
- classify.py: Main Python script for classification and plotting.
- [audio files]: Test .wav files (not included for copyright reasons).

Notes:
------
- This project uses only DFT-based (FFT) analysis for fundamental frequency detection, in line with course requirements.
- Noise reduction is applied using a high-pass filter.
- The project was developed iteratively, with improvements and verification aided by AI (see AI usage report for details).

AI Usage Disclosure:
--------------------
Key algorithm choices, code structure, and debugging were supported by OpenAI's ChatGPT, with user verification and hands-on testing at each step. See the included AI_usage_report.tex for specific prompts and verification steps.

Contact:
--------
For questions or suggestions, please contact [Your Name / Email].

