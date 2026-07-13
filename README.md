# Spoken Digit Recognition using MATLAB

Automatic spoken digit recognition (0–9) using signal processing and machine learning techniques.

This project was developed in MATLAB using the AudioMNIST dataset. It explores multiple stages of the speech recognition pipeline, including signal preprocessing, feature extraction in different domains, time-frequency analysis, and automatic classification.

## Features

- Audio preprocessing
  - Silence removal
  - Amplitude normalization
  - Signal length standardization

- Temporal feature extraction
  - Total Energy
  - Maximum Amplitude
  - Minimum Amplitude
  - Mean
  - Standard Deviation

- Spectral analysis
  - Fourier Transform (FFT)
  - Peak Frequency
  - Spectral Energy
  - Spectral Edge Frequency (SEF90)

- Time-Frequency analysis
  - Short-Time Fourier Transform (STFT)
  - Discrete Wavelet Transform (DWT)

- Machine Learning
  - Decision Tree classifier
  - 9 selected features
  - 87.6% classification accuracy

## Dataset

AudioMNIST

- 60 speakers
- Spoken digits (0–9)
- 50 repetitions per digit

## Technologies

- MATLAB
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox

## Results

The Decision Tree classifier achieved:

- Accuracy: **87.6%**
- 438 correctly classified samples out of 500

## Repository Structure

```
meta1.mlx
meta2.mlx
report.pdf
README.md
```

## Authors

- João Natálio
- Gonçalo Costa
