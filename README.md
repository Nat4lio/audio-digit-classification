# audio-digit-classification

Minimal MATLAB pipeline for spoken digit classification on AudioMNIST using:

- FFT features
- STFT features
- Wavelet features
- Decision tree classification

## Usage

```matlab
[model, metrics] = trainAudioMNISTDecisionTree("path/to/AudioMNIST/data");
disp(metrics.accuracy)

predictedDigit = predictAudioMNISTDigit(model, "path/to/sample.wav");
disp(predictedDigit)
```

## Notes

- Audio files are parsed recursively from the provided directory.
- File names are expected in AudioMNIST style: `<digit>_<speaker>_<index>.wav`.
- `wavedec` (Wavelet Toolbox) is required for wavelet-based features.