function [model, metrics] = trainAudioMNISTDecisionTree(dataRoot, varargin)
% trainAudioMNISTDecisionTree Train a spoken digit classifier for AudioMNIST.
%
% This function extracts FFT, STFT, and wavelet features from .wav files and
% trains a decision-tree classifier that predicts the spoken digit.
%
% AudioMNIST file names are expected to follow: "<digit>_<speaker>_<index>.wav"
%
% Example:
%   [model, metrics] = trainAudioMNISTDecisionTree("AudioMNIST/data");
%
% Inputs:
%   dataRoot  - Path to AudioMNIST data directory.
%
% Name-value options:
%   "SampleRate"       - Target sample rate (default: 16000)
%   "WindowDuration"   - STFT window duration in seconds (default: 0.025)
%   "HopDuration"      - STFT hop duration in seconds (default: 0.010)
%   "Holdout"          - Test split fraction (default: 0.2)
%   "MaxFilesPerDigit" - Optional limit for each class (default: inf)
%
% Outputs:
%   model   - Struct containing trained decision tree and configuration.
%   metrics - Struct with accuracy, confusion matrix, and class labels.

arguments
    dataRoot (1,1) string
end
arguments (Repeating)
    varargin
end

p = inputParser;
addParameter(p, "SampleRate", 16000, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, "WindowDuration", 0.025, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, "HopDuration", 0.010, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, "Holdout", 0.2, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
addParameter(p, "MaxFilesPerDigit", inf, @(x) isnumeric(x) && isscalar(x) && x >= 1);
parse(p, varargin{:});
cfg = p.Results;

files = dir(fullfile(dataRoot, "**", "*.wav"));
if isempty(files)
    error("No .wav files found under %s", dataRoot);
end

if exist("wavedec", "file") ~= 2
    error("Wavelet Toolbox function 'wavedec' is required for wavelet features.");
end

featureRows = [];
labels = categorical.empty(0, 1);
perDigitCount = containers.Map("KeyType", "char", "ValueType", "double");

for i = 1:numel(files)
    filePath = fullfile(files(i).folder, files(i).name);
    digitLabel = parseDigitLabel(files(i).name);
    if strlength(digitLabel) == 0
        continue;
    end

    digitKey = char(digitLabel);
    if ~isKey(perDigitCount, digitKey)
        perDigitCount(digitKey) = 0;
    end
    if perDigitCount(digitKey) >= cfg.MaxFilesPerDigit
        continue;
    end

    [signal, fs] = audioread(filePath);
    if size(signal, 2) > 1
        signal = mean(signal, 2);
    end
    if fs ~= cfg.SampleRate
        signal = resample(signal, cfg.SampleRate, fs);
    end

    featureRows(end + 1, :) = extractFeatures(signal, cfg.SampleRate, cfg.WindowDuration, cfg.HopDuration); %#ok<AGROW>
    labels(end + 1, 1) = categorical(digitLabel); %#ok<AGROW>
    perDigitCount(digitKey) = perDigitCount(digitKey) + 1;
end

if isempty(featureRows)
    error("No valid AudioMNIST files were parsed.");
end

cv = cvpartition(labels, "Holdout", cfg.Holdout);
XTrain = featureRows(training(cv), :);
YTrain = labels(training(cv));
XTest = featureRows(test(cv), :);
YTest = labels(test(cv));

tree = fitctree(XTrain, YTrain);
YPred = predict(tree, XTest);

classes = categories(labels);
cm = confusionmat(YTest, YPred, "Order", categorical(classes));
accuracy = mean(YPred == YTest);

model = struct( ...
    "classifier", tree, ...
    "sampleRate", cfg.SampleRate, ...
    "windowDuration", cfg.WindowDuration, ...
    "hopDuration", cfg.HopDuration, ...
    "classNames", classes);

metrics = struct( ...
    "accuracy", accuracy, ...
    "confusionMatrix", cm, ...
    "classNames", {classes}, ...
    "numSamples", numel(labels));
end

function digit = parseDigitLabel(fileName)
parts = split(string(fileName), "_");
if isempty(parts)
    digit = "";
    return;
end

candidate = regexp(parts(1), "^\d$", "match", "once");
if isempty(candidate)
    digit = "";
else
    digit = string(candidate);
end
end

function feats = extractFeatures(x, fs, windowDuration, hopDuration)
x = x(:);
x = x - mean(x);
if rms(x) == 0
    x = x + eps;
end

nfft = 2^nextpow2(max(256, numel(x)));
mag = abs(fft(x, nfft));
mag = mag(1:floor(nfft/2) + 1);
f = linspace(0, fs/2, numel(mag)).';
magNorm = mag / (sum(mag) + eps);

centroid = sum(f .* magNorm);
spread = sqrt(sum(((f - centroid) .^ 2) .* magNorm));
cumulative = cumsum(magNorm);
rolloffIdx = find(cumulative >= 0.85, 1, "first");
if isempty(rolloffIdx)
    rolloff = f(end);
else
    rolloff = f(rolloffIdx);
end
flatness = exp(mean(log(mag + eps))) / (mean(mag + eps));
energy = rms(x);

win = max(16, round(windowDuration * fs));
hop = max(8, round(hopDuration * fs));
[S, ~, ~] = spectrogram(x, win, win - hop, nfft, fs);
S = abs(S);
stftLog = log1p(S);
stftMean = mean(stftLog, "all");
stftStd = std(stftLog, 0, "all");
powerPerFrame = S .^ 2;
frameNorm = powerPerFrame ./ (sum(powerPerFrame, 1) + eps);
spectralEntropy = -sum(frameNorm .* log2(frameNorm + eps), 1);
entropyMean = mean(spectralEntropy);

[coeffs, levels] = wavedec(x, 4, "db4");
a4 = appcoef(coeffs, levels, "db4", 4);
d1 = detcoef(coeffs, levels, 1);
d2 = detcoef(coeffs, levels, 2);
d3 = detcoef(coeffs, levels, 3);
d4 = detcoef(coeffs, levels, 4);
waveletEnergies = [sum(a4.^2), sum(d1.^2), sum(d2.^2), sum(d3.^2), sum(d4.^2)];
waveletEnergies = waveletEnergies ./ (sum(waveletEnergies) + eps);

feats = [centroid, spread, rolloff, flatness, energy, stftMean, stftStd, entropyMean, waveletEnergies];
end
