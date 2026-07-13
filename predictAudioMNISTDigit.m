function predictedDigit = predictAudioMNISTDigit(model, audioFile)
% predictAudioMNISTDigit Predict spoken digit from a WAV file.
%
% Example:
%   yhat = predictAudioMNISTDigit(model, "0_01_0.wav");

arguments
    model (1,1) struct
    audioFile (1,1) string
end

[signal, fs] = audioread(audioFile);
if size(signal, 2) > 1
    signal = mean(signal, 2);
end
if fs ~= model.sampleRate
    signal = resample(signal, model.sampleRate, fs);
end

features = localExtract(signal, model.sampleRate, model.windowDuration, model.hopDuration);
predictedDigit = predict(model.classifier, features);
end

function feats = localExtract(x, fs, windowDuration, hopDuration)
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
