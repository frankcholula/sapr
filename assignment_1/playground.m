files = {'hood_m.wav', 'hood_f.wav'};

function [fs, duration, numSamples] = playAudio(fileName)
    fileName = strcat('speech/',fileName);
    [audio, fs] = audioread(fileName);
    duration = length(audio) / fs;
    numSamples = length(audio);
    fprintf('Playing %s at %d Hz with duration %.2f seconds and %d samples\n',fileName, fs, duration, numSamples);
    figure;
    spectrogram(audio, hamming(256), 128, 1024, fs, 'yaxis');
    title(['Spectrogram of ', fileName]);
    sound(audio, fs);
    pause(duration + 1) 
end

[fs, duration, numSamples] = playAudio(files{1});
segmentDuration = 0.1;
numSegmentSamples = segmentDuration * fs;

