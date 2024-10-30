function [audio, fs, duration, numSamples] = playAudio(fileName)
    filePath = strcat('speech/',fileName);
    [audio, fs] = audioread(filePath);
    duration = length(audio) / fs;
    numSamples = length(audio);
    fprintf('Playing %s at %d Hz with duration %.2f seconds and %d samples\n',fileName, fs, duration, numSamples);
    
    figure();
    subplot(3,1,1);
    plotSpectrogram(audio, fs, fileName);
    subplot(3,1,2);
    plotWaveform(audio, fs, fileName, numSamples, duration)
    subplot(3, 1, 3);
    plotPowerSpectrum(audio, fs, fileName, numSamples);
end

function plotPowerSpectrum(audio, fs, fileName, numSamples)
    audio_fft = fft(audio);
    power_spectrum = abs(audio_fft(1:floor(numSamples / 2) + 1)).^2;
    freq = (0:floor(numSamples / 2)) * (fs / numSamples);
    plot(freq, 10 * log10(power_spectrum));
    title(['Power Spectrum of ', fileName]);
    xlabel('Frequency (Hz)');
    ylabel('Power (dB)');
    xlim([0 300]);
end

function plotWaveform(audio, fs, fileName, numSamples, duration)
    time = (0:numSamples - 1) / fs;
    plot(time, audio);
    title(['Waveform of ', fileName])
    xlabel('Time (s)');
    ylabel('Amplitude');
    sound(audio, fs);
    pause(duration + 1)
end

function plotSpectrogram (audio, fs, fileName)
    spectrogram(audio, hamming(960), 240, 1024, fs, 'yaxis');
    title(['Spectrogram of ', fileName]);
end