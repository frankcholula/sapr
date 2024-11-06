function [audio, fs, duration, numSamples] = playAudio(fileName)
    filePath = strcat('speech/',fileName);
    [audio, fs] = audioread(filePath);
    duration = length(audio) / fs;
    numSamples = length(audio);
    sound(audio, fs);
    fprintf('Playing %s at %d Hz with duration %.2f seconds and %d samples\n',fileName, fs, duration, numSamples);
    
    figure();
    % subplot(2,1,1);
    % plotWaveform(audio, fs, fileName, numSamples, duration)
    % subplot(2,1,2);
    plotPowerSpectrum(audio, fs, fileName, numSamples);
end

function plotPowerSpectrum(audio, fs, fileName, numSamples)
    audio_fft = fft(audio);
    power_spectrum = abs(audio_fft(1:floor(numSamples / 2) + 1)).^2;
    freq = (0:floor(numSamples / 2)) * (fs / numSamples);
    plot(freq, pow2db(power_spectrum));
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
    pause(duration + 1)
end
