function [audio, fs, duration, numSamples] = playAudio(fileName)
    filePath = strcat('speech/',fileName);
    [audio, fs] = audioread(filePath);
    duration = length(audio) / fs;
    numSamples = length(audio);
    fprintf('Playing %s at %d Hz with duration %.2f seconds and %d samples\n',fileName, fs, duration, numSamples);
    
    % plot the spectrogram
    figure;
    subplot(3,1,1);
    spectrogram(audio, hamming(960), 240, 1024, fs, 'yaxis');
    title(['Spectrogram of ', fileName]);
    
    % plot the waveform
    subplot(3,1,2);
    time = (0:numSamples - 1) / fs;
    plot(time, audio);
    title(['Waveform of ', fileName])
    sound(audio, fs);
    pause(duration + 1)

    % plot the spectrum as well
    subplot(3, 1, 3);
    audio_fft = fft(audio);
    power_spectrum = abs(audio_fft(1:floor(numSamples / 2) + 1)).^2;
    freq = (0:floor(numSamples / 2)) * (fs / numSamples);
    plot(freq, 10 * log10(power_spectrum));
    title(['Power Spectrum of ', fileName]);
    xlabel('Frequency (Hz)');
    ylabel('Power (dB)');
    xlim([0 300]);
end
