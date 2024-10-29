function f0 = estimateF0ByPowerSpectrum(segment, fs)
    N = length(segment);
    segmentFFT = fft(segment);
    segmentPowerSpectrum = abs(segmentFFT(1:floor(N/2)+1)).^2;
    freqAxis = (0:floor(N/2)) * (fs / N);

    powerSpectrumNoDC = segmentPowerSpectrum(2:end);
    freqAxisNoDC = freqAxis(2:end);
    [peaks, locs] = findpeaks(powerSpectrumNoDC, freqAxisNoDC, 'MinPeakHeight', max(powerSpectrumNoDC) * 0.1);
    if isempty(peaks)
        f0 = NaN;
        fprintf('No significant peak found for F0 estimation.\n');
        return;
    end
    f0 = locs(1);
    fprintf('Estimated F0 using power spectrum: %.2f Hz\n', f0);
end