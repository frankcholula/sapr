function [formants, f0] = plotLPCResponse(segment, fs, lpcOrder)
    lpcCoeffs = lpc(segment, lpcOrder);
    [h, f] = freqz(1, lpcCoeffs, 1024, fs);
    
    segmentFFT = fft(segment);
    segmentPowerSpectrum = abs(segmentFFT(1:floor(length(segment) / 2) + 1)).^2;
    freqAxis = (0:floor(length(segment) / 2)) * (fs / length(segment));
    
    figure;
    plot(freqAxis, pow2db(segmentPowerSpectrum), 'b');
    hold on;
   
    plot(f, mag2db(abs(h)), 'r', 'LineWidth', 2);
    title(['Amplitude Spectrum and LPC Filter Response (Order ', num2str(lpcOrder), ')']);
    xlabel('Frequency (Hz)');
    ylabel('Amplitude (dB)');
    legend('Amplitude Spectrum', 'LPC Filter Response');
    grid on;

     % Estimate the formants (first three peaks in LPC filter response)
    [peaks, locs] = findpeaks(mag2db(abs(h)), f, 'SortStr', 'descend');
    formants = sort(locs(1:3));
    fprintf('Estimated Formant Frequencies (Hz):\n');
    fprintf('F1: %.2f Hz\nF2: %.2f Hz\nF3: %.2f Hz\n', formants(1), formants(2), formants(3));

    % Estimate the fundamental frequency (first peak in the amplitude spectrum)
    dcComponent = segmentPowerSpectrum(1);
    fprintf('DC Component: %.2f dB\n', 10 * log10(dcComponent));
    powerSpectrumNoDC = segmentPowerSpectrum(2:end);
    [peakValue, f0IndexNoDC] = max(powerSpectrumNoDC);
    f0Index = f0IndexNoDC + 1;
    f0 = freqAxis(f0Index);
    fprintf('Estimated F0: %.2f Hz\n', f0);
end
