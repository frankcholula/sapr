function plotLPCResponse(segment, fs, lpcOrder)    
    % Plot the power spectrum and LPC response for reference
    [lpcCoeffs, predictionError] = lpc(segment, lpcOrder);
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
end
