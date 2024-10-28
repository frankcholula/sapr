function plotLPCPoleZero(segment, fs, lpcOrder)
    lpcCoeffs = lpc(segment, lpcOrder);
    [h, f] = freqz(1, lpcCoeffs, 1024, fs);

    % Plot frequency response (Amplitude Spectrum)
    figure;
    subplot(2, 1, 1);
    plot(f, mag2db(abs(h)), 'r', 'LineWidth', 2);
    title(['LPC Filter Response (Order ', num2str(lpcOrder), ')']);
    xlabel('Frequency (Hz)');
    ylabel('Amplitude (dB)');
    grid on;

    % Pole-Zero Plot
    subplot(2, 1, 2);
    zplane(roots(1), roots(lpcCoeffs));
    title('Pole-Zero Plot');
    xlabel('Real Part');
    ylabel('Imaginary Part');
    grid on;
end