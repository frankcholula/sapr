function plotSpectrogram (audio, fs, fileName)
    colormap('gray');
    map = colormap;
    imap = flipud(map);
    window_size = round(0.02 * fs);
    overlap = floor(window_size / 2);
    N = 2^nextpow2(4 * window_size);
    fprintf('Window Size: %d samples (%.2f ms)\n', window_size, (window_size / fs) * 1000);
    fprintf('Overlap: %d samples (%.2f ms)\n', overlap, (overlap / fs) * 1000);
    fprintf('FFT Length (N): %d' , N);
    spectrogram(audio, hamming(window_size), overlap, N, fs, 'yaxis');
    colormap(imap);
    clim([-60, -30]); % Adjust dynamic range for more distinct blacks
    % colorbar('off');
    title(['Spectrogram of ', fileName]);
    ylim([0, 4]);
end