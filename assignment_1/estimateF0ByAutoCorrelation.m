function [f0_estimate] = estimateF0ByAutoCorrelation(segment, fs, filename)
    % smooth the segment first
    segment = movmean(segment, 5);
    [autocorrValues, lags] = xcorr(segment, 'coeff');
    posLagIdx = lags >= 0;
    lags = lags(posLagIdx);
    autocorrValues = autocorrValues(posLagIdx);
    figure();
    plot(lags, autocorrValues);
    title(['Autocorrelation Function of ', filename]);
    xlabel('Lag (samples)');
    ylabel('Normalized Autocorrelation');
    xlim([0 1000]);
    [peakValues, locs] = findpeaks(autocorrValues, 'MinPeakHeight', 0.3, 'MinPeakDistance', fs / 90, 'MinPeakProminence', 0.2);
    
    if isempty(peakValues)
        f0_estimate = NaN; % Return NaN if no peaks are found
        disp('No peaks found for F0 estimation');
        return;
    end

    % Find the maximum peak value and its location
    [maxPeak, maxPeakIdx] = max(peakValues);
    peakLag = lags(locs(maxPeakIdx));

    % Convert lag of the maximum peak to frequency
    f0_estimate = fs / peakLag;
    fprintf('Estimated F0 using autocorrelation: %.2f Hz\n', f0_estimate);
end