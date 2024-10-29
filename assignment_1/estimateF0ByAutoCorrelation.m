function [f0_estimate] = estimateF0ByAutoCorrelation(segment, fs)
    % Compute autocorrelation and find peaks
    [autocorrValues, lags] = xcorr(segment, 'coeff');
    posLagIdx = lags >= 0;
    lags = lags(posLagIdx);
    autocorrValues = autocorrValues(posLagIdx);
    figure()
    plot(lags, autocorrValues);
    xlim([0 500]);
    [peakValues, locs] = findpeaks(autocorrValues);

    if isempty(peakValues)
        f0_estimate = NaN; % Return NaN if no peaks are found
        disp('No peaks found for F0 estimation');
        return;
    end

    % Find the maximum peak value and its location
    [maxPeak, maxPeakIdx] = max(peakValues);
    peakLag = lags(locs(maxPeakIdx)); % Get the lag corresponding to the max peak

    % Convert lag of the maximum peak to frequency
    f0_estimate = fs / peakLag;
    fprintf('Estimated F0 using autocorrelation: %.2f Hz\n', f0_estimate);
end