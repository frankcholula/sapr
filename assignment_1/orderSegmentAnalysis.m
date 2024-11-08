function orderSegmentAnalysis(signal, fs, filename)
    segmentLengths_ms = 10:10:150;
    lpcOrders = 10:30;
    predictionErrors = zeros(length(segmentLengths_ms), length(lpcOrders));

    for i = 1:length(segmentLengths_ms)
        segmentDuration = segmentLengths_ms(i) / 1000;
        segment = extractCenterSegment(signal, fs, segmentDuration, filename);
        for j = 1:length(lpcOrders)
            order = lpcOrders(j);
            [~, predictionError] = lpc(segment, order);
            predictionErrors(i, j) = predictionError;
        end
    end

    % Plot the prediction error as a 3D surface plot
    figure;
    [X, Y] = meshgrid(lpcOrders, segmentLengths_ms);
    surf(X, Y, predictionErrors);
    title(['LPC Prediction Error vs. Segment Length vs. LPC Order of ', filename]);
    xlabel('LPC Order');
    ylabel('Segment Length (ms)');
    zlabel('Prediction Error (Variance)');
    colorbar;
    grid on;

    % Plot the prediction error as a heatmap
    figure;
    imagesc(lpcOrders, segmentLengths_ms, predictionErrors);
    title(['Heatmap of ', filename]);
    xlabel('LPC Order');
    ylabel('Segment Length (ms)');
    colorbar;
    axis xy;
end
