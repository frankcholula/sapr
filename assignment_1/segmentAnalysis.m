function segmentAnalysis(audio, fs, lpcOrder, filename)
    segmentLengths_ms = 10:10:150;
    predictionErrors = zeros(1, length(segmentLengths_ms));
    for i = 1:length(segmentLengths_ms)
        segmentDuration = segmentLengths_ms(i) / 1000;
        segment = extractCenterSegment(audio, fs, segmentDuration, filename);
        [~, predictionError] = lpc(segment, lpcOrder);
        predictionErrors(i) = predictionError;
    end
    plot(segmentLengths_ms, predictionErrors, '-o');
    title(['LPC Prediction Error vs Segment Length of ', filename]);
    xlabel('Segment Length (ms)');
    ylabel('Prediction Error (Variance)');
    grid on;
end