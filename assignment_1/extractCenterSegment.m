function segment = extractCenterSegment(audio, fs, segmentDuration, filename)
    numSegmentSamples = round(segmentDuration * fs);

    numSamples = length(audio);
    centerIndex = round(numSamples / 2);
    startIndex = centerIndex - floor(numSegmentSamples / 2);
    endIndex = startIndex + numSegmentSamples - 1;

    segment = audio(startIndex:endIndex);
    % plotCenterSegment(segment, fs, filename, numSegmentSamples, segmentDuration);
end

function plotCenterSegment(segment, fs, filename, numSegmentSamples, segmentDuration)
    figure;
    timeSegment = (0:numSegmentSamples - 1) / fs;
    plot(timeSegment, segment);
    title([num2str(segmentDuration*1000), 'ms Quasi-Stationary Segment of ', filename]);
    xlabel('Time (s)');
    ylabel('Amplitude');
end