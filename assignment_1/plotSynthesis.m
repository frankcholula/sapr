function plotSynthesis(segment, fs, synthesizedImpulse, synthesizedSawtooth)
    numSamples = length(synthesizedImpulse);
    timeAxis = (0:numSamples - 1) / fs;
    originalTimeAxis = (0:length(segment) - 1) / fs;

    figure;
    subplot(3, 1, 1);
    plot(originalTimeAxis, segment);
    title('Original Segment');
    xlabel('Time (s)');
    ylabel('Amplitude');
    xlim([0, 0.05]);

    subplot(3, 1, 2);
    plot(timeAxis, synthesizedImpulse);
    title('Synthesized Signal (Impulse Train)');
    xlabel('Time (s)');
    ylabel('Amplitude');
    xlim([0, 0.05]);

    subplot(3, 1, 3);
    plot(timeAxis, synthesizedSawtooth);
    title('Synthesized Signal (Sawtooth Wave)');
    xlabel('Time (s)');
    ylabel('Amplitude');
    xlim([0, 0.05]);
end