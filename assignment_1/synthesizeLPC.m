function [synthesizedImpulse, synthesizedSawtooth] = synthesizeLPC(segment, fs, f0, lpcOrder, duration, filename)
    [lpcCoeffs, predictionError] = lpc(segment, lpcOrder);

    T0 = round(fs / f0);
    numSamples = round(duration * fs);
    impulseSignal = zeros(1, numSamples);
    impulseSignal(1:T0:end) = 1;

    t = (0:numSamples-1) / fs;
    sawtoothSignal = sawtooth(2 * pi * f0 * t);
    
    figure;
    subplot(2, 1, 1);
    timeAxis = (0:numSamples - 1) / fs;
    plot(timeAxis, impulseSignal);
    title(['Impulse Train Excitation with F0 = ', num2str(f0), ' Hz']);
    xlabel('Time (s)');
    ylabel('Amplitude');
    xlim([0, 0.05]);

    subplot(2, 1, 2);
    plot(timeAxis, sawtoothSignal);
    title(['Sawtooth Wave Excitation with F0 = ', num2str(f0), ' Hz']);
    xlabel('Time (s)');
    ylabel('Amplitude');
    xlim([0, 0.05]);

    originalPeak = max(abs(segment));
    synthesizedImpulse = filter(1, lpcCoeffs, impulseSignal);
    synthesizedImpulsePeak = max(abs(synthesizedImpulse));
    synthesizedImpulse = synthesizedImpulse * (originalPeak/ synthesizedImpulsePeak);
    sound(synthesizedImpulse, fs);
    pause(duration + 1);
    audiowrite(['synthesized_impulse/' filename], synthesizedImpulse, fs);


    synthesizedSawtooth = filter(1, lpcCoeffs, sawtoothSignal);
    synthesizedSawtoothPeak = max(abs(synthesizedSawtooth));
    synthesizedSawtooth = synthesizedSawtooth * (originalPeak/ synthesizedSawtoothPeak);
    sound(synthesizedSawtooth, fs);
    pause(duration + 1);
    audiowrite(['synthesized_sawtooth/' filename], synthesizedSawtooth, fs);
end
