function synthesizeVowelWithGlottalSource(segment, fs, f0, lpcOrder, duration)
    % synthesizeVowelWithGlottalSource synthesizes a vowel sound using LPC coefficients
    % and a glottal pulse train as the source.
    %
    % Inputs:
    %   segment - audio segment for LPC analysis
    %   fs - sampling rate of the audio segment
    %   f0 - fundamental frequency (F0) in Hz
    %   lpcOrder - order of the LPC model
    %   duration - duration of the synthesized sound in seconds

    % Step 1: Estimate LPC coefficients from the segment
    lpcCoeffs = lpc(segment, lpcOrder);

    % Step 2: Generate a glottal pulse train with the Rosenberg model
    T0 = round(fs / f0); % Period of glottal pulse in samples
    numSamples = round(duration * fs);
    glottalPulse = generateGlottalPulse(T0); % Generate one cycle of glottal pulse
    glottalSource = repmat(glottalPulse, 1, ceil(numSamples / length(glottalPulse)));
    glottalSource = glottalSource(1:numSamples); % Trim to required duration

    % Plot the glottal pulse for visualization
    figure;
    timeAxis = (0:length(glottalPulse) - 1) / fs;
    plot(timeAxis, glottalPulse);
    title(['Glottal Pulse with Fundamental Frequency F0 = ', num2str(f0), ' Hz']);
    xlabel('Time (s)');
    ylabel('Amplitude');

    % Step 3: Filter the glottal source with the LPC filter
    synthesizedSignal = filter(1, lpcCoeffs, glottalSource);

    % Step 4: Play the synthesized sound
    sound(synthesizedSignal, fs);
    pause(duration + 1); % Pause to ensure playback completes
end

function glottalPulse = generateGlottalPulse(T0)
    % generateGlottalPulse creates a single glottal pulse based on the Rosenberg model.
    %
    % Input:
    %   T0 - period of the pulse in samples (depends on F0 and fs)
    % Output:
    %   glottalPulse - a single glottal pulse waveform

    % Define phase durations (portion of T0 for each phase)
    openPhase = 0.6;   % Opening phase (60% of T0)
    closingPhase = 0.3; % Closing phase (30% of T0)
    closedPhase = 0.1; % Closed phase (10% of T0)

    % Number of samples for each phase
    nOpen = round(T0 * openPhase);
    nClose = round(T0 * closingPhase);
    nClosed = T0 - nOpen - nClose;

    % Create the glottal pulse
    glottalPulse = [linspace(0, 1, nOpen), linspace(1, 0, nClose), zeros(1, nClosed)];
end