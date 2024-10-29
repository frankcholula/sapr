function [formants] = estimateFormants(segment, fs, lpcOrder)
    lpcCoeffs = lpc(segment, lpcOrder);
    poles = roots(lpcCoeffs);
    poles = poles(abs(poles) < 1 & imag(poles) ~= 0);

    angles = angle(poles);
    frequencies = abs(angles * (fs / (2 * pi)));

    uniqueFrequencies = unique(sort(frequencies));
    formants = uniqueFrequencies(1:min(3, length(uniqueFrequencies)));  % First three formants
    fprintf('Estimated Formant Frequencies (Hz):\n');
    for i = 1:length(formants)
        fprintf('F%d: %.2f Hz\n', i, formants(i));
    end
end