function lpcCoeffs = estimateLPC(segment, lpcOrder)
    lpcCoeffs = lpc(segment, lpcOrder);
    fprintf('LPC Coefficients (order %d):\n', lpcOrder);
    disp(lpcCoeffs);
end
