function orderAnalysis(segment, minOrder, maxOrder)
    predictionErrors = zeros(1, maxOrder - minOrder + 1);
    for order = minOrder:maxOrder
        [~, predictionError] = lpc(segment, order);
        predictionErrors(order - minOrder + 1) = predictionError;
    end
    figure;
    plot(minOrder:maxOrder, predictionErrors, '-o');
    title('LPC Prediction Error as a Function of LPC Order');
    xlabel('LPC Order');
    ylabel('Prediction Error (Variance)');
    grid on;
end
