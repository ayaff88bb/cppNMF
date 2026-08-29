function plot_factor_matrices(Wks, Hks, stft_frequency, stft_time)
%PLOT_FACTOR_MATRICES 绘制各层基矩阵和时间激活矩阵。

layer_count = numel(Wks);
for layer = 1:layer_count
    W = Wks{layer};
    figure;
    if layer == 1
        for component = 1:size(W, 2)
            subplot(size(W, 2), 1, component);
            plot(stft_frequency, W(:, component));
            xlabel('Frequency (Hz)');
            ylabel(['W1(', num2str(component), ')']);
            title(['Basis ', num2str(component)]);
        end
        sgtitle('Layer 1 - Basis Matrix W1');
    else
        imagesc(W);
        colorbar;
        xlabel(['Coefficient index (Layer ', num2str(layer-1), ')']);
        ylabel(['Coefficient index (Layer ', num2str(layer), ')']);
        sgtitle(['Layer ', num2str(layer), ...
            ' - Basis Combination Matrix W', num2str(layer)]);
    end

    H = Hks{layer};
    figure;
    for component = 1:size(H, 1)
        subplot(size(H, 1), 1, component);
        plot(stft_time, H(component, :));
        xlabel('Time (s)');
        ylabel(['H', num2str(layer), '(', num2str(component), ')']);
        title(['Activation ', num2str(component)]);
    end
    sgtitle(['Layer ', num2str(layer), ' - Time Activations H', num2str(layer)]);
end
end

