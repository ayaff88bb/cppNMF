function [x_sub_all, env_sub_all] = reconstruct_subspaces( ...
        W_rec_top, H_top, original_magnitude, phase_positive, ...
        zero_row, signal_data, window, noverlap, N_stft)
%RECONSTRUCT_SUBSPACES 用分量掩膜从原始 STFT 幅值重构各子空间。
%   掩膜指数 p_mask 和数值稳定常数沿用原程序固定值。

p_mask = 1;
eps_mask = 1e-12;

subspace_count = size(H_top, 1);
parts = cell(1, subspace_count);
part_sum = zeros(size(original_magnitude));

for subspace = 1:subspace_count
    parts{subspace} = max( ...
        W_rec_top(:, subspace) * H_top(subspace, :), 0);
    part_sum = part_sum + parts{subspace};
end

x_sub_all = cell(1, subspace_count);
env_sub_all = cell(1, subspace_count);

for subspace = 1:subspace_count
    mask = (parts{subspace}.^p_mask) ./ ...
        (part_sum.^p_mask + eps_mask);
    subspace_magnitude = mask .* original_magnitude;

    reconstructed = reconstruct_signal_from_magnitude( ...
        subspace_magnitude, phase_positive, zero_row, signal_data.fs, ...
        window, noverlap, N_stft, signal_data.length);
    x_sub_all{subspace} = reconstructed;

    envelope = abs(hilbert(reconstructed));
    envelope_spectrum = abs(fft(envelope - mean(envelope))) ...
        / signal_data.length;
    env_sub_all{subspace} = envelope_spectrum;

    figure;
    subplot(2, 1, 1);
    plot(signal_data.time, reconstructed);
    title(['MO-Masked Reconstructed Time-domain Signal - Subspace ', ...
        num2str(subspace)]);
    xlabel('Time (s)');
    ylabel('Amplitude');

    subplot(2, 1, 2);
    plot(signal_data.frequency_half, ...
        envelope_spectrum(1:signal_data.half_length));
    xlim([0, 300]);
    xlabel('Frequency (Hz)');
    ylabel('Amplitude');
    title('Envelope Spectrum');
end
end

