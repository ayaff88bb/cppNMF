function signal = reconstruct_signal_from_magnitude( ...
        magnitude, phase_positive, zero_row, fs, window, ...
        noverlap, N_stft, original_length)
%RECONSTRUCT_SIGNAL_FROM_MAGNITUDE 使用原始相位执行 ISTFT 重构。

if ~isequal(size(magnitude), size(phase_positive))
    error('reconstruct_signal_from_magnitude:SizeMismatch', ...
        'magnitude 与 phase_positive 的尺寸必须一致。');
end

positive = magnitude .* phase_positive;
negative = flipud(conj(positive(1:end-1, :)));
full_stft = [negative; zero_row; positive];

reconstructed = istft(full_stft, fs, ...
    "Window", window, ...
    "OverlapLength", noverlap, ...
    "FFTLength", N_stft);

if numel(reconstructed) < original_length
    error('reconstruct_signal_from_magnitude:ShortISTFTOutput', ...
        'ISTFT 输出长度 %d 小于原始信号长度 %d。', ...
        numel(reconstructed), original_length);
end
signal = reconstructed(1:original_length);
end

