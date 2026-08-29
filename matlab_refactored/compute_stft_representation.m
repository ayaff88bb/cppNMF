function data = compute_stft_representation(signal, fs, N_stft, window, hop)
%COMPUTE_STFT_REPRESENTATION 计算后续分解和重构需要的 STFT 数据。

if ~isscalar(N_stft) || N_stft <= 0 || N_stft ~= fix(N_stft) || mod(N_stft, 2) ~= 0
    error('compute_stft_representation:InvalidFFTLength', ...
        'N_stft 必须是正偶整数。');
end
if ~isvector(window) || isempty(window) || length(window) > N_stft
    error('compute_stft_representation:InvalidWindow', ...
        '窗必须是非空向量，且长度不能超过 N_stft。');
end
if ~isscalar(hop) || hop <= 0 || hop ~= fix(hop) || hop > length(window)
    error('compute_stft_representation:InvalidHop', ...
        'hop 必须是 1 到窗长之间的整数。');
end

window = window(:);
noverlap = length(window) - hop;
original_length = length(signal);
pad_len = mod(-(original_length - length(window)), hop);
signal_pad = [signal; zeros(pad_len, 1)];

if ~iscola(window, noverlap)
    warning('compute_stft_representation:NonCOLA', ...
        '当前窗和重叠长度不满足 COLA 条件，ISTFT 可能无法完美重构。');
end

[stft_full, frequency_full, time] = stft(signal_pad, fs, ...
    "Window", window, ...
    "OverlapLength", noverlap, ...
    "FFTLength", N_stft, ...
    "FrequencyRange", "centered");

positive_start = N_stft / 2 + 1;
zero_row = stft_full(N_stft / 2, :);

full_magnitude = abs(stft_full);
full_phase = ones(size(stft_full));
nonzero = full_magnitude > 0;
full_phase(nonzero) = stft_full(nonzero) ./ full_magnitude(nonzero);

data = struct();
data.magnitude = full_magnitude(positive_start:end, :);
data.frequency = frequency_full(positive_start:end, :);
data.time = time;
data.phase_positive = full_phase(positive_start:end, :);
data.zero_row = zero_row;
data.noverlap = noverlap;
data.original_length = original_length;
data.padding_length = pad_len;
end

