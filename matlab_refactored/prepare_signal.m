function data = prepare_signal(signal, fs0, signal_channel)
%PREPARE_SIGNAL 选择通道、清理非有限值并计算基础频谱。

if ~isnumeric(signal) || isempty(signal) || ~ismatrix(signal)
    error('prepare_signal:InvalidSignal', 'signal 必须是非空数值向量或二维矩阵。');
end
if ~isscalar(fs0) || ~isnumeric(fs0) || ~isfinite(fs0) || fs0 <= 0
    error('prepare_signal:InvalidSamplingRate', 'fs0 必须是正的有限标量。');
end
if ~isscalar(signal_channel) || signal_channel < 1 || ...
        signal_channel ~= fix(signal_channel) || signal_channel > size(signal, 2)
    error('prepare_signal:InvalidChannel', ...
        'signal_channel 必须是 1 到 %d 之间的整数。', size(signal, 2));
end

x = signal(:, signal_channel);
x(~isfinite(x)) = 0;
x = x - mean(x);

N = length(x);
if N < 2
    error('prepare_signal:SignalTooShort', '输入信号至少需要 2 个采样点。');
end

time = (0:N-1)' / fs0;
N_half = floor(N / 2);
frequency_half = (0:N_half-1)' * (fs0 / N);

spectrum = abs(fft(x)) / N;
spectrum = spectrum(1:N_half);
if numel(spectrum) >= 2
    spectrum(2:end) = 2 * spectrum(2:end);
end

envelope = abs(hilbert(x));
envelope_spectrum = abs(fft(envelope - mean(envelope))) / N;
envelope_spectrum = envelope_spectrum(1:N_half);
if numel(envelope_spectrum) >= 2
    envelope_spectrum(2:end) = 2 * envelope_spectrum(2:end);
end

fprintf('\n===== Input signal =====\n');
fprintf('fs = %.6f Hz\n', fs0);
fprintf('N  = %d samples\n', N);
fprintf('duration = %.6f s\n', N / fs0);

data = struct();
data.signal = x;
data.fs = fs0;
data.length = N;
data.time = time;
data.half_length = N_half;
data.frequency_half = frequency_half;
data.spectrum = spectrum;
data.envelope_spectrum = envelope_spectrum;
end
