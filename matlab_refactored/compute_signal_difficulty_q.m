function [q_nsd, q_components] = compute_signal_difficulty_q(x, fs, TF_mag)
%COMPUTE_SIGNAL_DIFFICULTY_Q 计算信号难度指标 q_NSD。
%   指标权重、周期频段、平坦度频段及熵阶数均采用原程序固定值。

weights = [0.4, 0.4, 0.2];
period_freq_range = [5, 300];
flatness_band = [5, 1000];
tf_entropy_power = 2;

w = max(weights(:).', 0);
w = w / (sum(w) + eps);

E_TF = compute_tf_entropy(TF_mag, tf_entropy_power);
C_per = compute_envelope_period_clarity(x, fs, period_freq_range);
D_per = 1 - C_per;
F_flat = compute_envelope_spectral_flatness(x, fs, flatness_band);

q_nsd = w(1) * E_TF + w(2) * D_per + w(3) * F_flat;
q_nsd = min(max(q_nsd, 0), 1);

q_components = struct();
q_components.E_TF = E_TF;
q_components.C_per = C_per;
q_components.D_per = D_per;
q_components.F_flat = F_flat;
q_components.weights = w;
q_components.period_freq_range = period_freq_range;
q_components.flatness_band = flatness_band;
end

function E_TF = compute_tf_entropy(TF_mag, power_order)
P = abs(TF_mag(:)).^power_order;
P(~isfinite(P)) = 0;
Psum = sum(P);

if Psum <= 0
    E_TF = 0;
    return;
end

p = P / (Psum + eps);
idx = p > 0;
E_TF = -sum(p(idx) .* log(p(idx))) / log(numel(p) + eps);
E_TF = min(max(E_TF, 0), 1);
end

function C_per = compute_envelope_period_clarity(x, fs, freq_range)
x = x(:);
x(~isfinite(x)) = 0;
x = x - mean(x);

if numel(x) < 8 || norm(x) <= eps
    C_per = 0;
    return;
end

env = abs(hilbert(x));
env = env - mean(env);
if norm(env) <= eps
    C_per = 0;
    return;
end

f_min = max(min(freq_range), eps);
f_max = max(freq_range);
lag_min = max(1, floor(fs / f_max));
lag_max = min(numel(env) - 1, ceil(fs / f_min));

if lag_max <= lag_min
    C_per = 0;
    return;
end

acf = xcorr(env, lag_max, 'coeff');
acf_pos = acf(lag_max+1:end);
segment = acf_pos((lag_min+1):(lag_max+1));
segment = segment(isfinite(segment));

if isempty(segment)
    C_per = 0;
else
    C_per = min(max(max(segment), 0), 1);
end
end

function F_flat = compute_envelope_spectral_flatness(x, fs, freq_band)
x = x(:);
x(~isfinite(x)) = 0;
x = x - mean(x);

if numel(x) < 8 || norm(x) <= eps
    F_flat = 1;
    return;
end

env = abs(hilbert(x));
env = env - mean(env);

N = numel(env);
NFFT = 2^nextpow2(N);
Y = fft(env, NFFT);
P = abs(Y(1:floor(NFFT/2)+1)).^2;
f = (0:floor(NFFT/2))' * (fs / NFFT);

idx = f >= freq_band(1) & f <= freq_band(2);
Pband = P(idx);
Pband = Pband(isfinite(Pband) & Pband > 0);

if isempty(Pband)
    F_flat = 1;
    return;
end

F_flat = exp(mean(log(Pband + eps))) / (mean(Pband + eps) + eps);
F_flat = min(max(F_flat, 0), 1);
end

