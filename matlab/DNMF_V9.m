%% Deep_nsNMF_MO_input_cleaned_minimal.m
% 精简版 Deep nsNMF + MO 输入 + q_NSD 自适应 theta + 掩膜式子空间重构
%   1) 读取单组信号；
%   2) STFT 幅值矩阵；
%   3) q_NSD 信号难度指标；
%   4) MO 输入幅值压缩；
%   5) q_NSD -> 中层敏感 sigmoid 自适应 theta；
%   6) Deep nsNMF 分解；
%   7) MO 空间子空间掩膜 -> 原始 STFT 幅值 -> 原始相位 ISTFT；
%   8) 子空间时域波形、包络谱、顶层 H 频谱绘制。

%% 初始化
clc;  close all;
clearvars -except signal fs0
addpath(genpath(pwd));

%% ========================= 信号处理参数 ====================
% STFT 参数
stftoptions.N_stft = 1024;  %重要
win_len = 120;  %重要
hop = 20;  %重要
stftoptions.win = hamming(win_len, "periodic"); %重要
stftoptions.noverlap = length(stftoptions.win) - hop;

% Deep nsNMF 秩
inner_ranks = [40, 16, 4];  %重要

% MO 输入压缩
use_input_mo = true; %重要
mo_gamma = 0.6;  %重要

% 自适应 theta
use_adaptive_theta = 0;
manual_theta = [0.1, 0.4, 0.7];  %重要

adaptive_theta_model = struct();
adaptive_theta_model.name   = 'middle_sensitive_sigmoid_q_d_psi';
adaptive_theta_model.q_min  = 0.381664;
adaptive_theta_model.q_max  = 0.863117;
adaptive_theta_model.beta0  = -2.378484;
adaptive_theta_model.beta_d =  3.330215;
adaptive_theta_model.beta_q =  0.244229;
adaptive_theta_model.beta_m =  0.303702;

% DNMF 参数
options.fastapprox = 1;          % 1: NNDSVD 初始化；0: 随机初始化
options.verbose = 1;
options.tolfun = 1e-3;
options.finetune_maxiter = 800;
options.inner_mu = 10;
options.eps = 1e-12;

% 顶层前几阶谐波完整性奖励 rep
% 若不想使用周期奖励，直接设为 0。
options.lambda_rep = 1e-2;  %重要
options.rep_lags = 5:80;  
options.rep_weights = [];
options.rep_temperature = 0.5;
options.harmonic_M = 3; 
options.harmonic_weights = [];
options.eps_rep = 1e-12;

% 子空间掩膜锐化指数
% p_mask=1 较软；p_mask=1.5 折中；p_mask=2 较硬。
p_mask = 1;
eps_mask = 1e-12;

% 绘图与保存
plot_initial_signal = true;
plot_factor_matrices = true;
plot_subspaces = true;
plot_top_H_spectrum = true;
result_file = 'Deep_nsNMF_result_cleaned_minimal.mat';

%% ========================= 信号参数 =========================
signal_channel = 1;  %重要
signal = signal(:,signal_channel);
signal(~isfinite(signal)) = 0;
fs = fs0;
N = length(signal);
t = (0:N-1)' / fs;
N_half = floor(N / 2);
detf_half = (0:N_half-1)' * (fs / N);

fprintf('\n===== Input signal =====\n');
fprintf('fs = %.6f Hz\n', fs);
fprintf('N  = %d samples\n', N);
fprintf('duration = %.6f s\n', N / fs);

%% ========================= 初始时域、频谱与包络谱 =========================
signal = signal - mean(signal);

fft_sig = abs(fft(signal)) / N;
fft_sig = fft_sig(1:N_half);
if numel(fft_sig) >= 2
    fft_sig(2:end) = 2 * fft_sig(2:end);
end

sig_hil = abs(hilbert(signal));
sig_env = abs(fft(sig_hil - mean(sig_hil))) / N;
sig_env = sig_env(1:N_half);
if numel(sig_env) >= 2
    sig_env(2:end) = 2 * sig_env(2:end);
end

if plot_initial_signal
    figure;
    plot(t, signal);
    xlabel('Time/s');
    ylabel('Amplitude');
    title('Time-domain Signal');

    figure;
    subplot(2,1,1);
    plot(detf_half, fft_sig);
    xlabel('Frequency/Hz');
    ylabel('Amplitude');
    title('Spectrum');

    subplot(2,1,2);
    plot(detf_half, sig_env);
    xlabel('Frequency/Hz');
    ylabel('Amplitude');
    title('Envelope Spectrum');
    linkaxes(findobj(gcf, 'Type', 'axes'), 'x');
end

%% ========================= STFT =========================
N0 = length(signal);
pad_len = mod(-(N0 - win_len), hop);
signal_pad = [signal; zeros(pad_len, 1)];

if ~iscola(stftoptions.win, stftoptions.noverlap)
    warning('当前窗和重叠长度不满足 COLA 条件，ISTFT 可能无法完美重构。');
end

[sig_stft_full, stft_f_full, stft_t_full] = stft(signal_pad, fs, ...
    "Window", stftoptions.win, ...
    "OverlapLength", stftoptions.noverlap, ...
    "FFTLength", stftoptions.N_stft, ...
    "FrequencyRange", "centered");

idx0 = stftoptions.N_stft / 2 + 1;
X_zero = sig_stft_full(stftoptions.N_stft / 2, :);

sig_stft_mag = abs(sig_stft_full);
sig_stft_phase = ones(size(sig_stft_full));
notzero = sig_stft_mag > 0;
sig_stft_phase(notzero) = sig_stft_full(notzero) ./ sig_stft_mag(notzero);

sig_stft = sig_stft_mag(idx0:end, :);
stft_f = stft_f_full(idx0:end, :);
phase_pos = sig_stft_phase(idx0:end, :);
TF_mag = sig_stft;

figure;
imagesc(stft_t_full, stft_f, TF_mag);
axis xy;
xlabel('Time/s');
ylabel('Frequency/Hz');
title('Original STFT Magnitude');
colorbar;

%% ========================= q_NSD 信号难度指标 =========================
qopt = struct();
qopt.weights = [0.4, 0.4, 0.2];
qopt.period_freq_range = [5, 300];
qopt.flatness_band = [5, 1000];
qopt.tf_entropy_power = 2;

[q_nsd, q_components] = compute_signal_difficulty_q(signal, fs, TF_mag, qopt);

fprintf('\n===== q_NSD signal difficulty =====\n');
fprintf('E_TF   = %.6f\n', q_components.E_TF);
fprintf('C_per  = %.6f, D_per = %.6f\n', q_components.C_per, q_components.D_per);
fprintf('F_flat = %.6f\n', q_components.F_flat);
fprintf('q_NSD  = %.6f\n', q_nsd);

%% ========================= MO 输入幅值压缩 =========================
X_mag_raw = TF_mag;

if use_input_mo
    X_scale = max(X_mag_raw(:)) + eps;
    X_norm = X_mag_raw / X_scale;
    fea_mo = (X_norm + eps).^mo_gamma;
    fea_mo = fea_mo * norm(X_mag_raw, 'fro') / (norm(fea_mo, 'fro') + eps);
    fea = fea_mo;
else
    fea = X_mag_raw;
end

figure;
imagesc(stft_t_full, stft_f, fea);
axis xy;
xlabel('Time/s');
ylabel('Frequency/Hz');
if use_input_mo
    title(['MO-compressed STFT Magnitude, \gamma = ', num2str(mo_gamma)]);
else
    title('Original STFT Magnitude Used as DNMF Input');
end
colorbar;

%% ========================= 自适应 theta =========================
nbLayers = length(inner_ranks);

[theta_adaptive, q_bar, layer_depth, middle_sensitivity] = ...
    sigmoid_theta_schedule(q_nsd, nbLayers, adaptive_theta_model);

if use_adaptive_theta
    options.thl = theta_adaptive;
else
    options.thl = manual_theta;
end

fprintf('\n===== Theta scheduling =====\n');
fprintf('use_adaptive_theta = %d\n', use_adaptive_theta);
fprintf('model = %s\n', adaptive_theta_model.name);
fprintf('q_bar = %.6f  [q_min=%.6f, q_max=%.6f]\n', ...
    q_bar, adaptive_theta_model.q_min, adaptive_theta_model.q_max);
fprintf('layer depth d = [%s]\n', num2str(layer_depth, ' %.3f'));
fprintf('middle sensitivity psi(d) = [%s]\n', num2str(middle_sensitivity, ' %.3f'));
fprintf('theta_adaptive = [%s]\n', num2str(theta_adaptive, ' %.6f'));
fprintf('Final options.thl = [%s]\n', num2str(options.thl, ' %.6f'));

%% ========================= Deep nsNMF =========================
tic;
[Wks, Hks, S, objhistory] = DOSNMF(fea, inner_ranks, options);
toc;

%% ========================= 分解结果绘图 =========================
if plot_factor_matrices
    for i = 1:nbLayers
        W = Wks{i};

        figure;
        if i == 1
            for k = 1:size(W, 2)
                subplot(size(W, 2), 1, k);
                plot(stft_f, W(:, k));
                xlabel('Frequency (Hz)');
                ylabel(['W1(', num2str(k), ')']);
                title(['Basis ', num2str(k)]);
            end
            sgtitle('Layer 1 - Basis Matrix W1');
        else
            imagesc(W);
            colorbar;
            xlabel(['Coefficient index (Layer ', num2str(i-1), ')']);
            ylabel(['Coefficient index (Layer ', num2str(i), ')']);
            sgtitle(['Layer ', num2str(i), ' - Basis Combination Matrix W', num2str(i)]);
        end

        H = Hks{i};
        figure;
        for k = 1:size(H, 1)
            subplot(size(H, 1), 1, k);
            plot(stft_t_full, H(k, :));
            xlabel('Time (s)');
            ylabel(['H', num2str(i), '(', num2str(k), ')']);
            title(['Activation ', num2str(k)]);
        end
        sgtitle(['Layer ', num2str(i), ' - Time Activations H', num2str(i)]);
    end
end

%% ========================= 多层重构矩阵 =========================
X_rec_layers = cell(1, nbLayers);
W_rec_layers = cell(1, nbLayers);

W_cum = Wks{1};
W_rec_layers{1} = W_cum;
X_rec_layers{1} = W_cum * Hks{1};

for i = 2:nbLayers
    W_cum = W_cum * Wks{i};
    W_rec_layers{i} = W_cum;
    X_rec_layers{i} = W_cum * Hks{i};
end

X_rec_top = X_rec_layers{nbLayers};
W_rec_top = W_rec_layers{nbLayers};
H_top = Hks{nbLayers};

%% ========================= 整体重构，仅用于检查 =========================
X_pos = X_rec_top .* phase_pos;
X_neg = flipud(conj(X_pos(1:end-1, :)));
X_full = [X_neg; X_zero; X_pos];

x_rec_full = istft(X_full, fs, ...
    "Window", stftoptions.win, ...
    "OverlapLength", stftoptions.noverlap, ...
    "FFTLength", stftoptions.N_stft);
x_rec = x_rec_full(1:N0);

%% ========================= 子空间重构：MO 掩膜 -> 原始幅值 -> ISTFT =========================
numSub = size(H_top, 1);
X_parts = cell(1, numSub);
X_sum = zeros(size(X_rec_top));

for i = 1:numSub
    X_parts{i} = max(W_rec_top(:, i) * H_top(i, :), 0);
    X_sum = X_sum + X_parts{i};
end

x_sub_all = cell(1, numSub);
env_sub_all = cell(1, numSub);

for i = 1:numSub
    M_i = (X_parts{i}.^p_mask) ./ (X_sum.^p_mask + eps_mask);
    X_sub_mag = M_i .* sig_stft;

    X_sub_pos = X_sub_mag .* phase_pos;
    X_sub_neg = flipud(conj(X_sub_pos(1:end-1, :)));
    X_sub_full = [X_sub_neg; X_zero; X_sub_pos];

    x_sub_rec_full = istft(X_sub_full, fs, ...
        "Window", stftoptions.win, ...
        "OverlapLength", stftoptions.noverlap, ...
        "FFTLength", stftoptions.N_stft);

    x_sub_rec = x_sub_rec_full(1:N0);
    x_sub_all{i} = x_sub_rec;

    sig_hilbert = abs(hilbert(x_sub_rec));
    sig_env_sub = abs(fft(sig_hilbert - mean(sig_hilbert))) / N;
    env_sub_all{i} = sig_env_sub;

    if plot_subspaces
        figure;
        subplot(2, 1, 1);
        plot(t, x_sub_rec);
        title(['MO-Masked Reconstructed Time-domain Signal - Subspace ', num2str(i)]);
        xlabel('Time (s)');
        ylabel('Amplitude');

        subplot(2, 1, 2);
        plot(detf_half, sig_env_sub(1:N_half));
        xlim([0, 300]);
        xlabel('Frequency (Hz)');
        ylabel('Amplitude');
        title('Envelope Spectrum');
    end
end

% %% ========================= 子空间叠加显示 =========================
% if plot_subspaces
%     figure;
%     cmap = lines(numSub);
% 
%     subplot(2, 1, 1); hold on; grid on;
%     title('MO-Masked Reconstructed Time-domain Signals');
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% 
%     subplot(2, 1, 2); hold on; grid on;
%     title('Envelope Spectra of MO-Masked Subspaces');
%     xlabel('Frequency (Hz)');
%     ylabel('Amplitude');
%     xlim([0, 300]);
% 
%     for i = 1:numSub
%         subplot(2, 1, 1);
%         plot(t, x_sub_all{i}, 'Color', cmap(i, :), 'LineWidth', 1.2);
% 
%         subplot(2, 1, 2);
%         plot(detf_half, env_sub_all{i}(1:N_half), 'Color', cmap(i, :), 'LineWidth', 1.2);
%     end
% 
%     subplot(2, 1, 1);
%     legend(arrayfun(@(i) ['Subspace ', num2str(i)], 1:numSub, 'UniformOutput', false));
% 
%     subplot(2, 1, 2);
%     legend(arrayfun(@(i) ['Subspace ', num2str(i)], 1:numSub, 'UniformOutput', false));
% end
% 
% %% ========================= 顶层 H 频谱 =========================
% if plot_top_H_spectrum
%     fsH = fs / hop;
% 
%     for k = 1:size(H_top, 1)
%         figure;
%         subplot(2, 1, 1);
%         plot(stft_t_full, H_top(k, :));
%         xlabel('Time (s)');
%         ylabel(['H_{top}(', num2str(k), ')']);
%         title(['Top-layer Activation ', num2str(k)]);
% 
%         H_sig = H_top(k, :) - mean(H_top(k, :));
%         Lh = length(H_sig);
%         NFFT = 2^nextpow2(Lh);
%         Y = fft(H_sig, NFFT);
%         P2 = abs(Y) / Lh;
%         P1 = P2(1:floor(NFFT/2)+1);
%         if numel(P1) >= 3
%             P1(2:end-1) = 2 * P1(2:end-1);
%         end
%         fH = (0:floor(NFFT/2)) * (fsH / NFFT);
% 
%         subplot(2, 1, 2);
%         plot(fH, P1);
%         xlabel('Frequency (Hz)');
%         ylabel('Amplitude');
%         title(sprintf('FFT of H_{top}(%d)', k));
%         xlim([0, min(300, fsH/2)]);
%     end
% end
% 
% %% ========================= 保存结果 =========================
% save(result_file, ...
%     'Wks', 'Hks', 'S', 'objhistory', 'options', 'fea', ...
%     'inner_ranks', 'nbLayers', ...
%     'q_nsd', 'q_components', 'q_bar', 'layer_depth', 'middle_sensitivity', ...
%     'theta_adaptive', 'adaptive_theta_model', 'use_adaptive_theta', 'manual_theta', ...
%     'x_rec', 'x_sub_all', 'env_sub_all', 'stft_f', 'stft_t_full', 'fs', 'hop', ...
%     'p_mask', 'use_input_mo', 'mo_gamma');
% 
% fprintf('\nResult saved to %s\n', result_file);

%% ========================= 局部函数 =========================

function [rec_err, rep_err, obj, rep_state] = cost_function_dosnmf(...
    X, Z, H, S, lambda_rep, rep_lags, rep_weights, rep_temperature, ...
    harmonic_M, harmonic_weights, eps_rep)

Xhat = reconstruction_dosnmf(Z, H, S);
rec_err = norm(X - Xhat, 'fro')^2;

HL = H{numel(H)};
[rep_err, ~, rep_state] = repetition_complete_adaptive_value_and_grad(...
    HL, rep_lags, rep_weights, rep_temperature, harmonic_M, harmonic_weights, eps_rep);

obj = rec_err - lambda_rep * rep_err;
end

function [Z, H, S, objhistory] = DOSNMF(X, ranks, options)
% Deep nsNMF:
%   X ≈ Z1*S1*Z2*S2*...*ZL*SL*H_L
% 本精简版仅保留重构项和顶层 rep 奖励，不含 Z1 去相关、H 去相关、sep、在线监视。

if nargin < 3 || isempty(options)
    options = struct();
end

L = numel(ranks);

if isfield(options, 'TolFun') && ~isfield(options, 'tolfun')
    options.tolfun = options.TolFun;
end
if ~isfield(options, 'tolfun'); options.tolfun = 1e-3; end
if ~isfield(options, 'verbose'); options.verbose = 1; end
if ~isfield(options, 'fastapprox'); options.fastapprox = 1; end
if ~isfield(options, 'z0'); options.z0 = []; end
if ~isfield(options, 'h0'); options.h0 = []; end
if ~isfield(options, 'S0'); options.S0 = []; end

if ~isfield(options, 'thl')
    options.thl = linspace(0.1, 0.9, L);
end
if numel(options.thl) < L
    options.thl = [options.thl(:).' repmat(options.thl(end), 1, L - numel(options.thl))];
elseif numel(options.thl) > L
    options.thl = options.thl(1:L);
end

if ~isfield(options, 'finetune_maxiter'); options.finetune_maxiter = 800; end
if ~isfield(options, 'inner_mu'); options.inner_mu = 10; end
if ~isfield(options, 'lambda_rep'); options.lambda_rep = 0; end
if ~isfield(options, 'rep_lags'); options.rep_lags = 10:40; end
if ~isfield(options, 'rep_weights'); options.rep_weights = []; end
if ~isfield(options, 'rep_temperature'); options.rep_temperature = 1.0; end
if ~isfield(options, 'harmonic_M'); options.harmonic_M = 3; end
if ~isfield(options, 'harmonic_weights'); options.harmonic_weights = []; end
if ~isfield(options, 'eps_rep'); options.eps_rep = 1e-12; end
if ~isfield(options, 'eps'); options.eps = 1e-12; end
if ~isfield(options, 'recompute_intermediate_H'); options.recompute_intermediate_H = true; end

[Z, H, S] = DOSNMF_pretrain(X, ranks, options);
[Z, H, objhistory] = DOSNMF_finetune(X, Z, H, S, ranks, options);

if options.recompute_intermediate_H && numel(ranks) >= 2
    for l = L:-1:2
        H{l-1} = Z{l} * S{l} * H{l};
    end
end
end

function [Z, H, objhistory] = DOSNMF_finetune(X, Z, H, S, ranks, options)
L = numel(ranks);

verbose = options.verbose;
tolfun  = options.tolfun;
epsv    = options.eps;

lambda_rep       = options.lambda_rep;
rep_lags         = options.rep_lags;
rep_weights      = options.rep_weights;
rep_temperature  = options.rep_temperature;
harmonic_M       = options.harmonic_M;
harmonic_weights = options.harmonic_weights;
eps_rep          = options.eps_rep;

inner_mu = options.inner_mu;
maxiter  = options.finetune_maxiter;
obj_check_every = 5;

if verbose
    disp('----- finetune -----');
end

[~, ~, obj0, ~] = cost_function_dosnmf(...
    X, Z, H, S, lambda_rep, rep_lags, rep_weights, rep_temperature, ...
    harmonic_M, harmonic_weights, eps_rep);
objhistory = obj0;

printed_any = false;
last_progress_with_newline = true;

for iter = 1:maxiter
    for i = 1:L
        if i == 1
            B = S{i};
            for tt = (i+1):L
                B = B * Z{tt} * S{tt};
            end
            B = B * H{L};
            BBt = B * B';
            numer = X * B';

            for mu = 1:inner_mu
                Z{i} = Z{i} .* (numer ./ (Z{i} * BBt + eps(numer)));
            end
        else
            A = Z{1} * S{1};
            for tt = 2:(i-1)
                A = A * Z{tt} * S{tt};
            end
            AtA = A' * A;

            B = S{i};
            for tt = (i+1):L
                B = B * Z{tt} * S{tt};
            end
            B = B * H{L};
            BBt = B * B';
            numer = A' * X * B';

            for mu = 1:inner_mu
                Z{i} = Z{i} .* (numer ./ (AtA * Z{i} * BBt + eps(numer)));
            end
        end

        if i == L
            A = Z{1} * S{1};
            for tt = 2:L
                A = A * Z{tt} * S{tt};
            end

            numer = A' * X;
            AtA = A' * A;

            for mu = 1:inner_mu
                H{L} = H{L} .* (numer ./ (AtA * H{L} + eps(numer)));
            end

            if lambda_rep > 0
                [~, G_rep_now] = repetition_complete_adaptive_value_and_grad(...
                    H{L}, rep_lags, rep_weights, rep_temperature, ...
                    harmonic_M, harmonic_weights, eps_rep);
                H{L} = normalized_projected_step(H{L}, G_rep_now, lambda_rep, +1, epsv);
            end
        end
    end

    if rem(iter, obj_check_every) == 0 || iter == 1
        [~, ~, newobj, ~] = cost_function_dosnmf(...
            X, Z, H, S, lambda_rep, rep_lags, rep_weights, rep_temperature, ...
            harmonic_M, harmonic_weights, eps_rep);
        objhistory = [objhistory, newobj]; %#ok<AGROW>

        if numel(objhistory) >= 20
            obj_prev = objhistory(end-1);
            obj_now = objhistory(end);
            if obj_prev >= obj_now && (obj_prev - obj_now) <= tolfun * max(1, abs(obj_prev))
                break;
            end
        end
    end

    if verbose && rem(iter, 10) == 0
        fprintf('%d...', iter);
        printed_any = true;
        if rem(iter, 100) == 0
            fprintf('\n');
            last_progress_with_newline = true;
        else
            last_progress_with_newline = false;
        end
    end
end

if verbose && printed_any && ~last_progress_with_newline
    fprintf('\n');
end
end

function [Z, H, S] = DOSNMF_pretrain(X, ranks, options)
L = numel(ranks);
Z = cell(1, L);
H = cell(1, L);
S = cell(1, L);

verbose = options.verbose;
tolfun = options.tolfun;
fastapprox = options.fastapprox;
thl = options.thl;

if ~isempty(options.z0) && ~isempty(options.h0)
    Z = options.z0;
    H = options.h0;

    if isempty(options.S0)
        error('DOSNMF_pretrain:MissingS0', 'Provided z0/h0 but missing options.S0.');
    end
    S = options.S0;

    if verbose
        disp('Skipping pretrain, using provided init matrices...');
    end
    return;
end

if verbose
    disp('----- initialize -----');
end

for l = 1:L
    if l == 1
        V = X;
    else
        V = H{l-1};
    end

    if verbose
        fprintf('Initialising Layer #%d with k=%d, size(V)=%s ...\n', ...
            l, ranks(l), mat2str(size(V)));
    end

    nsopts = struct('maxiter', 500, 'TolFun', tolfun, 'thlta0', thl(l));
    [Z{l}, H{l}, S{l}, ~] = nsnmf(V, ranks(l), fastapprox, nsopts);

    Z{l} = max(Z{l}, 0);
    H{l} = max(H{l}, 0);
end
end

function [Z, H, S, objhistory] = nsnmf(X, k, fastapprox, opts)
if fastapprox
    [z0, h0] = NNDSVD(abs(X), k, 0);
else
    z0 = rand(size(X, 1), k);
    h0 = rand(k, size(X, 2));
end

optsdef = struct('z0', z0, 'h0', h0, 'bUpdateH', 1, ...
    'maxiter', 500, 'TolFun', 1e-3, 'thlta0', 0.9);
if ~exist('opts', 'var')
    opts = struct();
end

[z0, h0, bUpdateH, max_iter, tolfun, thlta] = scanparam(optsdef, opts);

S = (1 - thlta) * eye(k) + thlta / k * ones(k);
Z = z0;
H = h0;
objhistory = cost_function_nsnmf(X, Z, H, S);

for i = 1:max_iter
    B = S * H;
    numer = X * B';
    for mu = 1:10
        Z = Z .* (numer ./ (Z * (B * B') + eps(numer)));
    end

    if bUpdateH
        A = Z * S;
        numer = A' * X;
        for mu = 1:10
            H = H .* (numer ./ ((A' * A) * H + eps(numer)));
        end
    end

    if mod(i, 5) == 0
        newobj = cost_function_nsnmf(X, Z, H, S);
        objhistory = [objhistory, newobj]; %#ok<AGROW>
    end

    if numel(objhistory) >= 20
        obj_prev = objhistory(end-1);
        obj_now = objhistory(end);
        if obj_prev >= obj_now && obj_prev - obj_now <= tolfun * max(1, obj_prev)
            break;
        end
    end
end
end

function err = cost_function_nsnmf(X, Z, H, S)
err = norm(X - Z * S * H, 'fro');
end

function out = reconstruction_dosnmf(Z, H, S)
out = H{numel(H)};
for k = numel(H):-1:1
    out = Z{k} * S{k} * out;
end
end

function [W, H] = NNDSVD(A, k, flag)
if any(A(:) < 0)
    error('NNDSVD 输入矩阵不能包含负数。');
end

[m, ~] = size(A);
W = zeros(m, k);
H = zeros(k, size(A, 2));

[U, Sigma, V] = svds(A, k);

W(:, 1) = sqrt(Sigma(1, 1)) * abs(U(:, 1));
H(1, :) = sqrt(Sigma(1, 1)) * abs(V(:, 1)');

for i = 2:k
    uu = U(:, i);
    vv = V(:, i);

    uup = max(uu, 0);
    uun = max(-uu, 0);
    vvp = max(vv, 0);
    vvn = max(-vv, 0);

    n_uup = norm(uup);
    n_vvp = norm(vvp);
    n_uun = norm(uun);
    n_vvn = norm(vvn);

    termp = n_uup * n_vvp;
    termn = n_uun * n_vvn;

    if termp >= termn && n_uup > 0 && n_vvp > 0
        W(:, i) = sqrt(Sigma(i, i) * termp) * uup / n_uup;
        H(i, :) = sqrt(Sigma(i, i) * termp) * vvp' / n_vvp;
    elseif n_uun > 0 && n_vvn > 0
        W(:, i) = sqrt(Sigma(i, i) * termn) * uun / n_uun;
        H(i, :) = sqrt(Sigma(i, i) * termn) * vvn' / n_vvn;
    end
end

W(W < 1e-10) = 0.1;
H(H < 1e-10) = 0.1;

if flag == 1
    avg = mean(A(:));
    W(W == 0) = avg;
    H(H == 0) = avg;
elseif flag == 2
    avg = mean(A(:));
    indW = find(W == 0);
    indH = find(H == 0);
    W(indW) = avg * rand(numel(indW), 1) / 100;
    H(indH) = avg * rand(numel(indH), 1) / 100;
end
end

function [rep_val, G_rep, rep_state] = repetition_complete_adaptive_value_and_grad(...
    HL, rep_lags, rep_weights, rep_temperature, harmonic_M, harmonic_weights, eps_rep)
% 顶层前几阶谐波完整性奖励。
% 对每个顶层通道 h_r 和候选 lag tau，计算前 M 阶重复响应，
% 用几何加权形式强调“多阶谐波同时存在”，再用 softmax 在 lag 集合中自适应选择。

[r, T] = size(HL);
[rep_lags, rep_prior] = sanitize_rep_params(rep_lags, rep_weights, T);
[harmonic_M, harmonic_weights] = sanitize_complete_rep_params(harmonic_M, harmonic_weights);

temp = max(rep_temperature, eps_rep);

rep_state = struct('alpha', [], 'Cmat', [], 'confidence', 0, ...
    'confidence_each', [], 'entropy_each', [], 'dominant_lags', [], ...
    'dominant_lag', NaN, 'rep_prior', rep_prior, 'rep_lags', rep_lags);

if isempty(rep_lags)
    rep_val = 0;
    if nargout > 1
        G_rep = zeros(size(HL));
    else
        G_rep = [];
    end
    return;
end

mLag = numel(rep_lags);
Cmat = zeros(r, mLag);
if nargout > 1
    G_rep = zeros(size(HL));
else
    G_rep = [];
end

alpha = zeros(r, mLag);
entropy_each = zeros(1, r);
confidence_each = ones(1, r);
dominant_lags = NaN(1, r);
rep_val = 0;

for k = 1:r
    h = HL(k, :);
    gradC_each = cell(1, mLag);
    valid = false(1, mLag);

    for ii = 1:mLag
        tau = rep_lags(ii);
        if harmonic_M * tau >= T
            continue;
        end

        valid(ii) = true;
        q = zeros(1, harmonic_M);
        g_q = zeros(harmonic_M, T);

        for mm = 1:harmonic_M
            d = mm * tau;
            h1 = h(1:T-d);
            h2 = h(1+d:T);
            q(mm) = mean(h1 .* h2);

            g = zeros(1, T);
            scale = 1 / (T - d);
            g(1:T-d) = g(1:T-d) + scale * h2;
            g(1+d:T) = g(1+d:T) + scale * h1;
            g_q(mm, :) = g;
        end

        Ctau = exp(sum(harmonic_weights .* log(eps_rep + q)));
        Cmat(k, ii) = Ctau;

        if nargout > 1
            gC = zeros(1, T);
            for mm = 1:harmonic_M
                gC = gC + Ctau * (harmonic_weights(mm) / (eps_rep + q(mm))) * g_q(mm, :);
            end
            gradC_each{ii} = gC;
        end
    end

    idx = find(valid);
    if isempty(idx)
        continue;
    end

    Cvalid = Cmat(k, idx);
    logits = log(rep_prior(idx) + eps_rep) + Cvalid / temp;
    logits = logits - max(logits);
    p = exp(logits);
    p = p / (sum(p) + eps_rep);

    alpha(k, idx) = p;
    Fr = sum(p .* Cvalid);
    rep_val = rep_val + Fr;

    if numel(idx) > 1
        entropy_each(k) = -sum(p .* log(p + eps_rep));
        confidence_each(k) = max(0, 1 - entropy_each(k) / log(numel(idx)));
    else
        entropy_each(k) = 0;
        confidence_each(k) = 1;
    end

    [~, idmax] = max(p);
    dominant_lags(k) = rep_lags(idx(idmax));

    if nargout > 1
        coeff = p .* (1 + (Cvalid - Fr) / temp);
        gk_total = zeros(1, T);
        for jj = 1:numel(idx)
            gk_total = gk_total + coeff(jj) * gradC_each{idx(jj)};
        end
        G_rep(k, :) = gk_total;
    end
end

mean_alpha = mean(alpha, 1);
if any(mean_alpha > 0)
    [~, idx_global] = max(mean_alpha);
    dominant_lag = rep_lags(idx_global);
else
    dominant_lag = NaN;
end

rep_state.alpha = alpha;
rep_state.Cmat = Cmat;
rep_state.confidence_each = confidence_each;
rep_state.entropy_each = entropy_each;
rep_state.confidence = mean(confidence_each);
rep_state.dominant_lags = dominant_lags;
rep_state.dominant_lag = dominant_lag;
rep_state.rep_prior = rep_prior;
rep_state.rep_lags = rep_lags;
end

function [harmonic_M, harmonic_weights] = sanitize_complete_rep_params(harmonic_M, harmonic_weights)
if isempty(harmonic_M)
    harmonic_M = 3;
end
harmonic_M = max(1, round(harmonic_M));

if isempty(harmonic_weights)
    harmonic_weights = 1 ./ (1:harmonic_M);
else
    harmonic_weights = harmonic_weights(:).';
    if numel(harmonic_weights) ~= harmonic_M
        error('harmonic_weights 的长度必须与 harmonic_M 一致。');
    end
    harmonic_weights = max(harmonic_weights, 0);
end

harmonic_weights = harmonic_weights / (sum(harmonic_weights) + eps);
end

function [rep_lags, rep_weights] = sanitize_rep_params(rep_lags, rep_weights, T)
rep_lags = round(rep_lags(:).');
rep_lags = unique(rep_lags);
rep_lags = rep_lags(rep_lags >= 1 & rep_lags < T);

if isempty(rep_lags)
    rep_weights = [];
    return;
end

if isempty(rep_weights)
    rep_weights = ones(1, numel(rep_lags)) / numel(rep_lags);
else
    rep_weights = rep_weights(:).';
    if numel(rep_weights) ~= numel(rep_lags)
        error('rep_weights 的长度必须与 rep_lags 一致。');
    end
    rep_weights = max(rep_weights, 0);
    rep_weights = rep_weights / (sum(rep_weights) + eps);
end
end

function Xnew = normalized_projected_step(X, G, lambda, direction, epsv)
if lambda <= 0
    Xnew = max(X, 0);
    return;
end

g_norm = norm(G, 'fro');
x_norm = norm(X, 'fro');

if g_norm <= epsv || x_norm <= epsv
    Xnew = max(X, 0);
    return;
end

step = lambda * x_norm / (g_norm + epsv);
Xnew = max(X + direction * step * G, 0);
end

function [q_nsd, q_components] = compute_signal_difficulty_q(x, fs, TF_mag, qopt)
if nargin < 4 || isempty(qopt)
    qopt = struct();
end
if ~isfield(qopt, 'weights');           qopt.weights = [0.4, 0.4, 0.2]; end
if ~isfield(qopt, 'period_freq_range'); qopt.period_freq_range = [5, 300]; end
if ~isfield(qopt, 'flatness_band');     qopt.flatness_band = [5, 1000]; end
if ~isfield(qopt, 'tf_entropy_power');  qopt.tf_entropy_power = 2; end

w = qopt.weights(:).';
w = max(w, 0);
w = w / (sum(w) + eps);

E_TF = compute_tf_entropy(TF_mag, qopt.tf_entropy_power);
C_per = compute_envelope_period_clarity(x, fs, qopt.period_freq_range);
D_per = 1 - C_per;
F_flat = compute_envelope_spectral_flatness(x, fs, qopt.flatness_band);

q_nsd = w(1) * E_TF + w(2) * D_per + w(3) * F_flat;
q_nsd = min(max(q_nsd, 0), 1);

q_components = struct();
q_components.E_TF = E_TF;
q_components.C_per = C_per;
q_components.D_per = D_per;
q_components.F_flat = F_flat;
q_components.weights = w;
q_components.period_freq_range = qopt.period_freq_range;
q_components.flatness_band = qopt.flatness_band;
end

function E_TF = compute_tf_entropy(TF_mag, power_order)
if nargin < 2 || isempty(power_order)
    power_order = 2;
end

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
seg = acf_pos((lag_min+1):(lag_max+1));
seg = seg(isfinite(seg));

if isempty(seg)
    C_per = 0;
else
    C_per = max(seg);
    C_per = min(max(C_per, 0), 1);
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

function [theta, q_bar, d, psi_d] = sigmoid_theta_schedule(q_nsd, L, model)
if nargin < 3 || isempty(model)
    model = struct();
end
if ~isfield(model, 'q_min');  model.q_min = 0.381664; end
if ~isfield(model, 'q_max');  model.q_max = 0.863117; end
if ~isfield(model, 'beta0');  model.beta0 = -2.378484; end
if ~isfield(model, 'beta_d'); model.beta_d = 3.330215; end
if ~isfield(model, 'beta_q'); model.beta_q = 0.244229; end
if ~isfield(model, 'beta_m'); model.beta_m = 0.303702; end

q_bar = (q_nsd - model.q_min) / (model.q_max - model.q_min + eps);
q_bar = min(max(q_bar, 0), 1);

if L <= 1
    d = 0;
else
    d = (0:L-1) / (L-1);
end

psi_d = 4 .* d .* (1 - d);

eta = model.beta0 ...
    + model.beta_d .* d ...
    + model.beta_q .* q_bar ...
    + model.beta_m .* q_bar .* psi_d;

theta = 1 ./ (1 + exp(-eta));
theta = min(max(theta, 1e-4), 1 - 1e-4);
end

function varargout = scanparam(defoptions, options)
allfields = fieldnames(options);
opts = defoptions;

for k = 1:numel(allfields)
    name = allfields{k};
    if isfield(defoptions, name) && strcmp(class(options.(name)), class(defoptions.(name))) %#ok<STISA>
        if ~isempty(options.(name))
            opts.(name) = options.(name);
        end
    else
        fprintf('Warning! Unexpected field name or data type: %s.\n', name);
        fprintf('Warning! The corresponding value is not transferred.\n');
    end
end

if nargout > 1
    varargout = struct2cell(opts);
else
    varargout{1} = opts;
end
end
