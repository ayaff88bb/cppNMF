function results = DNMF_main(signal, fs0)
%DNMF_MAIN Deep nsNMF 冲击特征提取的主函数。
%   RESULTS = DNMF_MAIN(SIGNAL, FS0) 对 SIGNAL 的指定通道执行：
%   预处理、STFT、q_NSD 估计、MO 幅值压缩、Deep nsNMF 分解和
%   掩膜式子空间重构。
%
%   重要参数集中在下面的“主要参数”区域。其余固定参数均封装在
%   对应子函数中，原始文件 DNMF_V9.m 不会被修改。

if nargin ~= 2
    error('DNMF_main:InvalidInputCount', ...
        '用法：results = DNMF_main(signal, fs0)');
end

close all;

%% ========================= 主要参数（重要） =========================
% STFT 参数
N_stft = 1024;
win_len = 120;
hop = 20;
stft_window = hamming(win_len, "periodic");

% Deep nsNMF 各层秩
inner_ranks = [40, 16, 4];

% MO 输入压缩
use_input_mo = true;
mo_gamma = 0.6;

% 各层手动 theta
manual_theta = [0.1, 0.4, 0.7];

% 顶层前几阶谐波完整性奖励；设为 0 可关闭周期奖励
lambda_rep = 1e-2;

% 输入信号通道
signal_channel = 1;

%% ========================= 主处理流程 =========================
signal_data = prepare_signal(signal, fs0, signal_channel);
plot_input_analysis(signal_data);

stft_data = compute_stft_representation( ...
    signal_data.signal, signal_data.fs, N_stft, stft_window, hop);
plot_tf_matrix(stft_data.time, stft_data.frequency, ...
    stft_data.magnitude, 'Original STFT Magnitude');

[q_nsd, q_components] = compute_signal_difficulty_q( ...
    signal_data.signal, signal_data.fs, stft_data.magnitude);
print_signal_difficulty(q_nsd, q_components);

fea = compress_mo_input(stft_data.magnitude, use_input_mo, mo_gamma);
if use_input_mo
    input_title = ['MO-compressed STFT Magnitude, \gamma = ', num2str(mo_gamma)];
else
    input_title = 'Original STFT Magnitude Used as DNMF Input'; %#ok<UNRCH>
end
plot_tf_matrix(stft_data.time, stft_data.frequency, fea, input_title);

[options, theta_info] = configure_dnmf( ...
    q_nsd, numel(inner_ranks), manual_theta, lambda_rep);
print_theta_schedule(theta_info, options.thl);

solver_timer = tic;
[Wks, Hks, S, objhistory] = DOSNMF(fea, inner_ranks, options);
elapsed_seconds = toc(solver_timer);

plot_factor_matrices(Wks, Hks, stft_data.frequency, stft_data.time);

[X_rec_layers, W_rec_layers] = build_layer_reconstructions(Wks, Hks);
X_rec_top = X_rec_layers{end};
W_rec_top = W_rec_layers{end};
H_top = Hks{end};

x_rec = reconstruct_signal_from_magnitude( ...
    X_rec_top, stft_data.phase_positive, stft_data.zero_row, ...
    signal_data.fs, stft_window, stft_data.noverlap, N_stft, ...
    stft_data.original_length);

[x_sub_all, env_sub_all] = reconstruct_subspaces( ...
    W_rec_top, H_top, stft_data.magnitude, stft_data.phase_positive, ...
    stft_data.zero_row, signal_data, stft_window, ...
    stft_data.noverlap, N_stft);

%% ========================= 汇总输出 =========================
results = struct();
results.Wks = Wks;
results.Hks = Hks;
results.S = S;
results.objhistory = objhistory;
results.options = options;
results.features = fea;
results.inner_ranks = inner_ranks;
results.q_nsd = q_nsd;
results.q_components = q_components;
results.theta = theta_info;
results.x_rec = x_rec;
results.x_sub_all = x_sub_all;
results.env_sub_all = env_sub_all;
results.stft_frequency = stft_data.frequency;
results.stft_time = stft_data.time;
results.fs = signal_data.fs;
results.hop = hop;
results.use_input_mo = use_input_mo;
results.mo_gamma = mo_gamma;
results.elapsed_seconds = elapsed_seconds;
end
