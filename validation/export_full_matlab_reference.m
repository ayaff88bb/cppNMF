function export_full_matlab_reference(input_file, output_dir)
%EXPORT_FULL_MATLAB_REFERENCE Export a plot-free MATLAB reference run.
%   The exported matrices are consumed by compare_full_parity.py.  This
%   function deliberately uses the same parameters as DNMF_main.m and the
%   C++ "reference" preset.

if nargin < 1 || isempty(input_file)
    root_dir = fileparts(fileparts(mfilename('fullpath')));
    input_file = fullfile(root_dir, '01formulate_signal_neo.mat');
else
    root_dir = fileparts(fileparts(mfilename('fullpath')));
end
if nargin < 2 || isempty(output_dir)
    output_dir = fullfile(root_dir, 'outputs', 'full_parity', 'matlab');
end

addpath(fullfile(root_dir, 'matlab_refactored'));
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

loaded = load(input_file, 'signal', 'fs0');
if ~isfield(loaded, 'signal') || ~isfield(loaded, 'fs0')
    error('export_full_matlab_reference:MissingInput', ...
        '输入 MAT 文件必须包含 signal 和 fs0。');
end

% Keep these values synchronized with matlab_refactored/DNMF_main.m and
% include/cppnmf/*.hpp.
N_stft = 1024;
win_len = 120;
hop = 20;
stft_window = hamming(win_len, "periodic");
inner_ranks = [40, 16, 4];
manual_theta = [0.1, 0.4, 0.7];
use_input_mo = true;
mo_gamma = 0.6;
lambda_rep = 1e-2;
signal_channel = 1;

signal_data = prepare_signal(loaded.signal, loaded.fs0, signal_channel);
stft_data = compute_stft_representation( ...
    signal_data.signal, signal_data.fs, N_stft, stft_window, hop);
[q_nsd, q_components] = compute_signal_difficulty_q( ...
    signal_data.signal, signal_data.fs, stft_data.magnitude);
features = compress_mo_input(stft_data.magnitude, use_input_mo, mo_gamma);
[options, theta_info] = configure_dnmf( ...
    q_nsd, numel(inner_ranks), manual_theta, lambda_rep);
options.verbose = 0;

solver_timer = tic;
[Wks, Hks, S, objhistory] = DOSNMF(features, inner_ranks, options);
elapsed_seconds = toc(solver_timer);

[X_rec_layers, W_rec_layers] = build_layer_reconstructions(Wks, Hks);
solver_reconstruction = reconstruct_solver(Wks, Hks, S);
reconstructed_signal = reconstruct_signal_from_magnitude( ...
    X_rec_layers{end}, stft_data.phase_positive, stft_data.zero_row, ...
    signal_data.fs, stft_window, stft_data.noverlap, N_stft, ...
    stft_data.original_length);
[subspace_magnitudes, subspace_signals] = reconstruct_subspaces_without_plots( ...
    W_rec_layers{end}, Hks{end}, stft_data.magnitude, ...
    stft_data.phase_positive, stft_data.zero_row, signal_data.fs, ...
    stft_window, stft_data.noverlap, N_stft, signal_data.length);

% Export the deterministic NNDSVD call separately.  A repeated call checks
% whether MATLAB's svds backend itself changes the result in this run.
[init_basis_1, init_activation_1] = validation_nndsvd(features, inner_ranks(1));
[init_basis_2, init_activation_2] = validation_nndsvd(features, inner_ranks(1));

write_matrix(output_dir, 'prepared_signal.csv', signal_data.signal);
write_matrix(output_dir, 'stft_magnitude.csv', stft_data.magnitude);
write_matrix(output_dir, 'stft_frequency_hz.csv', stft_data.frequency);
write_matrix(output_dir, 'stft_time_seconds.csv', stft_data.time);
write_matrix(output_dir, 'features.csv', features);
write_matrix(output_dir, 'solver_reconstruction.csv', solver_reconstruction);
write_matrix(output_dir, 'reconstructed_signal.csv', reconstructed_signal);
write_matrix(output_dir, 'top_basis.csv', W_rec_layers{end});
write_matrix(output_dir, 'top_activation.csv', Hks{end});
write_matrix(output_dir, 'objective_history.csv', objhistory(:));
write_matrix(output_dir, 'nndsvd_basis_layer_1.csv', init_basis_1);
write_matrix(output_dir, 'nndsvd_activation_layer_1.csv', init_activation_1);
write_matrix(output_dir, 'nndsvd_basis_layer_1_repeat.csv', init_basis_2);
write_matrix(output_dir, 'nndsvd_activation_layer_1_repeat.csv', init_activation_2);

for layer = 1:numel(Wks)
    suffix = sprintf('%d.csv', layer);
    write_matrix(output_dir, ['basis_layer_', suffix], Wks{layer});
    write_matrix(output_dir, ['activation_layer_', suffix], Hks{layer});
    write_matrix(output_dir, ['smoothing_layer_', suffix], S{layer});
    write_matrix(output_dir, ['cumulative_basis_layer_', suffix], ...
        W_rec_layers{layer});
    write_matrix(output_dir, ['layer_reconstruction_', suffix], ...
        X_rec_layers{layer});
end
for component = 1:numel(subspace_signals)
    suffix = sprintf('%d.csv', component);
    write_matrix(output_dir, ['subspace_', suffix], subspace_signals{component});
    write_matrix(output_dir, ['subspace_magnitude_', suffix], ...
        subspace_magnitudes{component});
end

metadata = struct();
metadata.input_file = char(input_file);
metadata.matlab_version = version;
metadata.sample_rate_hz = signal_data.fs;
metadata.sample_count = signal_data.length;
metadata.stft_fft_length = N_stft;
metadata.stft_window_length = win_len;
metadata.stft_hop_length = hop;
metadata.stft_bins = size(stft_data.magnitude, 1);
metadata.stft_frames = size(stft_data.magnitude, 2);
metadata.ranks = inner_ranks;
metadata.theta = theta_info.final;
metadata.mo_gamma = mo_gamma;
metadata.lambda_rep = lambda_rep;
metadata.pretrain_max_iterations = 500;
metadata.finetune_max_iterations = options.finetune_maxiter;
metadata.inner_updates = options.inner_mu;
metadata.tolerance = options.tolfun;
metadata.epsilon = options.eps;
metadata.q_nsd = q_nsd;
metadata.q_components = q_components;
metadata.solver_elapsed_seconds = elapsed_seconds;
metadata.nndsvd_repeat_basis_relative_error = relative_frobenius( ...
    init_basis_1, init_basis_2);
metadata.nndsvd_repeat_activation_relative_error = relative_frobenius( ...
    init_activation_1, init_activation_2);
write_json(fullfile(output_dir, 'metadata.json'), metadata);

fprintf('MATLAB full reference exported to %s\n', output_dir);
fprintf('Solver elapsed seconds: %.6f\n', elapsed_seconds);
end

function reconstruction = reconstruct_solver(Wks, Hks, S)
reconstruction = Hks{end};
for layer = numel(Wks):-1:1
    reconstruction = Wks{layer} * S{layer} * reconstruction;
end
end

function [magnitudes, signals] = reconstruct_subspaces_without_plots( ...
        cumulative_basis, top_activation, original_magnitude, ...
        phase_positive, zero_row, fs, window, noverlap, N_stft, original_length)
component_count = size(top_activation, 1);
parts = cell(1, component_count);
part_sum = zeros(size(original_magnitude));
for component = 1:component_count
    parts{component} = max( ...
        cumulative_basis(:, component) * top_activation(component, :), 0);
    part_sum = part_sum + parts{component};
end

magnitudes = cell(1, component_count);
signals = cell(1, component_count);
for component = 1:component_count
    mask = parts{component} ./ (part_sum + 1e-12);
    magnitudes{component} = mask .* original_magnitude;
    signals{component} = reconstruct_signal_from_magnitude( ...
        magnitudes{component}, phase_positive, zero_row, fs, window, ...
        noverlap, N_stft, original_length);
end
end

function [W, H] = validation_nndsvd(A, rank_count)
if any(A(:) < 0)
    error('export_full_matlab_reference:NegativeNNDSVDInput', ...
        'NNDSVD 输入必须非负。');
end
[row_count, column_count] = size(A);
W = zeros(row_count, rank_count);
H = zeros(rank_count, column_count);
[U, Sigma, V] = svds(A, rank_count);

W(:, 1) = sqrt(Sigma(1, 1)) * abs(U(:, 1));
H(1, :) = sqrt(Sigma(1, 1)) * abs(V(:, 1)');
for component = 2:rank_count
    u = U(:, component);
    v = V(:, component);
    u_positive = max(u, 0);
    u_negative = max(-u, 0);
    v_positive = max(v, 0);
    v_negative = max(-v, 0);
    positive_term = norm(u_positive) * norm(v_positive);
    negative_term = norm(u_negative) * norm(v_negative);
    if positive_term >= negative_term && ...
            norm(u_positive) > 0 && norm(v_positive) > 0
        W(:, component) = sqrt(Sigma(component, component) * positive_term) ...
            * u_positive / norm(u_positive);
        H(component, :) = sqrt(Sigma(component, component) * positive_term) ...
            * v_positive' / norm(v_positive);
    elseif norm(u_negative) > 0 && norm(v_negative) > 0
        W(:, component) = sqrt(Sigma(component, component) * negative_term) ...
            * u_negative / norm(u_negative);
        H(component, :) = sqrt(Sigma(component, component) * negative_term) ...
            * v_negative' / norm(v_negative);
    end
end
W(W < 1e-10) = 0.1;
H(H < 1e-10) = 0.1;
end

function write_matrix(output_dir, filename, value)
writematrix(value, fullfile(output_dir, filename));
end

function write_json(path, value)
file_id = fopen(path, 'w', 'n', 'UTF-8');
if file_id < 0
    error('export_full_matlab_reference:CannotWriteJson', ...
        '无法创建 %s。', path);
end
cleanup = onCleanup(@() fclose(file_id)); %#ok<NASGU>
fprintf(file_id, '%s\n', jsonencode(value, PrettyPrint=true));
end

function value = relative_frobenius(expected, actual)
denominator = norm(expected, 'fro');
if denominator == 0
    value = norm(actual, 'fro');
else
    value = norm(expected - actual, 'fro') / denominator;
end
end
