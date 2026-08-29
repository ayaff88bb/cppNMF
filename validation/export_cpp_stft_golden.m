function export_cpp_stft_golden()
%EXPORT_CPP_STFT_GOLDEN Export bounded MATLAB reference data for C++ tests.

project_dir = fileparts(fileparts(mfilename('fullpath')));
refactored_dir = fullfile(project_dir, 'matlab_refactored');
output_dir = fullfile(project_dir, 'validation', 'public_golden');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

addpath(refactored_dir);

% Deterministic synthetic impact signal for public tests. This fixture
% deliberately avoids loading private research data and uses no randomness.
fs0 = 20000;
sample_count = 20000;
sample_index = (0:sample_count - 1).';
time = sample_index / fs0;

carrier = sin(2 * pi * 1800 * time);
impact_phase = mod(sample_index, 200);
impact_envelope = double(impact_phase < 12) .* ...
    exp(-impact_phase / 4);

raw_signal = 0.15 * sin(2 * pi * 260 * time) + ...
    impact_envelope .* carrier;

N_stft = 1024;
win_len = 120;
hop = 20;
window = hamming(win_len, "periodic");

prepared = prepare_signal(raw_signal, fs0, 1);
stft_data = compute_stft_representation( ...
    prepared.signal, prepared.fs, N_stft, window, hop);
positive_complex = stft_data.magnitude .* stft_data.phase_positive;

bin_indices = [1, 2, 3, 17, 65, 129, 257, 511, 512];
frame_indices = [1, 2, 3, 10, 100, 498, 993, 994, 995];
row_count = numel(bin_indices) * numel(frame_indices);

bin_zero_based = zeros(row_count, 1);
frame_zero_based = zeros(row_count, 1);
magnitude = zeros(row_count, 1);
real_part = zeros(row_count, 1);
imag_part = zeros(row_count, 1);
frequency_hz = zeros(row_count, 1);
time_seconds = zeros(row_count, 1);

row = 1;
for frame = frame_indices
    for bin = bin_indices
        value = positive_complex(bin, frame);
        bin_zero_based(row) = bin - 1;
        frame_zero_based(row) = frame - 1;
        magnitude(row) = stft_data.magnitude(bin, frame);
        real_part(row) = real(value);
        imag_part(row) = imag(value);
        frequency_hz(row) = stft_data.frequency(bin);
        time_seconds(row) = stft_data.time(frame);
        row = row + 1;
    end
end

signature = table( ...
    bin_zero_based, frame_zero_based, magnitude, real_part, imag_part, ...
    frequency_hz, time_seconds);

reconstructed = reconstruct_signal_from_magnitude( ...
    stft_data.magnitude, stft_data.phase_positive, stft_data.zero_row, ...
    prepared.fs, window, stft_data.noverlap, N_stft, prepared.length);

[q_nsd, q_components] = compute_signal_difficulty_q( ...
    prepared.signal, prepared.fs, stft_data.magnitude);
feature_summary = table( ...
    q_nsd, q_components.E_TF, q_components.C_per, ...
    q_components.D_per, q_components.F_flat, ...
    'VariableNames', {'q_nsd', 'E_TF', 'C_per', 'D_per', 'F_flat'});

mo_features = compress_mo_input(stft_data.magnitude, true, 0.6);
feature_row_zero_based = zeros(row_count, 1);
feature_column_zero_based = zeros(row_count, 1);
feature_value = zeros(row_count, 1);
row = 1;
for frame = frame_indices
    for bin = bin_indices
        feature_row_zero_based(row) = bin - 1;
        feature_column_zero_based(row) = frame - 1;
        feature_value(row) = mo_features(bin, frame);
        row = row + 1;
    end
end
mo_signature = table( ...
    feature_row_zero_based, feature_column_zero_based, feature_value, ...
    'VariableNames', {'row_zero_based', 'column_zero_based', 'value'});

metadata = table( ...
    prepared.fs, prepared.length, N_stft, win_len, hop, ...
    size(stft_data.magnitude, 1), size(stft_data.magnitude, 2), ...
    stft_data.padding_length, ...
    'VariableNames', { ...
        'sample_rate_hz', 'sample_count', 'fft_length', 'window_length', ...
        'hop_length', 'bin_count', 'frame_count', 'padding_length'});

writematrix(raw_signal, fullfile(output_dir, 'raw_signal.csv'));
writematrix(prepared.signal, fullfile(output_dir, 'prepared_signal.csv'));
writematrix(window, fullfile(output_dir, 'periodic_hamming.csv'));
writematrix(reconstructed, ...
    fullfile(output_dir, 'stft_identity_reconstruction.csv'));
writetable(signature, fullfile(output_dir, 'stft_signature.csv'));
writetable(metadata, fullfile(output_dir, 'metadata.csv'));
writetable(feature_summary, fullfile(output_dir, 'feature_summary.csv'));
writetable(mo_signature, fullfile(output_dir, 'mo_feature_signature.csv'));

fprintf('CPP_STFT_GOLDEN_SAVED=%s\n', output_dir);
end
