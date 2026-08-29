function export_mat_to_csv(input_file, output_csv, signal_name, channel)
%EXPORT_MAT_TO_CSV Convert a MATLAB research signal into standalone CSV input.
% Example:
%   export_mat_to_csv('../01formulate_signal_neo.mat', ...
%       '../data/signal.csv', 'signal', 1)

if nargin < 3 || isempty(signal_name)
    signal_name = 'signal';
end
if nargin < 4 || isempty(channel)
    channel = 1;
end

loaded = load(input_file, signal_name, 'fs0');
if ~isfield(loaded, signal_name)
    error('export_mat_to_csv:MissingSignal', ...
        'Variable "%s" does not exist in %s.', signal_name, input_file);
end

signal = loaded.(signal_name);
if isvector(signal)
    signal = signal(:);
else
    if channel < 1 || channel > size(signal, 2)
        error('export_mat_to_csv:InvalidChannel', ...
            'Channel %d is outside the signal matrix.', channel);
    end
    signal = signal(:, channel);
end

output_dir = fileparts(output_csv);
if ~isempty(output_dir) && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
writematrix(signal, output_csv);

if isfield(loaded, 'fs0')
    fprintf('CSV=%s\nSAMPLE_RATE_HZ=%.17g\n', output_csv, loaded.fs0);
else
    fprintf('CSV=%s\nSAMPLE_RATE_HZ=<not found in MAT file>\n', output_csv);
end
end
