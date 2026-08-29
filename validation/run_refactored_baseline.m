%RUN_REFACTORED_BASELINE 使用真实输入运行拆分后的 DNMF_main.m。

project_dir = fileparts(fileparts(mfilename('fullpath')));
data_file = fullfile(project_dir, '01formulate_signal_neo.mat');
refactored_dir = fullfile(project_dir, 'matlab_refactored');
output_file = fullfile(project_dir, 'validation', 'refactored_baseline.mat');

addpath(refactored_dir);
input_data = load(data_file, 'signal', 'fs0');
results = DNMF_main(input_data.signal, input_data.fs0);
save(output_file, 'results', '-v7.3');

fprintf('REFACTORED_BASELINE_SAVED=%s\n', output_file);

