%RUN_ORIGINAL_BASELINE 使用真实输入运行未经修改的 DNMF_V9.m。
% 本脚本供自动验证使用，必须从 cppNMF/validation 目录运行。

project_dir = fileparts(fileparts(mfilename('fullpath')));
data_file = fullfile(project_dir, '01formulate_signal_neo.mat');
original_file = fullfile(project_dir, 'matlab', 'DNMF_V9.m');

load(data_file, 'signal', 'fs0');
run(original_file);

% DNMF_V9.m 会执行 clearvars，因此运行后重新获得输出路径。
project_dir = fileparts(fileparts(mfilename('fullpath')));
output_file = fullfile(project_dir, 'validation', 'original_baseline.mat');

save(output_file, ...
    'signal', 'fs0', 'Wks', 'Hks', 'S', 'objhistory', 'options', 'fea', ...
    'inner_ranks', 'q_nsd', 'q_components', 'theta_adaptive', ...
    'q_bar', 'layer_depth', 'middle_sensitivity', 'x_rec', ...
    'x_sub_all', 'env_sub_all', 'stft_f', 'stft_t_full', ...
    'use_input_mo', 'mo_gamma', '-v7.3');

fprintf('ORIGINAL_BASELINE_SAVED=%s\n', output_file);
