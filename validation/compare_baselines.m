%COMPARE_BASELINES 比较原始脚本与拆分版函数在同一输入上的数值输出。

project_dir = fileparts(fileparts(mfilename('fullpath')));
original = load(fullfile(project_dir, 'validation', 'original_baseline.mat'));
refactored_file = load(fullfile(project_dir, 'validation', 'refactored_baseline.mat'));
refactored = refactored_file.results;

row_count = 5 + 3 * numel(original.Wks) + 2 * numel(original.x_sub_all);
rows = cell(row_count, 5);
row_index = 1;
rows(row_index, :) = compare_value('q_nsd', original.q_nsd, refactored.q_nsd);
row_index = row_index + 1;
rows(row_index, :) = compare_value('features', original.fea, refactored.features);
row_index = row_index + 1;
rows(row_index, :) = compare_value('theta_final', original.options.thl, refactored.options.thl);
row_index = row_index + 1;
rows(row_index, :) = compare_value('objhistory', original.objhistory, refactored.objhistory);
row_index = row_index + 1;
rows(row_index, :) = compare_value('x_rec', original.x_rec, refactored.x_rec);
row_index = row_index + 1;

for layer = 1:numel(original.Wks)
    rows(row_index, :) = compare_value( ...
        sprintf('W%d', layer), original.Wks{layer}, refactored.Wks{layer});
    row_index = row_index + 1;
    rows(row_index, :) = compare_value( ...
        sprintf('H%d', layer), original.Hks{layer}, refactored.Hks{layer});
    row_index = row_index + 1;
    rows(row_index, :) = compare_value( ...
        sprintf('S%d', layer), original.S{layer}, refactored.S{layer});
    row_index = row_index + 1;
end

for subspace = 1:numel(original.x_sub_all)
    rows(row_index, :) = compare_value( ...
        sprintf('x_sub_%d', subspace), ...
        original.x_sub_all{subspace}, refactored.x_sub_all{subspace});
    row_index = row_index + 1;
    rows(row_index, :) = compare_value( ...
        sprintf('env_sub_%d', subspace), ...
        original.env_sub_all{subspace}, refactored.env_sub_all{subspace});
    row_index = row_index + 1;
end

report = cell2table(rows, 'VariableNames', ...
    {'Name', 'SameSize', 'MaxAbsError', 'RelativeFroError', 'Correlation'});
report_file = fullfile(project_dir, 'validation', 'baseline_comparison.csv');
writetable(report, report_file);
disp(report);

% 同一实现路径在相同输入下应达到接近机器精度的误差。
required = ismember(report.Name, ...
    {'q_nsd', 'features', 'theta_final', 'objhistory', 'x_rec'});
if any(~report.SameSize(required)) || ...
        any(report.RelativeFroError(required) > 1e-10)
    error('compare_baselines:Mismatch', ...
        '原始程序与重构程序的关键输出未通过一致性阈值。');
end

fprintf('BASELINE_COMPARISON_OK=%s\n', report_file);

function row = compare_value(name, expected, actual)
same_size = isequal(size(expected), size(actual));
if ~same_size
    row = {name, false, Inf, Inf, NaN};
    return;
end

expected = double(expected(:));
actual = double(actual(:));
difference = actual - expected;
max_abs_error = max(abs(difference), [], 'omitnan');
relative_error = norm(difference) / max(norm(expected), eps);

if numel(expected) < 2 || std(expected) <= eps || std(actual) <= eps
    correlation = double(relative_error <= eps);
else
    correlation_matrix = corrcoef(expected, actual);
    correlation = correlation_matrix(1, 2);
end

row = {name, true, max_abs_error, relative_error, correlation};
end
