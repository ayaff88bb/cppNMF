function print_theta_schedule(info, final_theta)
%PRINT_THETA_SCHEDULE 输出 theta 调度信息。

fprintf('\n===== Theta scheduling =====\n');
fprintf('use_adaptive_theta = %d\n', info.use_adaptive);
fprintf('model = %s\n', info.model.name);
fprintf('q_bar = %.6f  [q_min=%.6f, q_max=%.6f]\n', ...
    info.q_bar, info.model.q_min, info.model.q_max);
fprintf('layer depth d = [%s]\n', num2str(info.layer_depth, ' %.3f'));
fprintf('middle sensitivity psi(d) = [%s]\n', ...
    num2str(info.middle_sensitivity, ' %.3f'));
fprintf('theta_adaptive = [%s]\n', num2str(info.adaptive, ' %.6f'));
fprintf('Final options.thl = [%s]\n', num2str(final_theta, ' %.6f'));
end

