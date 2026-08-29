function print_signal_difficulty(q_nsd, components)
%PRINT_SIGNAL_DIFFICULTY 输出 q_NSD 及其组成。

fprintf('\n===== q_NSD signal difficulty =====\n');
fprintf('E_TF   = %.6f\n', components.E_TF);
fprintf('C_per  = %.6f, D_per = %.6f\n', components.C_per, components.D_per);
fprintf('F_flat = %.6f\n', components.F_flat);
fprintf('q_NSD  = %.6f\n', q_nsd);
end

