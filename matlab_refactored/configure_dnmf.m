function [options, theta_info] = configure_dnmf(q_nsd, layer_count, manual_theta, lambda_rep)
%CONFIGURE_DNMF 构造固定的 DNMF 参数和 theta 调度信息。
%   除主函数传入的主要参数外，其余算法参数集中隐藏在本函数中。

if numel(manual_theta) ~= layer_count
    error('configure_dnmf:ThetaSizeMismatch', ...
        'manual_theta 的元素数必须与 inner_ranks 的层数一致。');
end

% 固定的自适应 theta 模型
model = struct();
model.name = 'middle_sensitive_sigmoid_q_d_psi';
model.q_min = 0.381664;
model.q_max = 0.863117;
model.beta0 = -2.378484;
model.beta_d = 3.330215;
model.beta_q = 0.244229;
model.beta_m = 0.303702;

[theta_adaptive, q_bar, layer_depth, middle_sensitivity] = ...
    sigmoid_theta_schedule(q_nsd, layer_count, model);

% 与原程序一致：当前固定使用手动 theta；自适应结果仍计算并返回供观察。
use_adaptive_theta = false;
final_theta = manual_theta;

% 固定的 DNMF 求解参数
options = struct();
options.fastapprox = 1;
options.verbose = 1;
options.tolfun = 1e-3;
options.finetune_maxiter = 800;
options.inner_mu = 10;
options.eps = 1e-12;
options.lambda_rep = lambda_rep;
options.rep_lags = 5:80;
options.rep_weights = [];
options.rep_temperature = 0.5;
options.harmonic_M = 3;
options.harmonic_weights = [];
options.eps_rep = 1e-12;
options.thl = final_theta;

theta_info = struct();
theta_info.adaptive = theta_adaptive;
theta_info.q_bar = q_bar;
theta_info.layer_depth = layer_depth;
theta_info.middle_sensitivity = middle_sensitivity;
theta_info.model = model;
theta_info.use_adaptive = use_adaptive_theta;
theta_info.manual = manual_theta;
theta_info.final = final_theta;
end

function [theta, q_bar, d, psi_d] = sigmoid_theta_schedule(q_nsd, L, model)
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
