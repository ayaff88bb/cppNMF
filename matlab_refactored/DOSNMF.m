function [Z, H, S, objhistory] = DOSNMF(X, ranks, options)
%DOSNMF 执行 Deep nsNMF 逐层预训练与整体微调。
%   X ~= Z1*S1*Z2*S2*...*ZL*SL*H_L

if nargin < 3 || isempty(options)
    options = struct();
end

L = numel(ranks);
if L < 1 || any(ranks < 1) || any(ranks ~= fix(ranks))
    error('DOSNMF:InvalidRanks', 'ranks 必须包含至少一个正整数。');
end

if isfield(options, 'TolFun') && ~isfield(options, 'tolfun')
    options.tolfun = options.TolFun;
end
options = set_default(options, 'tolfun', 1e-3);
options = set_default(options, 'verbose', 1);
options = set_default(options, 'fastapprox', 1);
options = set_default(options, 'z0', []);
options = set_default(options, 'h0', []);
options = set_default(options, 'S0', []);
options = set_default(options, 'thl', linspace(0.1, 0.9, L));
options = set_default(options, 'finetune_maxiter', 800);
options = set_default(options, 'inner_mu', 10);
options = set_default(options, 'lambda_rep', 0);
options = set_default(options, 'rep_lags', 10:40);
options = set_default(options, 'rep_weights', []);
options = set_default(options, 'rep_temperature', 1.0);
options = set_default(options, 'harmonic_M', 3);
options = set_default(options, 'harmonic_weights', []);
options = set_default(options, 'eps_rep', 1e-12);
options = set_default(options, 'eps', 1e-12);
options = set_default(options, 'recompute_intermediate_H', true);

if numel(options.thl) < L
    options.thl = [options.thl(:).', ...
        repmat(options.thl(end), 1, L - numel(options.thl))];
elseif numel(options.thl) > L
    options.thl = options.thl(1:L);
end

[Z, H, S] = dosnmf_pretrain(X, ranks, options);
[Z, H, objhistory] = dosnmf_finetune(X, Z, H, S, options);

if options.recompute_intermediate_H && L >= 2
    for layer = L:-1:2
        H{layer-1} = Z{layer} * S{layer} * H{layer};
    end
end
end

function options = set_default(options, name, value)
if ~isfield(options, name)
    options.(name) = value;
end
end

function [Z, H, S] = dosnmf_pretrain(X, ranks, options)
L = numel(ranks);
Z = cell(1, L);
H = cell(1, L);
S = cell(1, L);

if ~isempty(options.z0) && ~isempty(options.h0)
    Z = options.z0;
    H = options.h0;
    if isempty(options.S0)
        error('DOSNMF:MissingS0', '提供 z0/h0 时还必须提供 options.S0。');
    end
    S = options.S0;
    if options.verbose
        disp('Skipping pretrain, using provided init matrices...');
    end
    return;
end

if options.verbose
    disp('----- initialize -----');
end

for layer = 1:L
    if layer == 1
        V = X;
    else
        V = H{layer-1};
    end

    if options.verbose
        fprintf('Initialising Layer #%d with k=%d, size(V)=%s ...\n', ...
            layer, ranks(layer), mat2str(size(V)));
    end

    nsopts = struct('maxiter', 500, 'TolFun', options.tolfun, ...
        'thlta0', options.thl(layer));
    [Z{layer}, H{layer}, S{layer}] = ...
        nsnmf(V, ranks(layer), options.fastapprox, nsopts);
    Z{layer} = max(Z{layer}, 0);
    H{layer} = max(H{layer}, 0);
end
end

function [Z, H, objhistory] = dosnmf_finetune(X, Z, H, S, options)
L = numel(Z);
maxiter = options.finetune_maxiter;
inner_mu = options.inner_mu;
obj_check_every = 5;

if options.verbose
    disp('----- finetune -----');
end

[~, ~, obj0] = cost_function_dosnmf(X, Z, H, S, options);
objhistory = obj0;
printed_any = false;
last_progress_with_newline = true;

for iter = 1:maxiter
    for layer = 1:L
        if layer == 1
            B = S{layer};
            for next_layer = (layer+1):L
                B = B * Z{next_layer} * S{next_layer};
            end
            B = B * H{L};
            BBt = B * B';
            numer = X * B';

            for update = 1:inner_mu
                Z{layer} = Z{layer} .* ...
                    (numer ./ (Z{layer} * BBt + eps(numer)));
            end
        else
            A = Z{1} * S{1};
            for previous_layer = 2:(layer-1)
                A = A * Z{previous_layer} * S{previous_layer};
            end
            AtA = A' * A;

            B = S{layer};
            for next_layer = (layer+1):L
                B = B * Z{next_layer} * S{next_layer};
            end
            B = B * H{L};
            BBt = B * B';
            numer = A' * X * B';

            for update = 1:inner_mu
                Z{layer} = Z{layer} .* ...
                    (numer ./ (AtA * Z{layer} * BBt + eps(numer)));
            end
        end

        if layer == L
            A = Z{1} * S{1};
            for current_layer = 2:L
                A = A * Z{current_layer} * S{current_layer};
            end

            numer = A' * X;
            AtA = A' * A;
            for update = 1:inner_mu
                H{L} = H{L} .* (numer ./ (AtA * H{L} + eps(numer)));
            end

            if options.lambda_rep > 0
                [~, repetition_gradient] = ...
                    repetition_complete_adaptive_value_and_grad(H{L}, options);
                H{L} = normalized_projected_step( ...
                    H{L}, repetition_gradient, options.lambda_rep, ...
                    +1, options.eps);
            end
        end
    end

    if rem(iter, obj_check_every) == 0 || iter == 1
        [~, ~, newobj] = cost_function_dosnmf(X, Z, H, S, options);
        objhistory = [objhistory, newobj]; %#ok<AGROW>

        if numel(objhistory) >= 20
            previous_obj = objhistory(end-1);
            current_obj = objhistory(end);
            if previous_obj >= current_obj && ...
                    (previous_obj - current_obj) <= ...
                    options.tolfun * max(1, abs(previous_obj))
                break;
            end
        end
    end

    if options.verbose && rem(iter, 10) == 0
        fprintf('%d...', iter);
        printed_any = true;
        if rem(iter, 100) == 0
            fprintf('\n');
            last_progress_with_newline = true;
        else
            last_progress_with_newline = false;
        end
    end
end

if options.verbose && printed_any && ~last_progress_with_newline
    fprintf('\n');
end
end

function [rec_err, rep_err, obj, rep_state] = ...
        cost_function_dosnmf(X, Z, H, S, options)
Xhat = reconstruction_dosnmf(Z, H, S);
rec_err = norm(X - Xhat, 'fro')^2;

[rep_err, ~, rep_state] = ...
    repetition_complete_adaptive_value_and_grad(H{end}, options);
obj = rec_err - options.lambda_rep * rep_err;
end

function out = reconstruction_dosnmf(Z, H, S)
out = H{end};
for layer = numel(H):-1:1
    out = Z{layer} * S{layer} * out;
end
end

function [Z, H, S, objhistory] = nsnmf(X, k, fastapprox, opts)
if fastapprox
    [z0, h0] = nndsvd(abs(X), k, 0);
else
    z0 = rand(size(X, 1), k);
    h0 = rand(k, size(X, 2));
end

defaults = struct('z0', z0, 'h0', h0, 'bUpdateH', 1, ...
    'maxiter', 500, 'TolFun', 1e-3, 'thlta0', 0.9);
params = scan_parameters(defaults, opts);

S = (1 - params.thlta0) * eye(k) + params.thlta0 / k * ones(k);
Z = params.z0;
H = params.h0;
objhistory = cost_function_nsnmf(X, Z, H, S);

for iter = 1:params.maxiter
    B = S * H;
    numer = X * B';
    for update = 1:10
        Z = Z .* (numer ./ (Z * (B * B') + eps(numer)));
    end

    if params.bUpdateH
        A = Z * S;
        numer = A' * X;
        for update = 1:10
            H = H .* (numer ./ ((A' * A) * H + eps(numer)));
        end
    end

    if mod(iter, 5) == 0
        newobj = cost_function_nsnmf(X, Z, H, S);
        objhistory = [objhistory, newobj]; %#ok<AGROW>
    end

    if numel(objhistory) >= 20
        previous_obj = objhistory(end-1);
        current_obj = objhistory(end);
        if previous_obj >= current_obj && ...
                previous_obj - current_obj <= ...
                params.TolFun * max(1, previous_obj)
            break;
        end
    end
end
end

function err = cost_function_nsnmf(X, Z, H, S)
err = norm(X - Z * S * H, 'fro');
end

function params = scan_parameters(defaults, options)
params = defaults;
names = fieldnames(options);
for index = 1:numel(names)
    name = names{index};
    if isfield(defaults, name) && ...
            strcmp(class(options.(name)), class(defaults.(name)))
        if ~isempty(options.(name))
            params.(name) = options.(name);
        end
    else
        fprintf('Warning! Unexpected field name or data type: %s.\n', name);
        fprintf('Warning! The corresponding value is not transferred.\n');
    end
end
end

function [W, H] = nndsvd(A, k, flag)
if any(A(:) < 0)
    error('DOSNMF:NegativeNNDSVDInput', 'NNDSVD 输入矩阵不能包含负数。');
end

[m, n] = size(A);
if k > min(m, n)
    error('DOSNMF:RankTooLarge', ...
        'NNDSVD 的秩 %d 不能超过输入矩阵最小维度 %d。', k, min(m, n));
end

W = zeros(m, k);
H = zeros(k, n);
[U, Sigma, V] = svds(A, k);

W(:, 1) = sqrt(Sigma(1, 1)) * abs(U(:, 1));
H(1, :) = sqrt(Sigma(1, 1)) * abs(V(:, 1)');

for component = 2:k
    u = U(:, component);
    v = V(:, component);
    u_positive = max(u, 0);
    u_negative = max(-u, 0);
    v_positive = max(v, 0);
    v_negative = max(-v, 0);

    norm_u_positive = norm(u_positive);
    norm_v_positive = norm(v_positive);
    norm_u_negative = norm(u_negative);
    norm_v_negative = norm(v_negative);
    positive_term = norm_u_positive * norm_v_positive;
    negative_term = norm_u_negative * norm_v_negative;

    if positive_term >= negative_term && ...
            norm_u_positive > 0 && norm_v_positive > 0
        W(:, component) = sqrt(Sigma(component, component) * positive_term) ...
            * u_positive / norm_u_positive;
        H(component, :) = sqrt(Sigma(component, component) * positive_term) ...
            * v_positive' / norm_v_positive;
    elseif norm_u_negative > 0 && norm_v_negative > 0
        W(:, component) = sqrt(Sigma(component, component) * negative_term) ...
            * u_negative / norm_u_negative;
        H(component, :) = sqrt(Sigma(component, component) * negative_term) ...
            * v_negative' / norm_v_negative;
    end
end

W(W < 1e-10) = 0.1;
H(H < 1e-10) = 0.1;

if flag == 1
    average = mean(A(:));
    W(W == 0) = average;
    H(H == 0) = average;
elseif flag == 2
    average = mean(A(:));
    zero_w = find(W == 0);
    zero_h = find(H == 0);
    W(zero_w) = average * rand(numel(zero_w), 1) / 100;
    H(zero_h) = average * rand(numel(zero_h), 1) / 100;
end
end

function [rep_val, G_rep, state] = ...
        repetition_complete_adaptive_value_and_grad(HL, options)
[rank_count, sample_count] = size(HL);
[rep_lags, rep_prior] = sanitize_rep_params( ...
    options.rep_lags, options.rep_weights, sample_count);
[harmonic_M, harmonic_weights] = sanitize_harmonic_params( ...
    options.harmonic_M, options.harmonic_weights);

temperature = max(options.rep_temperature, options.eps_rep);
state = struct('alpha', [], 'Cmat', [], 'confidence', 0, ...
    'confidence_each', [], 'entropy_each', [], 'dominant_lags', [], ...
    'dominant_lag', NaN, 'rep_prior', rep_prior, 'rep_lags', rep_lags);

if isempty(rep_lags)
    rep_val = 0;
    G_rep = zeros(size(HL));
    return;
end

lag_count = numel(rep_lags);
Cmat = zeros(rank_count, lag_count);
G_rep = zeros(size(HL));
alpha = zeros(rank_count, lag_count);
entropy_each = zeros(1, rank_count);
confidence_each = ones(1, rank_count);
dominant_lags = NaN(1, rank_count);
rep_val = 0;

for component = 1:rank_count
    h = HL(component, :);
    gradients = cell(1, lag_count);
    valid = false(1, lag_count);

    for lag_index = 1:lag_count
        tau = rep_lags(lag_index);
        if harmonic_M * tau >= sample_count
            continue;
        end

        valid(lag_index) = true;
        q = zeros(1, harmonic_M);
        q_gradients = zeros(harmonic_M, sample_count);

        for harmonic = 1:harmonic_M
            delay = harmonic * tau;
            h1 = h(1:sample_count-delay);
            h2 = h(1+delay:sample_count);
            q(harmonic) = mean(h1 .* h2);

            gradient = zeros(1, sample_count);
            scale = 1 / (sample_count - delay);
            gradient(1:sample_count-delay) = ...
                gradient(1:sample_count-delay) + scale * h2;
            gradient(1+delay:sample_count) = ...
                gradient(1+delay:sample_count) + scale * h1;
            q_gradients(harmonic, :) = gradient;
        end

        Ctau = exp(sum(harmonic_weights .* log(options.eps_rep + q)));
        Cmat(component, lag_index) = Ctau;

        gradient_C = zeros(1, sample_count);
        for harmonic = 1:harmonic_M
            gradient_C = gradient_C + ...
                Ctau * (harmonic_weights(harmonic) / ...
                (options.eps_rep + q(harmonic))) * q_gradients(harmonic, :);
        end
        gradients{lag_index} = gradient_C;
    end

    valid_indices = find(valid);
    if isempty(valid_indices)
        continue;
    end

    valid_C = Cmat(component, valid_indices);
    logits = log(rep_prior(valid_indices) + options.eps_rep) ...
        + valid_C / temperature;
    logits = logits - max(logits);
    probabilities = exp(logits);
    probabilities = probabilities / (sum(probabilities) + options.eps_rep);

    alpha(component, valid_indices) = probabilities;
    component_value = sum(probabilities .* valid_C);
    rep_val = rep_val + component_value;

    if numel(valid_indices) > 1
        entropy_each(component) = ...
            -sum(probabilities .* log(probabilities + options.eps_rep));
        confidence_each(component) = max(0, 1 - ...
            entropy_each(component) / log(numel(valid_indices)));
    else
        entropy_each(component) = 0;
        confidence_each(component) = 1;
    end

    [~, max_index] = max(probabilities);
    dominant_lags(component) = rep_lags(valid_indices(max_index));

    coefficients = probabilities .* ...
        (1 + (valid_C - component_value) / temperature);
    total_gradient = zeros(1, sample_count);
    for valid_index = 1:numel(valid_indices)
        total_gradient = total_gradient + coefficients(valid_index) ...
            * gradients{valid_indices(valid_index)};
    end
    G_rep(component, :) = total_gradient;
end

mean_alpha = mean(alpha, 1);
if any(mean_alpha > 0)
    [~, global_index] = max(mean_alpha);
    dominant_lag = rep_lags(global_index);
else
    dominant_lag = NaN;
end

state.alpha = alpha;
state.Cmat = Cmat;
state.confidence_each = confidence_each;
state.entropy_each = entropy_each;
state.confidence = mean(confidence_each);
state.dominant_lags = dominant_lags;
state.dominant_lag = dominant_lag;
end

function [harmonic_M, harmonic_weights] = ...
        sanitize_harmonic_params(harmonic_M, harmonic_weights)
if isempty(harmonic_M)
    harmonic_M = 3;
end
harmonic_M = max(1, round(harmonic_M));

if isempty(harmonic_weights)
    harmonic_weights = 1 ./ (1:harmonic_M);
else
    harmonic_weights = harmonic_weights(:).';
    if numel(harmonic_weights) ~= harmonic_M
        error('DOSNMF:HarmonicWeightSizeMismatch', ...
            'harmonic_weights 的长度必须与 harmonic_M 一致。');
    end
    harmonic_weights = max(harmonic_weights, 0);
end
harmonic_weights = harmonic_weights / (sum(harmonic_weights) + eps);
end

function [rep_lags, rep_weights] = sanitize_rep_params(rep_lags, rep_weights, T)
rep_lags = unique(round(rep_lags(:).'));
rep_lags = rep_lags(rep_lags >= 1 & rep_lags < T);

if isempty(rep_lags)
    rep_weights = [];
    return;
end

if isempty(rep_weights)
    rep_weights = ones(1, numel(rep_lags)) / numel(rep_lags);
else
    rep_weights = rep_weights(:).';
    if numel(rep_weights) ~= numel(rep_lags)
        error('DOSNMF:RepWeightSizeMismatch', ...
            'rep_weights 的长度必须与 rep_lags 一致。');
    end
    rep_weights = max(rep_weights, 0);
    rep_weights = rep_weights / (sum(rep_weights) + eps);
end
end

function Xnew = normalized_projected_step(X, G, lambda, direction, epsv)
if lambda <= 0
    Xnew = max(X, 0);
    return;
end

gradient_norm = norm(G, 'fro');
input_norm = norm(X, 'fro');
if gradient_norm <= epsv || input_norm <= epsv
    Xnew = max(X, 0);
    return;
end

step = lambda * input_norm / (gradient_norm + epsv);
Xnew = max(X + direction * step * G, 0);
end
