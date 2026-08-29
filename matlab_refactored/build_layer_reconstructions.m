function [X_rec_layers, W_rec_layers] = build_layer_reconstructions(Wks, Hks)
%BUILD_LAYER_RECONSTRUCTIONS 形成每一层的累计基矩阵和重构矩阵。

layer_count = numel(Wks);
if layer_count == 0 || numel(Hks) ~= layer_count
    error('build_layer_reconstructions:InvalidFactors', ...
        'Wks 与 Hks 必须是长度相同的非空 cell 数组。');
end

X_rec_layers = cell(1, layer_count);
W_rec_layers = cell(1, layer_count);

W_cumulative = Wks{1};
W_rec_layers{1} = W_cumulative;
X_rec_layers{1} = W_cumulative * Hks{1};

for layer = 2:layer_count
    W_cumulative = W_cumulative * Wks{layer};
    W_rec_layers{layer} = W_cumulative;
    X_rec_layers{layer} = W_cumulative * Hks{layer};
end
end

