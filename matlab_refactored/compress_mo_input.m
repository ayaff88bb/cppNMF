function features = compress_mo_input(magnitude, use_input_mo, mo_gamma)
%COMPRESS_MO_INPUT 对 STFT 幅值执行 MO 幂次压缩和 Frobenius 范数校准。

if ~use_input_mo
    features = magnitude;
    return;
end

scale = max(magnitude(:)) + eps;
normalized = magnitude / scale;
compressed = (normalized + eps).^mo_gamma;
features = compressed * norm(magnitude, 'fro') / (norm(compressed, 'fro') + eps);
end

