# MATLAB 与 C++ 数值验证

本目录用于验证原始单文件程序与拆分版函数的数值一致性。

执行顺序：

1. `run_original_baseline.m`
2. `run_refactored_baseline.m`
3. `compare_baselines.m`

生成文件：

- `original_baseline.mat`：原始 `DNMF_V9.m` 的关键输出
- `refactored_baseline.mat`：`DNMF_main.m` 的结构化输出
- `baseline_comparison.csv`：尺寸、最大绝对误差、相对 Frobenius 误差和相关系数

基线文件由输入数据派生，后续可进一步裁剪为 C++ 单元测试所需的黄金数据。

## 可公开的 MATLAB/C++ golden tests

`export_cpp_stft_golden.m` 不读取研究数据，而是在脚本内生成固定的 20 kHz、
20,000 点合成冲击信号。执行：

```powershell
matlab -batch "addpath('validation'); export_cpp_stft_golden"
```

输出位于 `validation/public_golden/`，可提交并供 GoogleTest 和 CI 使用。
`validation/cpp_golden/` 保留为本机真实样本基准，已由 `.gitignore` 排除。

## MATLAB/C++ reference 私有全量验证

这条链路使用根目录 `01formulate_signal_neo.mat`，在两端统一采用：1024 点
FFT、120 点周期 Hamming 窗、20 点 hop、秩 `40,16,4`、theta
`0.1,0.4,0.7`、MO gamma `0.6`、逐层预训练 500 次、全局微调 800 次、
每步 10 次乘法更新和 `lambda_rep=0.01`。

执行：

```powershell
matlab -batch "addpath('tools','validation'); export_mat_to_csv('01formulate_signal_neo.mat',fullfile(pwd,'outputs','full_parity','input_signal.csv'),'signal',1); export_full_matlab_reference('01formulate_signal_neo.mat',fullfile(pwd,'outputs','full_parity','matlab'));"

build/windows-msvc/cppnmf_cli.exe `
  --input outputs/full_parity/input_signal.csv `
  --sample-rate 20000 --preset reference --validation-export `
  --output outputs/full_parity/cpp

python validation/compare_full_parity.py `
  --matlab-dir outputs/full_parity/matlab `
  --cpp-dir outputs/full_parity/cpp `
  --output-dir outputs/full_parity/comparison
```

比较器不会假设子空间顺序相同。它根据每个分量的“累计基向量 × 激活向量”
外积余弦相似度执行一对一最优匹配，再比较各层重构、各分量外积、顶层软掩膜
幅值和时域信号。生成：

- `verification_summary.json`：完整机器可读指标和最终判定
- `VERIFICATION_REPORT.md`：可直接阅读的结果摘要
- `layer_1_matches.csv` 至 `layer_3_matches.csv`：逐层分量映射
- `top_subspace_matches.csv`：顶层子空间映射与误差

当前实测结论见 `docs/MATLAB_CPP_PARITY.md`。`outputs/` 已加入 `.gitignore`，
避免把从研究数据派生的大型 CSV 提交到公开仓库。
