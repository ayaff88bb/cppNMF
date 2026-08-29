# DNMF MATLAB 重构版

本目录由原始单文件 `matlab/DNMF_V9.m` 拆分得到。原文件保持不变。

## 入口

```matlab
addpath(fullfile(pwd, 'matlab_refactored'));
results = DNMF_main(signal, fs0);
```

`signal` 可以是列向量或“采样点 × 通道”矩阵，`fs0` 是采样频率。

## 主要参数

所有原文件中带“重要”标记的参数集中在 `DNMF_main.m` 顶部：

- `N_stft`：FFT 点数
- `win_len`：窗长
- `hop`：帧移
- `stft_window`：窗函数
- `inner_ranks`：各层分解秩
- `use_input_mo`：是否进行 MO 幅值压缩
- `mo_gamma`：MO 压缩指数
- `manual_theta`：各层 theta
- `lambda_rep`：周期完整性奖励权重
- `signal_channel`：信号通道

其余固定参数位于其所属子函数内部。

## 文件职责

- `DNMF_main.m`：参数入口及完整处理流程
- `prepare_signal.m`：通道选择、清洗、时域/频谱/包络谱数据
- `compute_stft_representation.m`：STFT 数据及重构所需相位
- `compute_signal_difficulty_q.m`：q_NSD 指标
- `compress_mo_input.m`：MO 幅值压缩
- `configure_dnmf.m`：固定 DNMF 参数及 theta 调度
- `DOSNMF.m`：Deep nsNMF 求解器及内部数学辅助函数
- `build_layer_reconstructions.m`：多层累计重构
- `reconstruct_signal_from_magnitude.m`：ISTFT 重构
- `reconstruct_subspaces.m`：掩膜式子空间重构
- `plot_*.m`、`print_*.m`：绘图和控制台输出

## 兼容性说明

算法公式、默认数值和原始绘图行为保持不变。入口由依赖工作区变量的脚本改为显式接收 `signal`、`fs0` 的函数，便于后续测试和 C++ 迁移。

