# cppNMF

一个面向旋转机械冲击特征提取的 C++17 信号处理项目。项目将原 MATLAB
Deep nsNMF 研究程序重构为可复用静态库、命令行工具和自动化测试，并保留
MATLAB 数值基准用于跨语言回归验证。

当前版本已经形成完整闭环：

```text
CSV 信号 → 清洗/去均值 → STFT → q_NSD → MO 压缩
         → NNDSVD → Deep nsNMF → 软掩膜 → 时域子空间
```

## 工程特性

- C++17、CMake Presets 和清晰的 `include/src/apps/tests` 分层
- Eigen 矩阵运算，KISS FFT 实现 STFT、ISTFT、Hilbert 包络与包络谱
- 3 层 Deep nsNMF：NNDSVD 初始化、nsNMF 逐层预训练、全局乘法更新微调
- 可选的多谐波周期完整性奖励，与 MATLAB 的 `lambda_rep` 路径一致
- 频域软掩膜分离顶层子空间，并还原各分量时域信号
- CLI 文件输入、运行预设、参数覆盖、CSV/JSON 结果导出和分阶段计时
- GoogleTest 单元测试，以及基于真实信号的 MATLAB golden tests

## 构建

依赖版本固定为 Eigen 3.4.0、KISS FFT 131.2.0 和 GoogleTest 1.18.0，首次
配置时由 CMake `FetchContent` 获取。

在 Visual Studio x64 Developer PowerShell 中执行：

```powershell
cmake --preset windows-msvc
cmake --build --preset windows-msvc-release
ctest --preset windows-msvc-release
```

构建结果：

- `build/windows-msvc/cppnmf_cli.exe`
- `build/windows-msvc/cppnmf_tests.exe`
- `build/windows-msvc/cppnmf_core.lib`

## 运行

快速演示：

```powershell
build/windows-msvc/cppnmf_cli.exe --demo --output outputs/demo
```

处理单通道 CSV：

```powershell
build/windows-msvc/cppnmf_cli.exe `
  --input validation/public_golden/raw_signal.csv `
  --sample-rate 20000 `
  --preset quick `
  --output outputs/public_signal_quick
```

CSV 每行是一个采样点；多列文件可用 `--channel 2` 选择第二列。执行
`cppnmf_cli --help` 可查看秩、theta、MO 指数和迭代次数等参数。

`quick` 使用较小迭代预算，适合演示、调试和 CI；`reference` 使用 MATLAB
程序的 500 次逐层预训练、800 次全局微调和每步 10 次内部更新。

MAT 文件是研究交换格式，不作为部署期依赖。可一次性转换：

```matlab
addpath('tools');
export_mat_to_csv('path/to/private_input.mat', ...
    'data/signal.csv', 'signal', 1);
```

## 输出

每次运行产生：

- `summary.json`：数据尺寸、q_NSD、重构误差和各阶段耗时
- `objective_history.csv`：全局微调目标函数轨迹
- `top_basis.csv`、`top_activation.csv`：可解释的顶层因子
- `basis/activation/smoothing_layer_N.csv`：完整求解器因子，便于复现实验
- `reconstructed_signal.csv`：顶层幅值重构信号
- `subspace_1.csv` ... `subspace_4.csv`：软掩膜分离后的冲击子空间

当前真实样本的 Release/quick 基准：20,000 点、512×995 STFT，`q_NSD =
0.600238`，总耗时约 1.43 s，其中求解器约 1.30 s。性能数字用于本机功能
基线，不应当作跨设备承诺。

## 验证

当前 16 个测试覆盖参数检查、预处理、窗函数、STFT/ISTFT、q_NSD、MO
压缩、NNDSVD、Deep nsNMF 和软掩膜重构。对根目录 20,000 点真实样本执行
MATLAB 与 C++ `reference` 全量对照后，已验证：

- 512×995 STFT 幅值矩阵相对误差 `1.46e-15`
- NNDSVD 初始化重构矩阵相对误差 `7.37e-15`，MATLAB 连续两次初始化完全复现
- 三层排列无关的匹配子空间能量加权误差分别为 `1.58e-11`、`2.13e-11`、`1.09e-11`
- 顶层 4 个掩膜子空间幅值和时域信号误差分别为 `1.26e-11`、`1.01e-11`
- 两端目标历史长度相同，初值和终值相对误差均小于 `1e-12`

因此在当前数据、参数和运行环境下，两种实现可判为浮点误差范围内数值一致。
子空间使用最大外积余弦的一对一分配进行匹配，不依赖分量编号恰好相同。

更多细节见 [验证说明](docs/VALIDATION.md) 和
[架构说明](docs/ARCHITECTURE.md)。

可选基准程序：

```powershell
cmake --preset windows-msvc -DCPPNMF_BUILD_BENCHMARKS=ON
cmake --build --preset windows-msvc-release --target cppnmf_benchmark
build/windows-msvc/cppnmf_benchmark.exe
```

项目附带 GitHub Actions，在 Windows 和 Ubuntu 上执行 Release 构建与测试。

## 目录

```text
apps/                 CLI
include/cppnmf/       公共 C++ API
src/                  核心实现
tests/                单元与跨语言回归测试
tools/                MAT → CSV 辅助脚本
validation/           MATLAB golden-data 导出与基准数据
matlab_refactored/    函数化 MATLAB 参考实现
matlab/DNMF_V9.m      未修改的原始研究程序
docs/                 架构、验证、简历与面试材料
```

## 状态与边界

这是可运行、可验证的求职作品 MVP，而不是经过现场认证的状态监测产品。
下一阶段可继续增加 WAV/二进制流输入、分块实时 STFT、模型持久化、多线程
优化和长期数据集评价。原始 MATLAB 文件 `matlab/DNMF_V9.m` 保持不变。

## 附录：项目目录与文件说明

当前根目录同时包含项目源码、MATLAB 参考实现、验证数据，以及在下载依赖、
编译和运行过程中自动生成的内容。它们并不都需要提交到 GitHub。下面按照
实际目录逐项说明。

### 根目录下的文件夹

| 文件夹 | 内容与用途 | 保留及 Git 建议 |
|---|---|---|
| `.deps/` | Eigen、GoogleTest 和 KISS FFT 的本地源码副本。CMake 编译时从这里取依赖，当前约 44 MB。 | 本机保留可避免重复下载；删除后可由 CMake 重新下载。已被 `.gitignore` 忽略，不提交。 |
| `.download/` | 依赖下载过程留下的临时缓存，目前主要是 GoogleTest，约 5 MB。它不是项目运行所必需的。 | 可以删除，需要时重新下载。已忽略，不提交。 |
| `.git/` | Git 仓库内部数据，包括分支、对象、索引和配置。没有它就没有本地版本历史。 | 必须由 Git 管理，不要手工修改；GitHub 只接收其中记录的版本历史，不会把该目录当普通文件上传。 |
| `.github/` | GitHub 专用配置。目前 `workflows/ci.yml` 定义 Windows 和 Ubuntu 的自动构建与测试。 | 应保留并提交，用于展示持续集成能力。 |
| `apps/` | 面向最终用户的应用程序入口。目前包含命令行程序 `cppnmf_cli.cpp`，负责解析参数、读取信号、调用流水线并导出结果。 | 核心源码，应保留并提交。 |
| `benchmarks/` | 性能基准程序。`cppnmf_benchmark.cpp` 对固定规模信号重复运行，输出求解器和端到端耗时。 | 应保留并提交；它与正确性测试不同，主要观察性能变化。 |
| `build/` | CMake/Ninja/MSVC 自动生成的构建目录，包含目标文件、缓存、测试发现文件、静态库和 `.exe`。当前约 89 MB。 | 可以随时删除并重新构建。已忽略，不提交。不要在这里手工修改源码。 |
| `docs/` | 项目的补充文档：架构与算法、验证方法、简历表述和面试问答。 | 应保留并提交。 |
| `include/` | C++ 公共头文件，即核心库对外暴露的数据结构和函数声明。调用 `cppnmf_core` 的程序主要包含这里的文件。 | 核心源码，应保留并提交。 |
| `MathWorks/` | MATLAB 启动时在当前目录生成的本地状态文件，目前只有一个极小的 `graphicsState.bin`。它不是算法代码。 | 可以删除。已忽略，不提交。 |
| `matlab/` | 原始 MATLAB 研究程序，仅包含未修改的 `DNMF_V9.m`，用于追溯算法来源。 | 建议保留；公开前确认其中不含不宜公开的信息。 |
| `matlab_refactored/` | 把原单文件 MATLAB 程序拆成主函数和子函数后的参考实现，是 C++ 移植时的清晰算法基准。 | 建议保留并提交，能够说明迁移过程和模块对应关系。 |
| `outputs/` | 运行 `cppnmf_cli` 后产生的结果目录，目前有合成演示和真实信号两组输出。 | 可删除、可重新生成。已忽略，通常不提交；README 所需的少量图片可另放 `docs/images/`。 |
| `src/` | `cppnmf_core` 的 C++ 实现文件，与 `include/cppnmf/` 中的公共接口一一对应。 | 核心源码，应保留并提交。 |
| `tests/` | GoogleTest 自动化测试，包括单元测试、算法性质测试和 MATLAB golden-data 回归测试。 | 应保留并提交，是项目可信度的重要组成部分。 |
| `tools/` | 开发和数据准备辅助工具。目前提供 `export_mat_to_csv.m`，用于把研究期 MAT 信号一次性转换为部署侧 CSV。 | 应保留并提交；最终 C++ 程序本身不依赖 MATLAB。 |
| `validation/` | MATLAB 拆分前后基线比较、MATLAB/C++ 数值对齐脚本，以及供 C++ 测试读取的 golden data。 | 验证代码应保留；数据文件公开前必须确认授权和隐私。 |

### 根目录下的普通文件

| 文件 | 用途 | 注意事项 |
|---|---|---|
| `.gitignore` | 告诉 Git 忽略构建目录、依赖缓存、运行输出、IDE 文件和根目录 MAT 数据。 | 它只影响未被 Git 跟踪的文件，不会自动删除任何内容。 |
| `01formulate_signal_neo.mat` | 当前研究信号及采样率的原始 MATLAB 数据，是 MATLAB 基线和真实样本验证的输入。 | 根目录 `*.mat` 已被忽略；不要在没有数据授权的情况下上传。 |
| `CMakeLists.txt` | 项目的主构建说明：声明 C++17 工程、第三方依赖、核心库、CLI、测试和 benchmark 目标。 | 新增源文件或构建目标时通常需要修改这里。 |
| `CMakePresets.json` | 保存 Windows MSVC + Ninja 的配置、构建和测试预设，因此可以使用简短的 `cmake --preset ...` 命令。 | 属于项目配置，应提交。 |
| `LICENSE` | MIT 开源许可证，规定他人复制、修改和分发源码时的权利与免责条件。 | 正式公开前可把版权主体改成自己的姓名或 GitHub 用户名。 |
| `README.md` | 项目首页，说明目标、构建、运行、输出、验证和目录结构。 | GitHub 访问者首先看到的文件。 |

### C++ 核心模块对应关系

`include/cppnmf/` 放声明，`src/` 放实现。将两者分开后，其他应用只需要依赖
公共头文件，不必了解内部实现细节。

| 头文件 / 实现文件 | 负责的功能 |
|---|---|
| `config.hpp` | STFT 与处理流水线的基础配置结构。它目前只有头文件，不需要单独的 `.cpp`。 |
| `signal.hpp` / `signal.cpp` | 信号有限值清洗、去均值和 MATLAB-compatible 周期 Hamming 窗。 |
| `stft.hpp` / `stft.cpp` | KISS FFT 前向 STFT、复数谱/幅值/相位保存、ISTFT 与重叠相加归一化。 |
| `features.hpp` / `features.cpp` | Hilbert 包络、时频熵、周期清晰度、谱平坦度、`q_NSD` 和 MO 幅值压缩。 |
| `dnmf.hpp` / `dnmf.cpp` | NNDSVD 初始化、单层 nsNMF、Deep nsNMF 预训练与全局微调、theta 调度和周期奖励。 |
| `reconstruction.hpp` / `reconstruction.cpp` | 用参考相位从幅值矩阵恢复时域信号，并用软掩膜重构顶层子空间。 |
| `pipeline.hpp` / `pipeline.cpp` | 串联预处理、STFT、特征、DNMF 和重构，记录各阶段运行时间。 |
| `io.hpp` / `io.cpp` | CSV 信号读取，以及向量、矩阵和结果 CSV 的写出。 |

### 应用和性能入口

- `apps/cppnmf_cli.cpp`：生成 `cppnmf_cli.exe`。它支持 `--input`、
  `--sample-rate`、`--preset`、`--ranks`、`--theta`、`--mo-gamma`、
  `--lambda-rep` 和输出目录等参数。
- `benchmarks/cppnmf_benchmark.cpp`：生成可选的 `cppnmf_benchmark.exe`。
  它使用固定的 20,000 点合成冲击信号，重复运行后报告中位耗时，不负责判断
  算法正确性。

### 自动化测试文件

| 测试文件 | 测试内容 |
|---|---|
| `signal_tests.cpp` | 非有限值处理、去均值、非法参数和周期 Hamming 窗。 |
| `stft_tests.cpp` | STFT 尺寸、尾部补零、往返重构、非法配置和 MATLAB 复数谱 golden tests。 |
| `features_tests.cpp` | MO 压缩的范数/非负性，以及 `q_NSD`、MO 特征与 MATLAB 的数值对齐。 |
| `dnmf_tests.cpp` | NNDSVD 维度和非负性、Deep nsNMF 层级结构、目标下降与 theta 调度。 |
| `reconstruction_tests.cpp` | 幅值恒等重构和多个软掩膜对原始幅值的分区性质。 |

### MATLAB 参考实现

`matlab_refactored/` 中的文件按职责拆分如下：

| 文件 | 作用 |
|---|---|
| `DNMF_main.m` | MATLAB 拆分版总入口，集中保存原程序中标为“重要”的参数。 |
| `prepare_signal.m` | 选择通道、清洗信号，并准备时域、频谱和包络谱数据。 |
| `compute_stft_representation.m` | 计算 STFT，并保存 C++ 重构所需的幅值、相位和 DC 行。 |
| `compute_signal_difficulty_q.m` | 计算 `q_NSD` 及其各组成量。 |
| `compress_mo_input.m` | 对 STFT 幅值做 MO 幂次压缩和 Frobenius 范数校准。 |
| `configure_dnmf.m` | 组织固定 DNMF 参数、手动 theta 和自适应 theta 候选值。 |
| `DOSNMF.m` | Deep nsNMF 主求解器及其 NNDSVD、乘法更新、周期奖励等内部函数。 |
| `build_layer_reconstructions.m` | 生成各层累计基矩阵和重构矩阵。 |
| `reconstruct_signal_from_magnitude.m` | 把指定幅值、参考相位和 DC 行合成为时域信号。 |
| `reconstruct_subspaces.m` | 构造分量软掩膜并恢复各顶层子空间。 |
| `plot_input_analysis.m` | 绘制输入信号、频谱和包络谱。 |
| `plot_tf_matrix.m` | 绘制时频矩阵。 |
| `plot_factor_matrices.m` | 绘制各层基矩阵和激活矩阵。 |
| `print_signal_difficulty.m` | 在 MATLAB 控制台打印 `q_NSD` 组成。 |
| `print_theta_schedule.m` | 打印 theta 调度信息。 |
| `README.md` | 说明 MATLAB 拆分版入口、重要参数和文件职责。 |

### 验证目录

`validation/` 包含三条验证链路：原 MATLAB 单文件与拆分版比较、有限
golden-data 自动测试，以及 MATLAB/C++ reference 全量端到端比较。

| 文件或目录 | 作用 |
|---|---|
| `run_original_baseline.m` | 运行原始 `DNMF_V9.m` 并保存关键输出。 |
| `run_refactored_baseline.m` | 运行函数化的 `DNMF_main.m` 并保存相同类型输出。 |
| `compare_baselines.m` | 比较两份 MATLAB 输出的尺寸、误差和相关系数。 |
| `original_baseline.mat` | 原始程序产生的基线结果，可由脚本重新生成。 |
| `refactored_baseline.mat` | 拆分版程序产生的基线结果，可由脚本重新生成。 |
| `baseline_comparison.csv` | 两个 MATLAB 版本的定量比较汇总。 |
| `export_cpp_stft_golden.m` | 生成确定性的合成冲击信号，并从 MATLAB 导出 C++ 测试所需的有限基准数据。 |
| `export_full_matlab_reference.m` | 无绘图运行 MATLAB reference 配置，导出 STFT、NNDSVD、各层因子/重构及顶层子空间。 |
| `compare_full_parity.py` | 对各层物理子空间做排列无关的一对一匹配，生成误差 JSON、Markdown 报告和映射 CSV。 |
| `public_golden/` | C++ 自动测试读取的公开合成信号、窗函数、STFT 签名、ISTFT、`q_NSD` 和 MO 特征基准。 |
| `cpp_golden/` | 由真实研究信号生成的本机私有基准；已由 `.gitignore` 排除，不随公开仓库发布。 |
| `README.md` | 记录 MATLAB 基线验证的执行顺序。 |

其中 `original_baseline.mat`、`refactored_baseline.mat` 和 `cpp_golden/` 都由
当前研究信号派生，因此已排除在公开版本之外。`public_golden/` 仅由脚本中的
确定性合成冲击信号生成，可以提交并供本地测试及 CI 使用。

### `build/windows-msvc/` 中常见内容

- `cppnmf_cli.exe`：最终命令行程序。
- `cppnmf_tests.exe`：测试可执行文件。
- `cppnmf_benchmark.exe`：可选性能基准程序。
- `cppnmf_core.lib`：核心静态库。
- `CMakeCache.txt`、`CMakeFiles/`、`build.ninja`：CMake/Ninja 的构建状态。
- `_deps/`、`bin/`、`lib/`：构建后的第三方依赖目标和库。
- `Testing/`、`CTestTestfile.cmake`、`cppnmf_tests[1]_*.cmake`：CTest 和
  GoogleTest 自动发现产生的文件。
- `.ninja_deps`、`.ninja_log`：Ninja 的增量构建记录。

这些内容全部由构建系统生成。若构建缓存损坏，可以删除整个 `build/` 后重新
执行配置、构建和测试命令，而不应逐个修补其中的文件。

### `outputs/` 中的运行结果

- `demo_quick/`：CLI 内置合成冲击信号的快速模式结果。
- `real_signal_quick/`：当前 20,000 点真实信号的快速模式结果。
- `summary.json`：输入尺寸、`q_NSD`、重构误差和分阶段耗时。
- `objective_history.csv`：求解目标的迭代轨迹。
- `basis_layer_N.csv`、`activation_layer_N.csv`、`smoothing_layer_N.csv`：
  第 N 层完整因子。
- `top_basis.csv`、`top_activation.csv`：用于解释和子空间重构的顶层因子。
- `reconstructed_signal.csv`：整体重构的时域信号。
- `subspace_N.csv`：第 N 个软掩膜子空间的时域信号。

这些是运行产物而不是源码，可以在需要时重新生成。

### 最简保留与清理原则

- **必须保留的项目主体**：`.github/`、`apps/`、`benchmarks/`、`docs/`、
  `include/`、`src/`、`tests/`、`tools/`、`CMakeLists.txt`、
  `CMakePresets.json`、`README.md`、`LICENSE` 和 `.gitignore`。
- **算法追溯与验证资料**：`matlab/`、`matlab_refactored/`、`validation/`；
  建议保留，但公开数据前先检查授权。
- **可安全重新生成的本地内容**：`build/`、`outputs/`、`.download/`、
  `MathWorks/`。`.deps/` 也能重新下载，但保留它可避免再次获取依赖。
- **不要手工处理的目录**：`.git/`。版本操作应使用 Git 命令完成。
