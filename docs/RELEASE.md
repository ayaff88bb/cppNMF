# 发布流程

本文记录 cppNMF 的可复现发布流程。版本号同时出现在 `CMakeLists.txt` 的
`project(... VERSION ...)` 和 Git 标签中；发布前两者必须一致。

## 1. 发布前检查

从最新 `main` 创建发布准备分支，并完成配置、构建和测试：

```powershell
cmake --preset windows-msvc
cmake --build --preset windows-msvc-release
ctest --preset windows-msvc-release
```

## 2. 生成本机交付包

```powershell
cpack --preset windows-msvc-release
```

输出目录为 `build/windows-msvc/packages/`。ZIP 中应包含：

```text
cppNMF-<version>-windows-x64/
├── bin/
│   ├── cppnmf_cli.exe
│   └── MSVC runtime DLLs
├── LICENSE
└── README.md
```

解压到一个独立目录，执行以下冒烟测试：

```powershell
bin/cppnmf_cli.exe --demo --output outputs/demo
```

## 3. Pull Request

将发布准备分支推送到 GitHub，创建 Pull Request。只有 Windows 和 Ubuntu 的
配置、构建、测试与打包任务全部通过后，才能合并到 `main`。

## 4. 创建版本标签

合并后同步本地 `main`，创建并推送带说明的标签：

```powershell
git switch main
git pull --ff-only
git tag -a v0.1.0 -m "cppNMF v0.1.0"
git push origin v0.1.0
```

`.github/workflows/release.yml` 会重新构建和测试 Windows/Linux 版本，生成带
SHA-256 校验文件的交付包，并自动创建 GitHub Release。不要在 PR 合并前创建
标签，否则标签可能指向未经保护流程验收的提交。
