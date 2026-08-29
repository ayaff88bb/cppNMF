"""Compare full MATLAB and C++ cppNMF reference exports.

The comparison is permutation-invariant at every DNMF layer.  Components are
matched by the cosine similarity of their physical outer-product contribution
(cumulative basis column x activation row), then compared numerically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


STRICT_RELATIVE_TOLERANCE = 1e-10
DNMF_NUMERIC_TOLERANCE = 1e-6
DNMF_ENGINEERING_TOLERANCE = 1e-3
MATCH_COSINE_TOLERANCE = 0.999


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matlab-dir", required=True, type=Path)
    parser.add_argument("--cpp-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_csv(path: Path, *, skip_header: bool = False) -> np.ndarray:
    value = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1 if skip_header else 0,
        ndmin=2,
    )
    if not np.all(np.isfinite(value)):
        raise ValueError(f"non-finite value in {path}")
    return value


def metric(expected: np.ndarray, actual: np.ndarray) -> dict[str, float | list[int]]:
    if expected.shape != actual.shape:
        raise ValueError(f"shape mismatch: {expected.shape} != {actual.shape}")
    difference = actual - expected
    expected_norm = float(np.linalg.norm(expected))
    actual_norm = float(np.linalg.norm(actual))
    difference_norm = float(np.linalg.norm(difference))
    relative = difference_norm / expected_norm if expected_norm else difference_norm
    denominator = expected_norm * actual_norm
    cosine = (
        float(np.vdot(expected.ravel(), actual.ravel()).real) / denominator
        if denominator
        else (1.0 if difference_norm == 0.0 else 0.0)
    )
    max_expected = float(np.max(np.abs(expected))) if expected.size else 0.0
    max_absolute = float(np.max(np.abs(difference))) if difference.size else 0.0
    return {
        "shape": list(expected.shape),
        "relative_frobenius_error": relative,
        "rmse": float(math.sqrt(np.mean(np.square(difference)))) if difference.size else 0.0,
        "max_absolute_error": max_absolute,
        "max_relative_to_peak": max_absolute / max_expected if max_expected else max_absolute,
        "cosine_similarity": cosine,
    }


def hungarian_minimum(cost: np.ndarray) -> np.ndarray:
    """Return the assigned column for each row using a square Hungarian solve."""
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("component matching requires a square cost matrix")
    size = cost.shape[0]
    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    p = np.zeros(size + 1, dtype=np.int64)
    way = np.zeros(size + 1, dtype=np.int64)
    for row in range(1, size + 1):
        p[0] = row
        column0 = 0
        minimum = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1, column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = np.empty(size, dtype=np.int64)
    for column in range(1, size + 1):
        assignment[p[column] - 1] = column - 1
    return assignment


def normalized_cross_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=0)
    right_norm = np.linalg.norm(right, axis=0)
    denominator = np.outer(left_norm, right_norm)
    result = left.T @ right
    return np.divide(result, denominator, out=np.zeros_like(result), where=denominator > 0)


def normalized_cross_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = np.outer(left_norm, right_norm)
    result = left @ right.T
    return np.divide(result, denominator, out=np.zeros_like(result), where=denominator > 0)


def component_matching(
    matlab_basis: np.ndarray,
    matlab_activation: np.ndarray,
    cpp_basis: np.ndarray,
    cpp_activation: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    if matlab_basis.shape != cpp_basis.shape or matlab_activation.shape != cpp_activation.shape:
        raise ValueError("factor shapes differ across languages")
    basis_cosine = normalized_cross_columns(matlab_basis, cpp_basis)
    activation_cosine = normalized_cross_rows(matlab_activation, cpp_activation)
    contribution_cosine = np.clip(basis_cosine * activation_cosine, -1.0, 1.0)
    assignment = hungarian_minimum(1.0 - contribution_cosine)

    rows: list[dict[str, Any]] = []
    squared_difference_sum = 0.0
    squared_expected_sum = 0.0
    for matlab_component, cpp_component in enumerate(assignment):
        matlab_basis_vector = matlab_basis[:, matlab_component]
        cpp_basis_vector = cpp_basis[:, cpp_component]
        matlab_activation_vector = matlab_activation[matlab_component, :]
        cpp_activation_vector = cpp_activation[cpp_component, :]
        expected_norm = float(
            np.linalg.norm(matlab_basis_vector) * np.linalg.norm(matlab_activation_vector)
        )
        actual_norm = float(
            np.linalg.norm(cpp_basis_vector) * np.linalg.norm(cpp_activation_vector)
        )
        expected_outer = np.outer(matlab_basis_vector, matlab_activation_vector)
        actual_outer = np.outer(cpp_basis_vector, cpp_activation_vector)
        squared_difference = float(
            np.sum(np.square(actual_outer - expected_outer), dtype=np.float64)
        )
        relative_outer_error = (
            math.sqrt(squared_difference) / expected_norm
            if expected_norm
            else math.sqrt(squared_difference)
        )
        squared_difference_sum += squared_difference
        squared_expected_sum += expected_norm**2

        basis_denominator = float(np.vdot(cpp_basis_vector, cpp_basis_vector).real)
        basis_scale = (
            float(np.vdot(cpp_basis_vector, matlab_basis_vector).real) / basis_denominator
            if basis_denominator
            else 0.0
        )
        basis_shape_error = float(
            np.linalg.norm(matlab_basis_vector - basis_scale * cpp_basis_vector)
            / max(np.linalg.norm(matlab_basis_vector), np.finfo(float).tiny)
        )
        rows.append(
            {
                "matlab_component": matlab_component + 1,
                "cpp_component": int(cpp_component) + 1,
                "contribution_cosine": float(
                    contribution_cosine[matlab_component, cpp_component]
                ),
                "basis_cosine": float(basis_cosine[matlab_component, cpp_component]),
                "activation_cosine": float(
                    activation_cosine[matlab_component, cpp_component]
                ),
                "relative_outer_product_error": relative_outer_error,
                "basis_scale_cpp_to_matlab": basis_scale,
                "basis_scale_invariant_error": basis_shape_error,
                "matlab_component_norm": expected_norm,
                "cpp_component_norm": actual_norm,
            }
        )

    cosines = np.array([row["contribution_cosine"] for row in rows])
    relative_errors = np.array([row["relative_outer_product_error"] for row in rows])
    weighted_error = math.sqrt(squared_difference_sum / squared_expected_sum)
    summary = {
        "component_count": len(rows),
        "minimum_contribution_cosine": float(np.min(cosines)),
        "median_contribution_cosine": float(np.median(cosines)),
        "components_cosine_at_least_0_999": int(np.sum(cosines >= MATCH_COSINE_TOLERANCE)),
        "maximum_relative_outer_product_error": float(np.max(relative_errors)),
        "median_relative_outer_product_error": float(np.median(relative_errors)),
        "energy_weighted_matched_component_error": weighted_error,
    }
    return summary, rows, assignment


def write_match_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def scientific(value: float) -> str:
    return f"{value:.6e}"


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matlab_metadata = json.loads((args.matlab_dir / "metadata.json").read_text(encoding="utf-8"))
    cpp_metadata = json.loads((args.cpp_dir / "summary.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "tolerances": {
            "strict_frontend_relative": STRICT_RELATIVE_TOLERANCE,
            "dnmf_numeric_relative": DNMF_NUMERIC_TOLERANCE,
            "dnmf_engineering_relative": DNMF_ENGINEERING_TOLERANCE,
            "component_match_cosine": MATCH_COSINE_TOLERANCE,
        },
        "matlab_metadata": matlab_metadata,
        "cpp_metadata": cpp_metadata,
    }

    result["prepared_signal"] = metric(
        load_csv(args.matlab_dir / "prepared_signal.csv").ravel(),
        load_csv(args.cpp_dir / "prepared_signal.csv", skip_header=True).ravel(),
    )
    result["stft_frequency_hz"] = metric(
        load_csv(args.matlab_dir / "stft_frequency_hz.csv").ravel(),
        load_csv(args.cpp_dir / "stft_frequency_hz.csv", skip_header=True).ravel(),
    )
    result["stft_time_seconds"] = metric(
        load_csv(args.matlab_dir / "stft_time_seconds.csv").ravel(),
        load_csv(args.cpp_dir / "stft_time_seconds.csv", skip_header=True).ravel(),
    )
    result["stft_magnitude"] = metric(
        load_csv(args.matlab_dir / "stft_magnitude.csv"),
        load_csv(args.cpp_dir / "stft_magnitude.csv"),
    )
    result["features"] = metric(
        load_csv(args.matlab_dir / "features.csv"),
        load_csv(args.cpp_dir / "features.csv"),
    )

    matlab_init_basis = load_csv(args.matlab_dir / "nndsvd_basis_layer_1.csv")
    matlab_init_activation = load_csv(args.matlab_dir / "nndsvd_activation_layer_1.csv")
    cpp_init_basis = load_csv(args.cpp_dir / "nndsvd_basis_layer_1.csv")
    cpp_init_activation = load_csv(args.cpp_dir / "nndsvd_activation_layer_1.csv")
    init_summary, init_rows, _ = component_matching(
        matlab_init_basis, matlab_init_activation, cpp_init_basis, cpp_init_activation
    )
    init_summary["aggregate_reconstruction"] = metric(
        matlab_init_basis @ matlab_init_activation,
        cpp_init_basis @ cpp_init_activation,
    )
    init_summary["matlab_repeat_basis"] = metric(
        matlab_init_basis,
        load_csv(args.matlab_dir / "nndsvd_basis_layer_1_repeat.csv"),
    )
    init_summary["matlab_repeat_activation"] = metric(
        matlab_init_activation,
        load_csv(args.matlab_dir / "nndsvd_activation_layer_1_repeat.csv"),
    )
    result["nndsvd_layer_1"] = init_summary
    write_match_csv(args.output_dir / "nndsvd_layer_1_matches.csv", init_rows)

    ranks = [int(value) for value in matlab_metadata["ranks"]]
    result["layers"] = []
    assignments: list[np.ndarray] = []
    for layer, rank in enumerate(ranks, start=1):
        matlab_basis = load_csv(args.matlab_dir / f"cumulative_basis_layer_{layer}.csv")
        cpp_basis = load_csv(args.cpp_dir / f"cumulative_basis_layer_{layer}.csv")
        matlab_activation = load_csv(args.matlab_dir / f"activation_layer_{layer}.csv")
        cpp_activation = load_csv(args.cpp_dir / f"activation_layer_{layer}.csv")
        if matlab_activation.shape[0] != rank:
            raise ValueError(f"unexpected MATLAB rank at layer {layer}")
        summary, rows, assignment = component_matching(
            matlab_basis, matlab_activation, cpp_basis, cpp_activation
        )
        summary["layer"] = layer
        summary["rank"] = rank
        summary["aggregate_reconstruction"] = metric(
            load_csv(args.matlab_dir / f"layer_reconstruction_{layer}.csv"),
            load_csv(args.cpp_dir / f"layer_reconstruction_{layer}.csv"),
        )
        summary["smoothing"] = metric(
            load_csv(args.matlab_dir / f"smoothing_layer_{layer}.csv"),
            load_csv(args.cpp_dir / f"smoothing_layer_{layer}.csv"),
        )
        result["layers"].append(summary)
        assignments.append(assignment)
        write_match_csv(args.output_dir / f"layer_{layer}_matches.csv", rows)

    result["solver_reconstruction"] = metric(
        load_csv(args.matlab_dir / "solver_reconstruction.csv"),
        load_csv(args.cpp_dir / "solver_reconstruction.csv"),
    )
    result["interpretable_reconstructed_signal"] = metric(
        load_csv(args.matlab_dir / "reconstructed_signal.csv").ravel(),
        load_csv(args.cpp_dir / "reconstructed_signal.csv", skip_header=True).ravel(),
    )

    top_assignment = assignments[-1]
    subspace_rows: list[dict[str, Any]] = []
    magnitude_difference_energy = 0.0
    magnitude_expected_energy = 0.0
    signal_difference_energy = 0.0
    signal_expected_energy = 0.0
    for matlab_component, cpp_component in enumerate(top_assignment):
        matlab_magnitude = load_csv(
            args.matlab_dir / f"subspace_magnitude_{matlab_component + 1}.csv"
        )
        cpp_magnitude = load_csv(
            args.cpp_dir / f"subspace_magnitude_{int(cpp_component) + 1}.csv"
        )
        matlab_signal = load_csv(
            args.matlab_dir / f"subspace_{matlab_component + 1}.csv"
        ).ravel()
        cpp_signal = load_csv(
            args.cpp_dir / f"subspace_{int(cpp_component) + 1}.csv",
            skip_header=True,
        ).ravel()
        magnitude_metric = metric(matlab_magnitude, cpp_magnitude)
        signal_metric = metric(matlab_signal, cpp_signal)
        magnitude_difference_energy += float(np.sum(np.square(cpp_magnitude - matlab_magnitude)))
        magnitude_expected_energy += float(np.sum(np.square(matlab_magnitude)))
        signal_difference_energy += float(np.sum(np.square(cpp_signal - matlab_signal)))
        signal_expected_energy += float(np.sum(np.square(matlab_signal)))
        subspace_rows.append(
            {
                "matlab_component": matlab_component + 1,
                "cpp_component": int(cpp_component) + 1,
                "magnitude_relative_error": magnitude_metric["relative_frobenius_error"],
                "magnitude_cosine": magnitude_metric["cosine_similarity"],
                "signal_relative_error": signal_metric["relative_frobenius_error"],
                "signal_cosine": signal_metric["cosine_similarity"],
            }
        )
    write_match_csv(args.output_dir / "top_subspace_matches.csv", subspace_rows)
    result["top_subspaces"] = {
        "matches": subspace_rows,
        "energy_weighted_magnitude_error": math.sqrt(
            magnitude_difference_energy / magnitude_expected_energy
        ),
        "energy_weighted_signal_error": math.sqrt(
            signal_difference_energy / signal_expected_energy
        ),
    }

    matlab_objective = load_csv(args.matlab_dir / "objective_history.csv").ravel()
    cpp_objective = load_csv(
        args.cpp_dir / "objective_history.csv", skip_header=True
    ).ravel()
    result["objective_history"] = {
        "matlab_length": int(matlab_objective.size),
        "cpp_length": int(cpp_objective.size),
        "initial_relative_error": abs(float(cpp_objective[0] - matlab_objective[0]))
        / max(abs(float(matlab_objective[0])), np.finfo(float).tiny),
        "final_relative_error": abs(float(cpp_objective[-1] - matlab_objective[-1]))
        / max(abs(float(matlab_objective[-1])), np.finfo(float).tiny),
        "matlab_initial": float(matlab_objective[0]),
        "cpp_initial": float(cpp_objective[0]),
        "matlab_final": float(matlab_objective[-1]),
        "cpp_final": float(cpp_objective[-1]),
    }

    frontend_keys = [
        "prepared_signal",
        "stft_frequency_hz",
        "stft_time_seconds",
        "stft_magnitude",
        "features",
    ]
    frontend_equal = all(
        result[key]["relative_frobenius_error"] <= STRICT_RELATIVE_TOLERANCE
        for key in frontend_keys
    )
    layer_numeric_equal = all(
        layer["energy_weighted_matched_component_error"] <= DNMF_NUMERIC_TOLERANCE
        for layer in result["layers"]
    )
    layer_engineering_equal = all(
        layer["energy_weighted_matched_component_error"] <= DNMF_ENGINEERING_TOLERANCE
        for layer in result["layers"]
    )
    subspace_numeric_equal = (
        result["top_subspaces"]["energy_weighted_magnitude_error"]
        <= DNMF_NUMERIC_TOLERANCE
    )
    subspace_engineering_equal = (
        result["top_subspaces"]["energy_weighted_magnitude_error"]
        <= DNMF_ENGINEERING_TOLERANCE
    )
    result["verdict"] = {
        "frontend_strictly_equal": frontend_equal,
        "dnmf_layers_numerically_equal": layer_numeric_equal,
        "dnmf_layers_engineering_equivalent": layer_engineering_equal,
        "top_subspaces_numerically_equal": subspace_numeric_equal,
        "top_subspaces_engineering_equivalent": subspace_engineering_equal,
        "overall_numerically_equal": frontend_equal
        and layer_numeric_equal
        and subspace_numeric_equal,
        "overall_engineering_equivalent": frontend_equal
        and layer_engineering_equal
        and subspace_engineering_equal,
    }

    result = clean_json(result)
    (args.output_dir / "verification_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# MATLAB/C++ 全量数值验证报告",
        "",
        "## 结论",
        "",
        f"- 前端（预处理、坐标轴、STFT 幅值、MO 特征）严格一致：`{frontend_equal}`。",
        f"- DNMF 各层匹配子空间达到 1e-6 数值等价：`{layer_numeric_equal}`。",
        f"- DNMF 各层匹配子空间达到 1e-3 工程等价：`{layer_engineering_equal}`。",
        f"- 顶层掩膜子空间达到 1e-6 数值等价：`{subspace_numeric_equal}`。",
        f"- 顶层掩膜子空间达到 1e-3 工程等价：`{subspace_engineering_equal}`。",
        "",
        "这里的子空间编号通过最大化“累计基向量 × 激活向量”外积的余弦相似度进行一对一匹配，未假设 MATLAB 和 C++ 的列顺序相同。",
        "",
        "## 前端矩阵",
        "",
        "| 对象 | 相对 Frobenius/L2 误差 | 最大绝对误差 | 余弦相似度 |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "prepared_signal": "预处理信号",
        "stft_frequency_hz": "STFT 频率轴",
        "stft_time_seconds": "STFT 时间轴",
        "stft_magnitude": "STFT 幅值矩阵",
        "features": "MO 压缩特征",
    }
    for key in frontend_keys:
        value = result[key]
        lines.append(
            f"| {labels[key]} | {scientific(value['relative_frobenius_error'])} | "
            f"{scientific(value['max_absolute_error'])} | {value['cosine_similarity']:.12f} |"
        )

    lines.extend(
        [
            "",
            "## NNDSVD 初始化",
            "",
            f"- MATLAB 连续两次 NNDSVD 的 W 相对误差：{scientific(init_summary['matlab_repeat_basis']['relative_frobenius_error'])}。",
            f"- MATLAB 连续两次 NNDSVD 的 H 相对误差：{scientific(init_summary['matlab_repeat_activation']['relative_frobenius_error'])}。",
            f"- MATLAB/C++ 初始化重构矩阵相对误差：{scientific(init_summary['aggregate_reconstruction']['relative_frobenius_error'])}。",
            f"- 匹配后初始化分量能量加权误差：{scientific(init_summary['energy_weighted_matched_component_error'])}。",
            "",
            "## 各层子空间",
            "",
            "| 层 | 秩 | 层重构相对误差 | 匹配分量能量加权误差 | 最低/中位余弦 | ≥0.999 数量 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for layer in result["layers"]:
        lines.append(
            f"| {layer['layer']} | {layer['rank']} | "
            f"{scientific(layer['aggregate_reconstruction']['relative_frobenius_error'])} | "
            f"{scientific(layer['energy_weighted_matched_component_error'])} | "
            f"{layer['minimum_contribution_cosine']:.6f} / {layer['median_contribution_cosine']:.6f} | "
            f"{layer['components_cosine_at_least_0_999']} |"
        )

    lines.extend(
        [
            "",
            "## 顶层掩膜子空间",
            "",
            f"- 四个匹配子空间幅值矩阵的能量加权相对误差：{scientific(result['top_subspaces']['energy_weighted_magnitude_error'])}。",
            f"- 四个匹配子空间时域信号的能量加权相对误差：{scientific(result['top_subspaces']['energy_weighted_signal_error'])}。",
            f"- 含 S 的求解器重构矩阵相对误差：{scientific(result['solver_reconstruction']['relative_frobenius_error'])}。",
            f"- 不含 S 的可解释重构时域信号相对误差：{scientific(result['interpretable_reconstructed_signal']['relative_frobenius_error'])}。",
            "",
            "逐分量映射见 `layer_1_matches.csv`、`layer_2_matches.csv`、`layer_3_matches.csv` 和 `top_subspace_matches.csv`。",
            "",
            "## 判定口径",
            "",
            "- STFT/前端严格阈值：相对误差 ≤ 1e-10。",
            "- DNMF 数值等价阈值：匹配子空间能量加权相对误差 ≤ 1e-6。",
            "- DNMF 工程等价阈值：匹配子空间能量加权相对误差 ≤ 1e-3。",
            "- NNDSVD 没有显式随机填充不代表不同 SVD 库必然返回逐元素相同的奇异子空间；本报告单独测量了初始化与最终结果。",
        ]
    )
    (args.output_dir / "VERIFICATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
