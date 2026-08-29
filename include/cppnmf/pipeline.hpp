#pragma once

#include "cppnmf/config.hpp"
#include "cppnmf/dnmf.hpp"
#include "cppnmf/features.hpp"
#include "cppnmf/reconstruction.hpp"
#include "cppnmf/signal.hpp"
#include "cppnmf/stft.hpp"

#include <Eigen/Core>

#include <vector>

namespace cppnmf {

struct PipelineConfig {
    ProcessingConfig processing{};
    bool use_mo_compression = true;
    double mo_gamma = 0.6;
    DnmfOptions dnmf{};
};

struct StageTimings {
    double preprocessing_ms = 0.0;
    double stft_ms = 0.0;
    double feature_ms = 0.0;
    double solver_ms = 0.0;
    double reconstruction_ms = 0.0;
    double total_ms = 0.0;
};

struct PipelineResult {
    PreparedSignal prepared;
    StftResult stft;
    DifficultyResult difficulty;
    Eigen::MatrixXd features;
    DnmfResult dnmf;
    Eigen::MatrixXd reconstructed_magnitude;
    std::vector<double> reconstructed_signal;
    SubspaceReconstruction subspaces;
    StageTimings timings;
};

PipelineResult run_pipeline(
    const std::vector<double>& signal,
    double sample_rate_hz,
    const PipelineConfig& config = {});

}  // namespace cppnmf
