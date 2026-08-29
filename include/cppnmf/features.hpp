#pragma once

#include <Eigen/Core>

#include <utility>
#include <vector>

namespace cppnmf {

struct DifficultyComponents {
    double time_frequency_entropy = 0.0;
    double periodic_clarity = 0.0;
    double periodic_difficulty = 1.0;
    double envelope_spectral_flatness = 1.0;
};

struct DifficultyResult {
    double q_nsd = 0.0;
    DifficultyComponents components{};
};

DifficultyResult compute_signal_difficulty(
    const std::vector<double>& signal,
    double sample_rate_hz,
    const Eigen::Ref<const Eigen::MatrixXd>& stft_magnitude);

Eigen::MatrixXd compress_mo_input(
    const Eigen::Ref<const Eigen::MatrixXd>& magnitude,
    bool enabled = true,
    double gamma = 0.6);

}  // namespace cppnmf
