#pragma once

#include "cppnmf/config.hpp"

#include <Eigen/Core>

#include <complex>
#include <cstddef>
#include <vector>

namespace cppnmf {

struct StftResult {
    Eigen::MatrixXcd spectrum;
    Eigen::MatrixXd magnitude;
    Eigen::MatrixXcd phase_positive;
    Eigen::RowVectorXcd zero_frequency_bin;
    Eigen::VectorXd frequency_hz;
    Eigen::VectorXd time_seconds;
    std::vector<double> window;
    std::size_t original_length = 0;
    std::size_t padding_length = 0;
    std::size_t hop_length = 0;
    std::size_t fft_length = 0;
};

StftResult compute_stft(
    const std::vector<double>& signal,
    double sample_rate_hz,
    const StftConfig& config = {});

std::vector<double> inverse_stft(const StftResult& stft);

double relative_l2_error(
    const std::vector<double>& expected,
    const std::vector<double>& actual);

}  // namespace cppnmf
