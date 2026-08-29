#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <vector>

namespace cppnmf {

struct NndsvdResult {
    Eigen::MatrixXd basis;
    Eigen::MatrixXd activation;
};

struct DnmfOptions {
    std::vector<std::size_t> ranks{40, 16, 4};
    std::vector<double> theta{0.1, 0.4, 0.7};
    std::size_t pretrain_max_iterations = 500;
    std::size_t finetune_max_iterations = 800;
    std::size_t inner_updates = 10;
    double tolerance = 1e-3;
    double lambda_repetition = 1e-2;
    std::size_t repetition_lag_min = 5;
    std::size_t repetition_lag_max = 80;
    double repetition_temperature = 0.5;
    std::size_t harmonic_count = 3;
    double epsilon = 1e-12;
};

struct DnmfResult {
    std::vector<Eigen::MatrixXd> basis;
    std::vector<Eigen::MatrixXd> activation;
    std::vector<Eigen::MatrixXd> smoothing;
    std::vector<double> objective_history;
};

struct ThetaSchedule {
    std::vector<double> adaptive;
    std::vector<double> selected;
    double normalized_difficulty = 0.0;
};

NndsvdResult nndsvd(
    const Eigen::Ref<const Eigen::MatrixXd>& input,
    std::size_t rank);

ThetaSchedule configure_theta(
    double q_nsd,
    const std::vector<double>& manual_theta);

DnmfResult deep_nsnmf(
    const Eigen::Ref<const Eigen::MatrixXd>& input,
    const DnmfOptions& options = {});

Eigen::MatrixXd reconstruct_dnmf(const DnmfResult& result);

std::vector<Eigen::MatrixXd> cumulative_bases(const DnmfResult& result);

std::vector<Eigen::MatrixXd> layer_reconstructions(const DnmfResult& result);

double relative_frobenius_error(
    const Eigen::Ref<const Eigen::MatrixXd>& expected,
    const Eigen::Ref<const Eigen::MatrixXd>& actual);

}  // namespace cppnmf
