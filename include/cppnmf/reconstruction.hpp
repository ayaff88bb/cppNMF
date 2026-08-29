#pragma once

#include "cppnmf/stft.hpp"

#include <Eigen/Core>

#include <vector>

namespace cppnmf {

struct SubspaceReconstruction {
    std::vector<Eigen::MatrixXd> masks;
    std::vector<Eigen::MatrixXd> magnitudes;
    std::vector<std::vector<double>> signals;
};

std::vector<double> reconstruct_from_magnitude(
    const StftResult& reference,
    const Eigen::Ref<const Eigen::MatrixXd>& magnitude);

SubspaceReconstruction reconstruct_subspaces(
    const Eigen::Ref<const Eigen::MatrixXd>& cumulative_basis,
    const Eigen::Ref<const Eigen::MatrixXd>& top_activation,
    const Eigen::Ref<const Eigen::MatrixXd>& original_magnitude,
    const StftResult& reference,
    double mask_epsilon = 1e-12);

}  // namespace cppnmf
