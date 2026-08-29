#include "cppnmf/reconstruction.hpp"

#include <stdexcept>

namespace cppnmf {

std::vector<double> reconstruct_from_magnitude(
    const StftResult& reference,
    const Eigen::Ref<const Eigen::MatrixXd>& magnitude) {
    if (magnitude.rows() != reference.magnitude.rows() ||
        magnitude.cols() != reference.magnitude.cols()) {
        throw std::invalid_argument(
            "reconstruction magnitude must match the reference STFT dimensions");
    }
    if (!magnitude.allFinite() || magnitude.minCoeff() < 0.0) {
        throw std::invalid_argument(
            "reconstruction magnitude must be finite and nonnegative");
    }

    StftResult modified = reference;
    for (Eigen::Index frame = 0; frame < magnitude.cols(); ++frame) {
        modified.spectrum(0, frame) = reference.zero_frequency_bin[frame];
        for (Eigen::Index bin = 0; bin < magnitude.rows(); ++bin) {
            modified.spectrum(bin + 1, frame) =
                magnitude(bin, frame) * reference.phase_positive(bin, frame);
        }
    }
    return inverse_stft(modified);
}

SubspaceReconstruction reconstruct_subspaces(
    const Eigen::Ref<const Eigen::MatrixXd>& cumulative_basis,
    const Eigen::Ref<const Eigen::MatrixXd>& top_activation,
    const Eigen::Ref<const Eigen::MatrixXd>& original_magnitude,
    const StftResult& reference,
    const double mask_epsilon) {
    if (cumulative_basis.cols() != top_activation.rows() ||
        cumulative_basis.rows() != original_magnitude.rows() ||
        top_activation.cols() != original_magnitude.cols()) {
        throw std::invalid_argument("subspace factor dimensions are inconsistent");
    }
    if (mask_epsilon <= 0.0) {
        throw std::invalid_argument("mask epsilon must be positive");
    }

    const Eigen::Index component_count = top_activation.rows();
    std::vector<Eigen::MatrixXd> parts;
    parts.reserve(static_cast<std::size_t>(component_count));
    Eigen::MatrixXd part_sum = Eigen::MatrixXd::Zero(
        original_magnitude.rows(), original_magnitude.cols());
    for (Eigen::Index component = 0; component < component_count; ++component) {
        Eigen::MatrixXd part =
            (cumulative_basis.col(component) * top_activation.row(component))
                .cwiseMax(0.0);
        part_sum += part;
        parts.push_back(std::move(part));
    }

    SubspaceReconstruction result;
    result.masks.reserve(static_cast<std::size_t>(component_count));
    result.magnitudes.reserve(static_cast<std::size_t>(component_count));
    result.signals.reserve(static_cast<std::size_t>(component_count));
    for (Eigen::Index component = 0; component < component_count; ++component) {
        Eigen::MatrixXd mask =
            (parts[static_cast<std::size_t>(component)].array() /
             (part_sum.array() + mask_epsilon))
                .matrix();
        Eigen::MatrixXd magnitude =
            (mask.array() * original_magnitude.array()).matrix();
        result.signals.push_back(reconstruct_from_magnitude(reference, magnitude));
        result.masks.push_back(std::move(mask));
        result.magnitudes.push_back(std::move(magnitude));
    }
    return result;
}

}  // namespace cppnmf
