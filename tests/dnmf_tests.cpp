#include "cppnmf/dnmf.hpp"

#include <gtest/gtest.h>

#include <algorithm>

namespace {

Eigen::MatrixXd make_low_rank_input() {
    Eigen::MatrixXd basis(12, 3);
    for (Eigen::Index row = 0; row < basis.rows(); ++row) {
        for (Eigen::Index column = 0; column < basis.cols(); ++column) {
            basis(row, column) = 0.2 +
                static_cast<double>(((row + 2) * (column + 3)) % 11) / 10.0;
        }
    }
    Eigen::MatrixXd activation(3, 30);
    for (Eigen::Index row = 0; row < activation.rows(); ++row) {
        for (Eigen::Index column = 0; column < activation.cols(); ++column) {
            activation(row, column) = 0.1 +
                static_cast<double>(((row + 5) * (column + 1)) % 13) / 12.0;
        }
    }
    return basis * activation;
}

TEST(Nndsvd, ProducesNonnegativeFactorsWithExpectedDimensions) {
    const Eigen::MatrixXd input = make_low_rank_input();
    const auto result = cppnmf::nndsvd(input, 3);
    EXPECT_EQ(result.basis.rows(), 12);
    EXPECT_EQ(result.basis.cols(), 3);
    EXPECT_EQ(result.activation.rows(), 3);
    EXPECT_EQ(result.activation.cols(), 30);
    EXPECT_GE(result.basis.minCoeff(), 0.0);
    EXPECT_GE(result.activation.minCoeff(), 0.0);
    EXPECT_LT(cppnmf::relative_frobenius_error(
                  input, result.basis * result.activation),
              0.8);
}

TEST(DeepNsnmf, BuildsConsistentNonnegativeLayerHierarchy) {
    const Eigen::MatrixXd input = make_low_rank_input();
    cppnmf::DnmfOptions options;
    options.ranks = {6, 3};
    options.theta = {0.1, 0.4};
    options.pretrain_max_iterations = 20;
    options.finetune_max_iterations = 20;
    options.inner_updates = 2;
    options.lambda_repetition = 0.0;

    const auto result = cppnmf::deep_nsnmf(input, options);
    ASSERT_EQ(result.basis.size(), 2U);
    ASSERT_EQ(result.activation.size(), 2U);
    ASSERT_EQ(result.smoothing.size(), 2U);
    EXPECT_EQ(result.basis[0].rows(), 12);
    EXPECT_EQ(result.basis[0].cols(), 6);
    EXPECT_EQ(result.basis[1].rows(), 6);
    EXPECT_EQ(result.basis[1].cols(), 3);
    EXPECT_EQ(result.activation[1].rows(), 3);
    EXPECT_EQ(result.activation[1].cols(), 30);
    EXPECT_GE(result.basis[0].minCoeff(), 0.0);
    EXPECT_GE(result.basis[1].minCoeff(), 0.0);
    EXPECT_GE(result.activation[1].minCoeff(), 0.0);

    const auto reconstruction = cppnmf::reconstruct_dnmf(result);
    EXPECT_EQ(reconstruction.rows(), input.rows());
    EXPECT_EQ(reconstruction.cols(), input.cols());
    EXPECT_LT(cppnmf::relative_frobenius_error(input, reconstruction), 0.35);

    const auto layer_outputs = cppnmf::layer_reconstructions(result);
    ASSERT_EQ(layer_outputs.size(), 2U);
    EXPECT_EQ(layer_outputs[0].rows(), input.rows());
    EXPECT_EQ(layer_outputs[0].cols(), input.cols());
    EXPECT_EQ(layer_outputs[1].rows(), input.rows());
    EXPECT_EQ(layer_outputs[1].cols(), input.cols());
    EXPECT_GE(layer_outputs[0].minCoeff(), 0.0);
    EXPECT_GE(layer_outputs[1].minCoeff(), 0.0);
    ASSERT_GE(result.objective_history.size(), 2U);
    EXPECT_LT(result.objective_history.back(), result.objective_history.front());
}

TEST(ThetaSchedule, KeepsManualScheduleAndComputesAdaptiveCandidate) {
    const auto schedule = cppnmf::configure_theta(0.6, {0.1, 0.4, 0.7});
    EXPECT_EQ(schedule.selected, (std::vector<double>{0.1, 0.4, 0.7}));
    ASSERT_EQ(schedule.adaptive.size(), 3U);
    EXPECT_TRUE(std::all_of(
        schedule.adaptive.begin(), schedule.adaptive.end(),
        [](const double value) { return value > 0.0 && value < 1.0; }));
}

}  // namespace
