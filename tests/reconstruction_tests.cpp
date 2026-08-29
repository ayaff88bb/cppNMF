#include "cppnmf/reconstruction.hpp"
#include "cppnmf/signal.hpp"
#include "cppnmf/stft.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

TEST(Reconstruction, IdentityMagnitudeRestoresPreparedSignal) {
    constexpr double sample_rate = 20000.0;
    constexpr double pi = 3.141592653589793238462643383279502884;
    std::vector<double> signal(2000);
    for (std::size_t index = 0; index < signal.size(); ++index) {
        const double time = static_cast<double>(index) / sample_rate;
        signal[index] = std::sin(2.0 * pi * 370.0 * time);
    }
    const auto prepared = cppnmf::prepare_signal(signal, sample_rate);
    const auto stft = cppnmf::compute_stft(prepared.samples, sample_rate);
    const auto reconstructed =
        cppnmf::reconstruct_from_magnitude(stft, stft.magnitude);
    EXPECT_LT(cppnmf::relative_l2_error(prepared.samples, reconstructed), 1e-12);
}

TEST(Reconstruction, SoftMasksPartitionOriginalMagnitude) {
    constexpr double sample_rate = 20000.0;
    std::vector<double> signal(1000, 0.0);
    for (std::size_t index = 0; index < signal.size(); ++index) {
        signal[index] = index % 40 < 3 ? 1.0 : 0.0;
    }
    const auto prepared = cppnmf::prepare_signal(signal, sample_rate);
    const auto stft = cppnmf::compute_stft(prepared.samples, sample_rate);

    Eigen::MatrixXd basis = Eigen::MatrixXd::Ones(stft.magnitude.rows(), 2);
    basis.col(1).array() *= 2.0;
    Eigen::MatrixXd activation = Eigen::MatrixXd::Ones(2, stft.magnitude.cols());
    activation.row(1).array() *= 0.5;
    const auto result = cppnmf::reconstruct_subspaces(
        basis, activation, stft.magnitude, stft);
    ASSERT_EQ(result.signals.size(), 2U);
    EXPECT_EQ(result.signals[0].size(), signal.size());
    EXPECT_EQ(result.signals[1].size(), signal.size());
    const Eigen::MatrixXd magnitude_sum =
        result.magnitudes[0] + result.magnitudes[1];
    EXPECT_LT(cppnmf::relative_l2_error(
                  std::vector<double>{stft.magnitude.norm()},
                  std::vector<double>{magnitude_sum.norm()}),
              1e-8);
    EXPECT_TRUE(magnitude_sum.isApprox(stft.magnitude, 1e-8));
}

}  // namespace
