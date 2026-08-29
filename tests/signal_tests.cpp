#include "cppnmf/signal.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

TEST(PrepareSignal, ReplacesNonFiniteValuesAndRemovesMean) {
    const std::vector<double> input{
        1.0,
        std::numeric_limits<double>::quiet_NaN(),
        3.0,
        std::numeric_limits<double>::infinity()};

    const auto prepared = cppnmf::prepare_signal(input, 20'000.0);

    ASSERT_EQ(prepared.samples.size(), input.size());
    for (const double value : prepared.samples) {
        EXPECT_TRUE(std::isfinite(value));
    }
    const double sum = std::accumulate(
        prepared.samples.begin(), prepared.samples.end(), 0.0);
    EXPECT_NEAR(sum, 0.0, 1e-14);
}

TEST(PrepareSignal, RejectsInvalidArguments) {
    EXPECT_THROW(
        cppnmf::prepare_signal(std::vector<double>{1.0}, 20'000.0),
        std::invalid_argument);
    EXPECT_THROW(
        cppnmf::prepare_signal(std::vector<double>{1.0, 2.0}, 0.0),
        std::invalid_argument);
}

TEST(PeriodicHamming, MatchesMatlabPeriodicDefinition) {
    const auto window = cppnmf::periodic_hamming(8);
    ASSERT_EQ(window.size(), 8U);
    EXPECT_NEAR(window.front(), 0.08, 1e-14);
    EXPECT_NEAR(window[4], 1.0, 1e-14);
    EXPECT_NEAR(window.back(), 0.21473088065418816, 1e-14);
}
