#include "cppnmf/features.hpp"
#include "cppnmf/signal.hpp"
#include "cppnmf/stft.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<double> read_column(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<double> values;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            values.push_back(std::stod(line));
        }
    }
    return values;
}

std::vector<double> read_summary(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open " + path);
    }
    std::string header;
    std::string values;
    std::getline(input, header);
    std::getline(input, values);
    std::stringstream stream(values);
    std::vector<double> result;
    std::string field;
    while (std::getline(stream, field, ',')) {
        result.push_back(std::stod(field));
    }
    return result;
}

struct FeatureSignatureRow {
    Eigen::Index row = 0;
    Eigen::Index column = 0;
    double value = 0.0;
};

std::vector<FeatureSignatureRow> read_feature_signature(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open " + path);
    }
    std::string line;
    std::getline(input, line);
    std::vector<FeatureSignatureRow> rows;
    while (std::getline(input, line)) {
        std::stringstream stream(line);
        std::string field;
        FeatureSignatureRow row;
        std::getline(stream, field, ',');
        row.row = static_cast<Eigen::Index>(std::stoll(field));
        std::getline(stream, field, ',');
        row.column = static_cast<Eigen::Index>(std::stoll(field));
        std::getline(stream, field, ',');
        row.value = std::stod(field);
        rows.push_back(row);
    }
    return rows;
}

TEST(MoCompression, PreservesFrobeniusNormAndNonnegativity) {
    Eigen::MatrixXd magnitude(2, 3);
    magnitude << 0.0, 1.0, 4.0, 9.0, 16.0, 25.0;
    const auto compressed = cppnmf::compress_mo_input(magnitude, true, 0.6);
    EXPECT_NEAR(compressed.norm(), magnitude.norm(), 1e-12);
    EXPECT_GE(compressed.minCoeff(), 0.0);
    EXPECT_TRUE(cppnmf::compress_mo_input(magnitude, false, 0.6).isApprox(magnitude));
}

TEST(FeaturesGolden, MatchesMatlabDifficultyAndMoCompression) {
    const std::string directory = CPPNMF_TEST_DATA_DIR;
    const auto raw = read_column(directory + "/raw_signal.csv");
    const auto prepared = cppnmf::prepare_signal(raw, 20000.0);
    const auto stft = cppnmf::compute_stft(prepared.samples, 20000.0);
    const auto difficulty = cppnmf::compute_signal_difficulty(
        prepared.samples, 20000.0, stft.magnitude);
    const auto expected = read_summary(directory + "/feature_summary.csv");
    ASSERT_EQ(expected.size(), 5U);
    EXPECT_NEAR(difficulty.q_nsd, expected[0], 2e-10);
    EXPECT_NEAR(difficulty.components.time_frequency_entropy, expected[1], 2e-10);
    EXPECT_NEAR(difficulty.components.periodic_clarity, expected[2], 2e-10);
    EXPECT_NEAR(difficulty.components.periodic_difficulty, expected[3], 2e-10);
    EXPECT_NEAR(difficulty.components.envelope_spectral_flatness, expected[4], 2e-10);

    const auto features = cppnmf::compress_mo_input(stft.magnitude, true, 0.6);
    for (const auto& row : read_feature_signature(
             directory + "/mo_feature_signature.csv")) {
        EXPECT_NEAR(features(row.row, row.column), row.value, 2e-10);
    }
}

}  // namespace
