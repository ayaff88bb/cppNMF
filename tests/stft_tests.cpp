#include "cppnmf/stft.hpp"
#include "cppnmf/signal.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<double> make_test_signal(
    const std::size_t sample_count,
    const double sample_rate_hz) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    std::vector<double> signal(sample_count);
    for (std::size_t index = 0; index < sample_count; ++index) {
        const double time = static_cast<double>(index) / sample_rate_hz;
        signal[index] = 0.8 * std::sin(2.0 * pi * 250.0 * time) +
                        0.2 * std::cos(2.0 * pi * 1'100.0 * time);
    }
    return signal;
}

std::vector<double> read_single_column_csv(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open test data: " + path);
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

struct StftSignatureRow {
    std::size_t bin = 0;
    std::size_t frame = 0;
    double magnitude = 0.0;
    double real_part = 0.0;
    double imag_part = 0.0;
    double frequency_hz = 0.0;
    double time_seconds = 0.0;
};

std::vector<StftSignatureRow> read_stft_signature(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open test data: " + path);
    }

    std::string line;
    std::getline(input, line);
    std::vector<StftSignatureRow> rows;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream stream(line);
        std::string cell;
        StftSignatureRow row;
        std::getline(stream, cell, ',');
        row.bin = static_cast<std::size_t>(std::stoull(cell));
        std::getline(stream, cell, ',');
        row.frame = static_cast<std::size_t>(std::stoull(cell));
        std::getline(stream, cell, ',');
        row.magnitude = std::stod(cell);
        std::getline(stream, cell, ',');
        row.real_part = std::stod(cell);
        std::getline(stream, cell, ',');
        row.imag_part = std::stod(cell);
        std::getline(stream, cell, ',');
        row.frequency_hz = std::stod(cell);
        std::getline(stream, cell, ',');
        row.time_seconds = std::stod(cell);
        rows.push_back(row);
    }
    return rows;
}

}  // namespace

TEST(Stft, MatchesProductionDimensions) {
    const auto signal = make_test_signal(20'000, 20'000.0);
    const auto result = cppnmf::compute_stft(signal, 20'000.0);

    EXPECT_EQ(result.magnitude.rows(), 512);
    EXPECT_EQ(result.magnitude.cols(), 995);
    EXPECT_EQ(result.spectrum.rows(), 513);
    EXPECT_EQ(result.spectrum.cols(), 995);
    EXPECT_EQ(result.padding_length, 0U);
}

TEST(Stft, PadsToAnIntegralNumberOfFrames) {
    cppnmf::StftConfig config;
    config.fft_length = 16;
    config.window_length = 8;
    config.hop_length = 3;
    const std::vector<double> signal(14, 1.0);

    const auto result = cppnmf::compute_stft(signal, 1'000.0, config);

    EXPECT_EQ(result.padding_length, 0U);
    EXPECT_EQ(result.magnitude.cols(), 3);
}

TEST(Stft, RoundTripHasNearMachinePrecisionError) {
    cppnmf::StftConfig config;
    config.fft_length = 128;
    config.window_length = 64;
    config.hop_length = 16;
    const auto signal = make_test_signal(2'048, 8'000.0);

    const auto result = cppnmf::compute_stft(signal, 8'000.0, config);
    const auto reconstructed = cppnmf::inverse_stft(result);

    EXPECT_EQ(reconstructed.size(), signal.size());
    EXPECT_LT(cppnmf::relative_l2_error(signal, reconstructed), 1e-11);
}

TEST(Stft, RejectsInvalidConfiguration) {
    const std::vector<double> signal(128, 0.0);
    cppnmf::StftConfig config;
    config.fft_length = 15;

    EXPECT_THROW(
        cppnmf::compute_stft(signal, 1'000.0, config),
        std::invalid_argument);
}

TEST(StftGolden, MatchesMatlabSyntheticImpactSignal) {
    const std::string data_directory = CPPNMF_TEST_DATA_DIR;
    const auto raw_signal = read_single_column_csv(
        data_directory + "/raw_signal.csv");
    const auto expected_prepared = read_single_column_csv(
        data_directory + "/prepared_signal.csv");
    const auto expected_window = read_single_column_csv(
        data_directory + "/periodic_hamming.csv");
    const auto signature = read_stft_signature(
        data_directory + "/stft_signature.csv");

    const auto prepared = cppnmf::prepare_signal(raw_signal, 20'000.0);
    ASSERT_EQ(prepared.samples.size(), expected_prepared.size());
    EXPECT_LT(
        cppnmf::relative_l2_error(expected_prepared, prepared.samples),
        1e-14);

    const auto result = cppnmf::compute_stft(
        prepared.samples, prepared.sample_rate_hz, cppnmf::StftConfig{});
    ASSERT_EQ(result.window.size(), expected_window.size());
    for (std::size_t index = 0; index < result.window.size(); ++index) {
        EXPECT_NEAR(result.window[index], expected_window[index], 1e-14);
    }

    ASSERT_EQ(signature.size(), 81U);
    for (const auto& row : signature) {
        const auto bin = static_cast<Eigen::Index>(row.bin);
        const auto frame = static_cast<Eigen::Index>(row.frame);
        const std::complex<double> value =
            result.magnitude(bin, frame) * result.phase_positive(bin, frame);
        EXPECT_NEAR(result.magnitude(bin, frame), row.magnitude, 1e-10);
        EXPECT_NEAR(value.real(), row.real_part, 1e-10);
        EXPECT_NEAR(value.imag(), row.imag_part, 1e-10);
        EXPECT_NEAR(result.frequency_hz[bin], row.frequency_hz, 1e-12);
        EXPECT_NEAR(result.time_seconds[frame], row.time_seconds, 1e-12);
    }
}

TEST(StftGolden, IdentityReconstructionMatchesMatlab) {
    const std::string data_directory = CPPNMF_TEST_DATA_DIR;
    const auto prepared = read_single_column_csv(
        data_directory + "/prepared_signal.csv");
    const auto expected_reconstruction = read_single_column_csv(
        data_directory + "/stft_identity_reconstruction.csv");

    const auto result = cppnmf::compute_stft(
        prepared, 20'000.0, cppnmf::StftConfig{});
    const auto reconstructed = cppnmf::inverse_stft(result);

    ASSERT_EQ(reconstructed.size(), expected_reconstruction.size());
    EXPECT_LT(
        cppnmf::relative_l2_error(expected_reconstruction, reconstructed),
        1e-10);
}
