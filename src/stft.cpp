#include "cppnmf/stft.hpp"

#include "cppnmf/signal.hpp"

#include <kiss_fftr.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace cppnmf {
namespace {

struct ForwardPlanDeleter {
    void operator()(kiss_fftr_cfg plan) const noexcept {
        kiss_fft_free(plan);
    }
};

using ForwardPlan = std::unique_ptr<kiss_fftr_state, ForwardPlanDeleter>;

void validate_stft_config(const StftConfig& config) {
    if (config.fft_length == 0 || config.fft_length % 2 != 0) {
        throw std::invalid_argument("FFT length must be a positive even number");
    }
    if (config.window_length == 0 || config.window_length > config.fft_length) {
        throw std::invalid_argument("window length must be in [1, FFT length]");
    }
    if (config.hop_length == 0 || config.hop_length > config.window_length) {
        throw std::invalid_argument("hop length must be in [1, window length]");
    }
}

std::size_t required_padding(
    const std::size_t signal_length,
    const std::size_t window_length,
    const std::size_t hop_length) {
    if (signal_length < window_length) {
        return window_length - signal_length;
    }
    const std::size_t remainder = (signal_length - window_length) % hop_length;
    return remainder == 0 ? 0 : hop_length - remainder;
}

}  // namespace

StftResult compute_stft(
    const std::vector<double>& signal,
    const double sample_rate_hz,
    const StftConfig& config) {
    validate_stft_config(config);
    if (signal.empty()) {
        throw std::invalid_argument("signal must not be empty");
    }
    if (!std::isfinite(sample_rate_hz) || sample_rate_hz <= 0.0) {
        throw std::invalid_argument("sample rate must be finite and positive");
    }

    const std::size_t padding_length = required_padding(
        signal.size(), config.window_length, config.hop_length);
    std::vector<double> padded = signal;
    padded.resize(signal.size() + padding_length, 0.0);

    const std::size_t frame_count =
        1 + (padded.size() - config.window_length) / config.hop_length;
    const std::size_t rfft_bins = config.fft_length / 2 + 1;
    const std::size_t algorithm_bins = config.fft_length / 2;

    StftResult result;
    result.spectrum.resize(
        static_cast<Eigen::Index>(rfft_bins),
        static_cast<Eigen::Index>(frame_count));
    result.magnitude.resize(
        static_cast<Eigen::Index>(algorithm_bins),
        static_cast<Eigen::Index>(frame_count));
    result.phase_positive.resize(
        static_cast<Eigen::Index>(algorithm_bins),
        static_cast<Eigen::Index>(frame_count));
    result.frequency_hz.resize(static_cast<Eigen::Index>(algorithm_bins));
    result.time_seconds.resize(static_cast<Eigen::Index>(frame_count));
    result.window = periodic_hamming(config.window_length);
    result.original_length = signal.size();
    result.padding_length = padding_length;
    result.hop_length = config.hop_length;
    result.fft_length = config.fft_length;

    for (std::size_t bin = 0; bin < algorithm_bins; ++bin) {
        result.frequency_hz[static_cast<Eigen::Index>(bin)] =
            static_cast<double>(bin + 1) * sample_rate_hz /
            static_cast<double>(config.fft_length);
    }
    result.zero_frequency_bin.resize(static_cast<Eigen::Index>(frame_count));

    ForwardPlan plan(kiss_fftr_alloc(
        static_cast<int>(config.fft_length), 0, nullptr, nullptr));
    if (!plan) {
        throw std::runtime_error("failed to create KISS FFT forward plan");
    }

    std::vector<double> frame(config.fft_length, 0.0);
    std::vector<kiss_fft_cpx> bins(rfft_bins);
    for (std::size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
        std::fill(frame.begin(), frame.end(), 0.0);
        const std::size_t frame_start = frame_index * config.hop_length;
        for (std::size_t sample = 0; sample < config.window_length; ++sample) {
            frame[sample] = padded[frame_start + sample] * result.window[sample];
        }

        kiss_fftr(plan.get(), frame.data(), bins.data());
        for (std::size_t bin = 0; bin < rfft_bins; ++bin) {
            const std::complex<double> value{bins[bin].r, bins[bin].i};
            result.spectrum(
                static_cast<Eigen::Index>(bin),
                static_cast<Eigen::Index>(frame_index)) = value;
            if (bin == 0) {
                result.zero_frequency_bin[static_cast<Eigen::Index>(frame_index)] = value;
            }
            if (bin > 0) {
                const std::size_t positive_index = bin - 1;
                const double magnitude = std::abs(value);
                result.magnitude(
                    static_cast<Eigen::Index>(positive_index),
                    static_cast<Eigen::Index>(frame_index)) = magnitude;
                result.phase_positive(
                    static_cast<Eigen::Index>(positive_index),
                    static_cast<Eigen::Index>(frame_index)) =
                    magnitude > 0.0 ? value / magnitude : std::complex<double>{1.0, 0.0};
            }
        }
        result.time_seconds[static_cast<Eigen::Index>(frame_index)] =
            (static_cast<double>(frame_start) +
             static_cast<double>(config.window_length) / 2.0) /
            sample_rate_hz;
    }

    return result;
}

std::vector<double> inverse_stft(const StftResult& stft) {
    if (stft.fft_length == 0 || stft.hop_length == 0 || stft.window.empty()) {
        throw std::invalid_argument("STFT metadata is incomplete");
    }
    const std::size_t expected_bins = stft.fft_length / 2 + 1;
    if (stft.spectrum.rows() != static_cast<Eigen::Index>(expected_bins)) {
        throw std::invalid_argument("STFT spectrum has an unexpected row count");
    }

    const std::size_t frame_count = static_cast<std::size_t>(stft.spectrum.cols());
    const std::size_t padded_length = frame_count == 0
                                          ? 0
                                          : (frame_count - 1) * stft.hop_length +
                                                stft.window.size();
    if (frame_count == 0 || padded_length < stft.original_length) {
        throw std::invalid_argument("STFT spectrum does not contain enough frames");
    }

    ForwardPlan inverse_plan(kiss_fftr_alloc(
        static_cast<int>(stft.fft_length), 1, nullptr, nullptr));
    if (!inverse_plan) {
        throw std::runtime_error("failed to create KISS FFT inverse plan");
    }

    std::vector<double> output(padded_length, 0.0);
    std::vector<double> weight(padded_length, 0.0);
    std::vector<double> frame(stft.fft_length, 0.0);
    std::vector<kiss_fft_cpx> bins(expected_bins);

    for (std::size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
        for (std::size_t bin = 0; bin < expected_bins; ++bin) {
            const std::complex<double> value = stft.spectrum(
                static_cast<Eigen::Index>(bin),
                static_cast<Eigen::Index>(frame_index));
            bins[bin].r = value.real();
            bins[bin].i = value.imag();
        }

        kiss_fftri(inverse_plan.get(), bins.data(), frame.data());
        const std::size_t frame_start = frame_index * stft.hop_length;
        for (std::size_t sample = 0; sample < stft.window.size(); ++sample) {
            const double window_value = stft.window[sample];
            output[frame_start + sample] +=
                frame[sample] / static_cast<double>(stft.fft_length) * window_value;
            weight[frame_start + sample] += window_value * window_value;
        }
    }

    constexpr double epsilon = 1e-14;
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (weight[index] > epsilon) {
            output[index] /= weight[index];
        }
    }
    output.resize(stft.original_length);
    return output;
}

double relative_l2_error(
    const std::vector<double>& expected,
    const std::vector<double>& actual) {
    if (expected.size() != actual.size()) {
        throw std::invalid_argument("vectors must have the same length");
    }

    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double difference = actual[index] - expected[index];
        numerator += difference * difference;
        denominator += expected[index] * expected[index];
    }
    if (denominator == 0.0) {
        return std::sqrt(numerator);
    }
    return std::sqrt(numerator / denominator);
}

}  // namespace cppnmf
