#include "cppnmf/features.hpp"

#include <kiss_fft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace cppnmf {
namespace {

struct ComplexPlanDeleter {
    void operator()(kiss_fft_cfg plan) const noexcept {
        kiss_fft_free(plan);
    }
};

using ComplexPlan = std::unique_ptr<kiss_fft_state, ComplexPlanDeleter>;

std::size_t next_power_of_two(std::size_t value) {
    std::size_t result = 1;
    while (result < value) {
        result <<= 1U;
    }
    return result;
}

std::vector<double> centered_finite_signal(const std::vector<double>& input) {
    std::vector<double> signal(input.size(), 0.0);
    for (std::size_t index = 0; index < input.size(); ++index) {
        signal[index] = std::isfinite(input[index]) ? input[index] : 0.0;
    }
    if (!signal.empty()) {
        const double mean = std::accumulate(signal.begin(), signal.end(), 0.0) /
                            static_cast<double>(signal.size());
        for (double& sample : signal) {
            sample -= mean;
        }
    }
    return signal;
}

double squared_norm(const std::vector<double>& values) {
    return std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
}

std::vector<double> analytic_envelope(const std::vector<double>& signal) {
    const std::size_t length = signal.size();
    if (length == 0) {
        return {};
    }

    ComplexPlan forward(kiss_fft_alloc(static_cast<int>(length), 0, nullptr, nullptr));
    ComplexPlan inverse(kiss_fft_alloc(static_cast<int>(length), 1, nullptr, nullptr));
    if (!forward || !inverse) {
        throw std::runtime_error("failed to create KISS FFT Hilbert-transform plans");
    }

    std::vector<kiss_fft_cpx> time(length);
    std::vector<kiss_fft_cpx> spectrum(length);
    std::vector<kiss_fft_cpx> analytic(length);
    for (std::size_t index = 0; index < length; ++index) {
        time[index].r = signal[index];
        time[index].i = 0.0;
    }
    kiss_fft(forward.get(), time.data(), spectrum.data());

    for (std::size_t bin = 0; bin < length; ++bin) {
        double multiplier = 0.0;
        if (bin == 0) {
            multiplier = 1.0;
        } else if (length % 2 == 0 && bin == length / 2) {
            multiplier = 1.0;
        } else if (bin <= (length - 1) / 2) {
            multiplier = 2.0;
        }
        spectrum[bin].r *= multiplier;
        spectrum[bin].i *= multiplier;
    }
    kiss_fft(inverse.get(), spectrum.data(), analytic.data());

    std::vector<double> envelope(length);
    const double scale = 1.0 / static_cast<double>(length);
    for (std::size_t index = 0; index < length; ++index) {
        envelope[index] = std::hypot(analytic[index].r, analytic[index].i) * scale;
    }
    return envelope;
}

double time_frequency_entropy(const Eigen::Ref<const Eigen::MatrixXd>& magnitude) {
    if (magnitude.size() == 0) {
        return 0.0;
    }
    long double total = 0.0L;
    for (Eigen::Index index = 0; index < magnitude.size(); ++index) {
        const double value = magnitude.data()[index];
        if (std::isfinite(value)) {
            total += static_cast<long double>(value) * value;
        }
    }
    if (total <= 0.0L) {
        return 0.0;
    }

    long double entropy = 0.0L;
    for (Eigen::Index index = 0; index < magnitude.size(); ++index) {
        const double value = magnitude.data()[index];
        if (!std::isfinite(value)) {
            continue;
        }
        const long double probability =
            static_cast<long double>(value) * value / total;
        if (probability > 0.0L) {
            entropy -= probability * std::log(probability);
        }
    }
    const long double denominator = std::log(
        static_cast<long double>(magnitude.size()) +
        std::numeric_limits<double>::epsilon());
    return std::clamp(static_cast<double>(entropy / denominator), 0.0, 1.0);
}

double envelope_periodic_clarity(
    const std::vector<double>& input,
    const double sample_rate_hz) {
    const auto signal = centered_finite_signal(input);
    if (signal.size() < 8 || squared_norm(signal) <= std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }

    auto envelope = analytic_envelope(signal);
    const double mean = std::accumulate(envelope.begin(), envelope.end(), 0.0) /
                        static_cast<double>(envelope.size());
    for (double& value : envelope) {
        value -= mean;
    }
    const double zero_lag = squared_norm(envelope);
    if (zero_lag <= std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }

    constexpr double minimum_frequency_hz = 5.0;
    constexpr double maximum_frequency_hz = 300.0;
    const std::size_t minimum_lag = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::floor(sample_rate_hz / maximum_frequency_hz)));
    const std::size_t maximum_lag = std::min<std::size_t>(
        envelope.size() - 1,
        static_cast<std::size_t>(std::ceil(sample_rate_hz / minimum_frequency_hz)));
    if (maximum_lag <= minimum_lag) {
        return 0.0;
    }

    double best = -std::numeric_limits<double>::infinity();
    for (std::size_t lag = minimum_lag; lag <= maximum_lag; ++lag) {
        double correlation = 0.0;
        for (std::size_t index = 0; index + lag < envelope.size(); ++index) {
            correlation += envelope[index] * envelope[index + lag];
        }
        best = std::max(best, correlation / zero_lag);
    }
    return std::clamp(best, 0.0, 1.0);
}

double envelope_spectral_flatness(
    const std::vector<double>& input,
    const double sample_rate_hz) {
    const auto signal = centered_finite_signal(input);
    if (signal.size() < 8 || squared_norm(signal) <= std::numeric_limits<double>::epsilon()) {
        return 1.0;
    }

    auto envelope = analytic_envelope(signal);
    const double mean = std::accumulate(envelope.begin(), envelope.end(), 0.0) /
                        static_cast<double>(envelope.size());
    for (double& value : envelope) {
        value -= mean;
    }

    const std::size_t fft_length = next_power_of_two(envelope.size());
    ComplexPlan forward(kiss_fft_alloc(
        static_cast<int>(fft_length), 0, nullptr, nullptr));
    if (!forward) {
        throw std::runtime_error("failed to create KISS FFT flatness plan");
    }
    std::vector<kiss_fft_cpx> time(fft_length);
    std::vector<kiss_fft_cpx> spectrum(fft_length);
    for (std::size_t index = 0; index < envelope.size(); ++index) {
        time[index].r = envelope[index];
    }
    kiss_fft(forward.get(), time.data(), spectrum.data());

    constexpr double minimum_frequency_hz = 5.0;
    constexpr double maximum_frequency_hz = 1000.0;
    long double log_sum = 0.0L;
    long double power_sum = 0.0L;
    std::size_t count = 0;
    const double epsilon = std::numeric_limits<double>::epsilon();
    for (std::size_t bin = 0; bin <= fft_length / 2; ++bin) {
        const double frequency = static_cast<double>(bin) * sample_rate_hz /
                                 static_cast<double>(fft_length);
        if (frequency < minimum_frequency_hz || frequency > maximum_frequency_hz) {
            continue;
        }
        const double power = spectrum[bin].r * spectrum[bin].r +
                             spectrum[bin].i * spectrum[bin].i;
        if (std::isfinite(power) && power > 0.0) {
            log_sum += std::log(power + epsilon);
            power_sum += power + epsilon;
            ++count;
        }
    }
    if (count == 0) {
        return 1.0;
    }
    const long double geometric_mean = std::exp(log_sum / count);
    const long double arithmetic_mean = power_sum / count;
    return std::clamp(
        static_cast<double>(geometric_mean / (arithmetic_mean + epsilon)),
        0.0,
        1.0);
}

}  // namespace

DifficultyResult compute_signal_difficulty(
    const std::vector<double>& signal,
    const double sample_rate_hz,
    const Eigen::Ref<const Eigen::MatrixXd>& stft_magnitude) {
    if (!std::isfinite(sample_rate_hz) || sample_rate_hz <= 0.0) {
        throw std::invalid_argument("sample rate must be finite and positive");
    }
    if (signal.empty() || stft_magnitude.size() == 0) {
        throw std::invalid_argument("signal and STFT magnitude must not be empty");
    }

    DifficultyResult result;
    result.components.time_frequency_entropy = time_frequency_entropy(stft_magnitude);
    result.components.periodic_clarity = envelope_periodic_clarity(signal, sample_rate_hz);
    result.components.periodic_difficulty = 1.0 - result.components.periodic_clarity;
    result.components.envelope_spectral_flatness =
        envelope_spectral_flatness(signal, sample_rate_hz);
    result.q_nsd = std::clamp(
        0.4 * result.components.time_frequency_entropy +
            0.4 * result.components.periodic_difficulty +
            0.2 * result.components.envelope_spectral_flatness,
        0.0,
        1.0);
    return result;
}

Eigen::MatrixXd compress_mo_input(
    const Eigen::Ref<const Eigen::MatrixXd>& magnitude,
    const bool enabled,
    const double gamma) {
    if (magnitude.size() == 0) {
        throw std::invalid_argument("magnitude matrix must not be empty");
    }
    if (!std::isfinite(gamma) || gamma <= 0.0) {
        throw std::invalid_argument("MO gamma must be finite and positive");
    }
    if (!enabled) {
        return magnitude;
    }

    const double epsilon = std::numeric_limits<double>::epsilon();
    const double scale = magnitude.maxCoeff() + epsilon;
    Eigen::MatrixXd compressed =
        ((magnitude.array() / scale) + epsilon).pow(gamma).matrix();
    const double magnitude_norm = magnitude.norm();
    const double compressed_norm = compressed.norm();
    compressed *= magnitude_norm / (compressed_norm + epsilon);
    return compressed;
}

}  // namespace cppnmf
