#include "cppnmf/signal.hpp"

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace cppnmf {

PreparedSignal prepare_signal(
    const std::vector<double>& samples,
    const double sample_rate_hz) {
    if (samples.size() < 2) {
        throw std::invalid_argument("signal must contain at least two samples");
    }
    if (!std::isfinite(sample_rate_hz) || sample_rate_hz <= 0.0) {
        throw std::invalid_argument("sample rate must be finite and positive");
    }

    PreparedSignal result;
    result.samples.resize(samples.size());
    result.sample_rate_hz = sample_rate_hz;

    for (std::size_t index = 0; index < samples.size(); ++index) {
        result.samples[index] = std::isfinite(samples[index]) ? samples[index] : 0.0;
    }

    const double mean = std::accumulate(
                            result.samples.begin(),
                            result.samples.end(),
                            0.0) /
                        static_cast<double>(result.samples.size());
    for (double& sample : result.samples) {
        sample -= mean;
    }
    return result;
}

std::vector<double> periodic_hamming(const std::size_t length) {
    if (length == 0) {
        throw std::invalid_argument("window length must be positive");
    }

    constexpr double pi = 3.141592653589793238462643383279502884;
    std::vector<double> window(length);
    for (std::size_t index = 0; index < length; ++index) {
        window[index] = 0.54 -
                        0.46 * std::cos(
                                   2.0 * pi * static_cast<double>(index) /
                                   static_cast<double>(length));
    }
    return window;
}

}  // namespace cppnmf
