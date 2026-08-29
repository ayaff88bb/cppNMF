#pragma once

#include <vector>

namespace cppnmf {

struct PreparedSignal {
    std::vector<double> samples;
    double sample_rate_hz = 0.0;
};

PreparedSignal prepare_signal(
    const std::vector<double>& samples,
    double sample_rate_hz);

std::vector<double> periodic_hamming(std::size_t length);

}  // namespace cppnmf
