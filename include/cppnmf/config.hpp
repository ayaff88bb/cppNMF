#pragma once

#include <cstddef>

namespace cppnmf {

struct StftConfig {
    std::size_t fft_length = 1024;
    std::size_t window_length = 120;
    std::size_t hop_length = 20;
};

struct ProcessingConfig {
    StftConfig stft{};
    std::size_t signal_channel = 0;
};

}  // namespace cppnmf
