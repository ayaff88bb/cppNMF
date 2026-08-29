#include "cppnmf/pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    constexpr double sample_rate_hz = 20000.0;
    constexpr double pi = 3.141592653589793238462643383279502884;
    std::vector<double> signal(20000, 0.0);
    for (std::size_t index = 0; index < signal.size(); ++index) {
        const double time = static_cast<double>(index) / sample_rate_hz;
        signal[index] = 0.2 * std::sin(2.0 * pi * 240.0 * time);
        if (index % 250 < 15) {
            signal[index] +=
                std::exp(-static_cast<double>(index % 250) / 5.0) *
                std::sin(2.0 * pi * 2200.0 * time);
        }
    }

    cppnmf::PipelineConfig config;
    config.dnmf.pretrain_max_iterations = 30;
    config.dnmf.finetune_max_iterations = 50;
    config.dnmf.inner_updates = 3;

    constexpr int repetitions = 3;
    std::vector<double> totals;
    std::vector<double> solvers;
    for (int run = 0; run < repetitions; ++run) {
        const auto result = cppnmf::run_pipeline(signal, sample_rate_hz, config);
        totals.push_back(result.timings.total_ms);
        solvers.push_back(result.timings.solver_ms);
    }
    std::sort(totals.begin(), totals.end());
    std::sort(solvers.begin(), solvers.end());
    std::cout << std::fixed << std::setprecision(3)
              << "samples,stft_bins,stft_frames,repetitions,median_solver_ms,median_total_ms\n"
              << signal.size() << ",512,995," << repetitions << ','
              << solvers[repetitions / 2] << ',' << totals[repetitions / 2] << '\n';
    return 0;
}
