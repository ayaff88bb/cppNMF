#include "cppnmf/pipeline.hpp"

#include <chrono>

namespace cppnmf {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

PipelineResult run_pipeline(
    const std::vector<double>& signal,
    const double sample_rate_hz,
    const PipelineConfig& config) {
    PipelineResult result;
    const auto total_start = Clock::now();

    auto stage_start = Clock::now();
    result.prepared = prepare_signal(signal, sample_rate_hz);
    auto stage_end = Clock::now();
    result.timings.preprocessing_ms = milliseconds(stage_start, stage_end);

    stage_start = Clock::now();
    result.stft = compute_stft(
        result.prepared.samples,
        result.prepared.sample_rate_hz,
        config.processing.stft);
    stage_end = Clock::now();
    result.timings.stft_ms = milliseconds(stage_start, stage_end);

    stage_start = Clock::now();
    result.difficulty = compute_signal_difficulty(
        result.prepared.samples,
        result.prepared.sample_rate_hz,
        result.stft.magnitude);
    result.features = compress_mo_input(
        result.stft.magnitude,
        config.use_mo_compression,
        config.mo_gamma);
    stage_end = Clock::now();
    result.timings.feature_ms = milliseconds(stage_start, stage_end);

    stage_start = Clock::now();
    result.dnmf = deep_nsnmf(result.features, config.dnmf);
    stage_end = Clock::now();
    result.timings.solver_ms = milliseconds(stage_start, stage_end);

    stage_start = Clock::now();
    result.reconstructed_magnitude = layer_reconstructions(result.dnmf).back();
    result.reconstructed_signal = reconstruct_from_magnitude(
        result.stft, result.reconstructed_magnitude);
    const auto bases = cumulative_bases(result.dnmf);
    result.subspaces = reconstruct_subspaces(
        bases.back(),
        result.dnmf.activation.back(),
        result.stft.magnitude,
        result.stft);
    stage_end = Clock::now();
    result.timings.reconstruction_ms = milliseconds(stage_start, stage_end);
    result.timings.total_ms = milliseconds(total_start, stage_end);
    return result;
}

}  // namespace cppnmf
