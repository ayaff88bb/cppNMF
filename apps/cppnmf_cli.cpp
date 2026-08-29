#include "cppnmf/dnmf.hpp"
#include "cppnmf/io.hpp"
#include "cppnmf/pipeline.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    std::filesystem::path input;
    std::filesystem::path output = "results";
    double sample_rate_hz = 0.0;
    std::size_t channel_zero_based = 0;
    std::string preset = "quick";
    cppnmf::PipelineConfig pipeline{};
    bool demo = false;
    bool validation_export = false;
};

void print_help() {
    std::cout
        << "cppNMF - Deep nsNMF impact-feature extraction\n\n"
        << "Usage:\n"
        << "  cppnmf_cli --input signal.csv --sample-rate 20000 [options]\n"
        << "  cppnmf_cli --demo [options]\n\n"
        << "Options:\n"
        << "  --output DIR             Output directory (default: results)\n"
        << "  --channel N              One-based CSV column (default: 1)\n"
        << "  --preset quick|reference Quick demo or MATLAB iteration budget\n"
        << "  --ranks 40,16,4          Deep NMF layer ranks\n"
        << "  --theta 0.1,0.4,0.7      Layer smoothing parameters\n"
        << "  --mo-gamma 0.6           Magnitude-order compression exponent\n"
        << "  --lambda-rep 0.01        Repetition reward; 0 disables it\n"
        << "  --pretrain-iterations N  Override per-layer pretraining budget\n"
        << "  --finetune-iterations N  Override global fine-tuning budget\n"
        << "  --validation-export      Export full intermediate matrices for MATLAB parity checks\n"
        << "  --help                    Show this help\n\n"
        << "MATLAB .mat input can be converted with tools/export_mat_to_csv.m.\n";
}

std::string require_value(const int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + argv[index]);
    }
    return argv[++index];
}

template <typename Value>
std::vector<Value> parse_list(const std::string& text) {
    std::vector<Value> values;
    std::stringstream stream(text);
    std::string field;
    while (std::getline(stream, field, ',')) {
        std::stringstream item(field);
        Value value{};
        item >> value;
        if (!item || !item.eof()) {
            throw std::invalid_argument("invalid comma-separated value: " + field);
        }
        values.push_back(value);
    }
    if (values.empty()) {
        throw std::invalid_argument("comma-separated list must not be empty");
    }
    return values;
}

CliOptions parse_arguments(const int argc, char** argv) {
    CliOptions options;
    bool pretrain_overridden = false;
    bool finetune_overridden = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            print_help();
            std::exit(EXIT_SUCCESS);
        } else if (argument == "--demo") {
            options.demo = true;
        } else if (argument == "--input") {
            options.input = require_value(argc, argv, index);
        } else if (argument == "--output") {
            options.output = require_value(argc, argv, index);
        } else if (argument == "--sample-rate") {
            options.sample_rate_hz = std::stod(require_value(argc, argv, index));
        } else if (argument == "--channel") {
            const auto channel = std::stoull(require_value(argc, argv, index));
            if (channel == 0) {
                throw std::invalid_argument("--channel is one-based and must be positive");
            }
            options.channel_zero_based = static_cast<std::size_t>(channel - 1);
        } else if (argument == "--preset") {
            options.preset = require_value(argc, argv, index);
        } else if (argument == "--ranks") {
            options.pipeline.dnmf.ranks =
                parse_list<std::size_t>(require_value(argc, argv, index));
        } else if (argument == "--theta") {
            options.pipeline.dnmf.theta =
                parse_list<double>(require_value(argc, argv, index));
        } else if (argument == "--mo-gamma") {
            options.pipeline.mo_gamma = std::stod(require_value(argc, argv, index));
        } else if (argument == "--lambda-rep") {
            options.pipeline.dnmf.lambda_repetition =
                std::stod(require_value(argc, argv, index));
        } else if (argument == "--pretrain-iterations") {
            options.pipeline.dnmf.pretrain_max_iterations =
                std::stoull(require_value(argc, argv, index));
            pretrain_overridden = true;
        } else if (argument == "--finetune-iterations") {
            options.pipeline.dnmf.finetune_max_iterations =
                std::stoull(require_value(argc, argv, index));
            finetune_overridden = true;
        } else if (argument == "--validation-export") {
            options.validation_export = true;
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }

    if (options.preset == "quick") {
        if (!pretrain_overridden) {
            options.pipeline.dnmf.pretrain_max_iterations = 30;
        }
        if (!finetune_overridden) {
            options.pipeline.dnmf.finetune_max_iterations = 50;
        }
        options.pipeline.dnmf.inner_updates = 3;
    } else if (options.preset != "reference") {
        throw std::invalid_argument("--preset must be quick or reference");
    }
    if (!options.demo && options.input.empty()) {
        throw std::invalid_argument("provide --input signal.csv or use --demo");
    }
    if (!options.demo && options.sample_rate_hz <= 0.0) {
        throw std::invalid_argument("--sample-rate must be positive");
    }
    if (options.pipeline.dnmf.ranks.size() != options.pipeline.dnmf.theta.size()) {
        throw std::invalid_argument("--ranks and --theta must have equal lengths");
    }
    return options;
}

std::vector<double> demo_signal(const double sample_rate_hz) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    const std::size_t sample_count = static_cast<std::size_t>(sample_rate_hz / 4.0);
    std::vector<double> signal(sample_count, 0.0);
    for (std::size_t index = 0; index < signal.size(); ++index) {
        const double time = static_cast<double>(index) / sample_rate_hz;
        const double carrier = std::sin(2.0 * pi * 1800.0 * time);
        const double impulse_train = index % 200 < 12
                                         ? std::exp(-static_cast<double>(index % 200) / 4.0)
                                         : 0.0;
        signal[index] = 0.15 * std::sin(2.0 * pi * 260.0 * time) +
                        impulse_train * carrier;
    }
    return signal;
}

void write_summary(
    const std::filesystem::path& path,
    const CliOptions& options,
    const cppnmf::PipelineResult& result) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot create summary: " + path.string());
    }
    output << std::setprecision(12)
           << "{\n"
           << "  \"sample_rate_hz\": " << result.prepared.sample_rate_hz << ",\n"
           << "  \"sample_count\": " << result.prepared.samples.size() << ",\n"
           << "  \"stft_bins\": " << result.stft.magnitude.rows() << ",\n"
           << "  \"stft_frames\": " << result.stft.magnitude.cols() << ",\n"
           << "  \"q_nsd\": " << result.difficulty.q_nsd << ",\n"
           << "  \"tf_entropy\": "
           << result.difficulty.components.time_frequency_entropy << ",\n"
           << "  \"periodic_clarity\": "
           << result.difficulty.components.periodic_clarity << ",\n"
           << "  \"envelope_flatness\": "
           << result.difficulty.components.envelope_spectral_flatness << ",\n"
           << "  \"relative_solver_reconstruction_error\": "
           << cppnmf::relative_frobenius_error(
                  result.features, cppnmf::reconstruct_dnmf(result.dnmf))
           << ",\n"
           << "  \"relative_interpretable_reconstruction_error\": "
           << cppnmf::relative_frobenius_error(
                  result.features, result.reconstructed_magnitude)
           << ",\n"
           << "  \"preset\": \"" << options.preset << "\",\n"
           << "  \"timings_ms\": {\n"
           << "    \"preprocessing\": " << result.timings.preprocessing_ms << ",\n"
           << "    \"stft\": " << result.timings.stft_ms << ",\n"
           << "    \"features\": " << result.timings.feature_ms << ",\n"
           << "    \"solver\": " << result.timings.solver_ms << ",\n"
           << "    \"reconstruction\": " << result.timings.reconstruction_ms << ",\n"
           << "    \"total\": " << result.timings.total_ms << "\n"
           << "  }\n"
           << "}\n";
}

void write_results(
    const CliOptions& options,
    const cppnmf::PipelineResult& result) {
    std::filesystem::create_directories(options.output);
    cppnmf::write_csv(
        options.output / "reconstructed_signal.csv",
        result.reconstructed_signal,
        "amplitude");
    cppnmf::write_csv(
        options.output / "top_basis.csv",
        cppnmf::cumulative_bases(result.dnmf).back());
    cppnmf::write_csv(
        options.output / "top_activation.csv",
        result.dnmf.activation.back());
    cppnmf::write_csv(
        options.output / "objective_history.csv",
        result.dnmf.objective_history,
        "objective");
    const auto cumulative = cppnmf::cumulative_bases(result.dnmf);
    const auto layer_reconstructed = cppnmf::layer_reconstructions(result.dnmf);
    for (std::size_t layer = 0; layer < result.dnmf.basis.size(); ++layer) {
        const std::string suffix = std::to_string(layer + 1) + ".csv";
        cppnmf::write_csv(
            options.output / ("basis_layer_" + suffix),
            result.dnmf.basis[layer]);
        cppnmf::write_csv(
            options.output / ("activation_layer_" + suffix),
            result.dnmf.activation[layer]);
        cppnmf::write_csv(
            options.output / ("smoothing_layer_" + suffix),
            result.dnmf.smoothing[layer]);
        if (options.validation_export) {
            cppnmf::write_csv(
                options.output / ("cumulative_basis_layer_" + suffix),
                cumulative[layer]);
            cppnmf::write_csv(
                options.output / ("layer_reconstruction_" + suffix),
                layer_reconstructed[layer]);
        }
    }
    for (std::size_t component = 0;
         component < result.subspaces.signals.size();
         ++component) {
        cppnmf::write_csv(
            options.output /
                ("subspace_" + std::to_string(component + 1) + ".csv"),
            result.subspaces.signals[component],
            "amplitude");
        if (options.validation_export) {
            cppnmf::write_csv(
                options.output /
                    ("subspace_magnitude_" + std::to_string(component + 1) + ".csv"),
                result.subspaces.magnitudes[component]);
        }
    }
    if (options.validation_export) {
        cppnmf::write_csv(
            options.output / "prepared_signal.csv",
            result.prepared.samples,
            "amplitude");
        cppnmf::write_csv(
            options.output / "stft_magnitude.csv",
            result.stft.magnitude);
        cppnmf::write_csv(
            options.output / "stft_frequency_hz.csv",
            std::vector<double>(
                result.stft.frequency_hz.data(),
                result.stft.frequency_hz.data() + result.stft.frequency_hz.size()),
            "frequency_hz");
        cppnmf::write_csv(
            options.output / "stft_time_seconds.csv",
            std::vector<double>(
                result.stft.time_seconds.data(),
                result.stft.time_seconds.data() + result.stft.time_seconds.size()),
            "time_seconds");
        cppnmf::write_csv(options.output / "features.csv", result.features);
        cppnmf::write_csv(
            options.output / "solver_reconstruction.csv",
            cppnmf::reconstruct_dnmf(result.dnmf));
        const auto initialization = cppnmf::nndsvd(
            result.features, options.pipeline.dnmf.ranks.front());
        cppnmf::write_csv(
            options.output / "nndsvd_basis_layer_1.csv",
            initialization.basis);
        cppnmf::write_csv(
            options.output / "nndsvd_activation_layer_1.csv",
            initialization.activation);
    }
    write_summary(options.output / "summary.json", options, result);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_arguments(argc, argv);
        double sample_rate_hz = options.sample_rate_hz;
        std::vector<double> signal;
        if (options.demo) {
            sample_rate_hz = 20000.0;
            signal = demo_signal(sample_rate_hz);
        } else {
            if (options.input.extension() == ".mat") {
                throw std::invalid_argument(
                    "MAT files are research interchange files; convert with "
                    "tools/export_mat_to_csv.m before standalone processing");
            }
            signal = cppnmf::read_csv_signal(
                options.input, options.channel_zero_based);
        }

        std::cout << "cppNMF pipeline\n"
                  << "  samples: " << signal.size() << '\n'
                  << "  sample rate: " << sample_rate_hz << " Hz\n"
                  << "  preset: " << options.preset << '\n'
                  << "  processing..." << std::endl;
        const auto result = cppnmf::run_pipeline(
            signal, sample_rate_hz, options.pipeline);
        write_results(options, result);

        std::cout << std::fixed << std::setprecision(4)
                  << "  q_NSD: " << result.difficulty.q_nsd << '\n'
                  << "  solver reconstruction error: "
                  << cppnmf::relative_frobenius_error(
                         result.features, cppnmf::reconstruct_dnmf(result.dnmf))
                  << '\n'
                  << "  interpretable reconstruction error: "
                  << cppnmf::relative_frobenius_error(
                         result.features, result.reconstructed_magnitude)
                  << '\n'
                  << "  solver time: " << result.timings.solver_ms << " ms\n"
                  << "  total time: " << result.timings.total_ms << " ms\n"
                  << "  results: " << std::filesystem::absolute(options.output).string()
                  << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "cppNMF error: " << error.what() << '\n'
                  << "Run cppnmf_cli --help for usage.\n";
        return EXIT_FAILURE;
    }
}
