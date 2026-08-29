#include "cppnmf/io.hpp"

#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cppnmf {
namespace {

bool parse_number(const std::string& text, double& value) {
    std::size_t consumed = 0;
    try {
        value = std::stod(text, &consumed);
    } catch (const std::exception&) {
        return false;
    }
    while (consumed < text.size() &&
           std::isspace(static_cast<unsigned char>(text[consumed])) != 0) {
        ++consumed;
    }
    return consumed == text.size();
}

}  // namespace

std::vector<double> read_csv_signal(
    const std::filesystem::path& path,
    const std::size_t channel_zero_based) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open input CSV: " + path.string());
    }

    std::vector<double> signal;
    std::string line;
    std::size_t line_number = 0;
    bool skipped_header = false;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }
        std::stringstream stream(line);
        std::string field;
        std::size_t column = 0;
        bool found = false;
        while (std::getline(stream, field, ',')) {
            if (column == channel_zero_based) {
                double value = 0.0;
                if (!parse_number(field, value)) {
                    if (signal.empty() && !skipped_header) {
                        skipped_header = true;
                        found = true;
                        break;
                    }
                    throw std::runtime_error(
                        "non-numeric CSV field at line " +
                        std::to_string(line_number));
                }
                signal.push_back(value);
                found = true;
                break;
            }
            ++column;
        }
        if (!found) {
            throw std::runtime_error(
                "requested channel is missing at CSV line " +
                std::to_string(line_number));
        }
    }
    if (signal.size() < 2) {
        throw std::runtime_error("input CSV contains fewer than two samples");
    }
    return signal;
}

void write_csv(
    const std::filesystem::path& path,
    const std::vector<double>& values,
    const char* header) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot create output CSV: " + path.string());
    }
    output << header << '\n' << std::setprecision(17);
    for (const double value : values) {
        output << value << '\n';
    }
}

void write_csv(
    const std::filesystem::path& path,
    const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot create output CSV: " + path.string());
    }
    output << std::setprecision(17);
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
            if (column > 0) {
                output << ',';
            }
            output << matrix(row, column);
        }
        output << '\n';
    }
}

}  // namespace cppnmf
