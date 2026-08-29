#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <filesystem>
#include <vector>

namespace cppnmf {

std::vector<double> read_csv_signal(
    const std::filesystem::path& path,
    std::size_t channel_zero_based = 0);

void write_csv(
    const std::filesystem::path& path,
    const std::vector<double>& values,
    const char* header = "value");

void write_csv(
    const std::filesystem::path& path,
    const Eigen::Ref<const Eigen::MatrixXd>& matrix);

}  // namespace cppnmf
