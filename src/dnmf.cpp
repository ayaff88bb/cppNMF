#include "cppnmf/dnmf.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cppnmf {
namespace {

void validate_nonnegative_finite(const Eigen::Ref<const Eigen::MatrixXd>& input) {
    if (input.size() == 0) {
        throw std::invalid_argument("DNMF input matrix must not be empty");
    }
    if (!input.allFinite()) {
        throw std::invalid_argument("DNMF input matrix must be finite");
    }
    if (input.minCoeff() < 0.0) {
        throw std::invalid_argument("DNMF input matrix must be nonnegative");
    }
}

void multiplicative_update(
    Eigen::MatrixXd& factor,
    const Eigen::Ref<const Eigen::MatrixXd>& numerator,
    const Eigen::Ref<const Eigen::MatrixXd>& denominator,
    const double epsilon) {
    factor.array() *= numerator.array() / denominator.array().max(epsilon);
    factor = factor.cwiseMax(0.0);
}

double nsnmf_cost(
    const Eigen::Ref<const Eigen::MatrixXd>& input,
    const Eigen::Ref<const Eigen::MatrixXd>& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& activation,
    const Eigen::Ref<const Eigen::MatrixXd>& smoothing) {
    return (input - basis * smoothing * activation).norm();
}

struct NsnmfResult {
    Eigen::MatrixXd basis;
    Eigen::MatrixXd activation;
    Eigen::MatrixXd smoothing;
};

NsnmfResult train_single_layer(
    const Eigen::Ref<const Eigen::MatrixXd>& input,
    const std::size_t rank,
    const double theta,
    const DnmfOptions& options) {
    auto initialization = nndsvd(input, rank);
    NsnmfResult result;
    result.basis = std::move(initialization.basis);
    result.activation = std::move(initialization.activation);
    result.smoothing =
        (1.0 - theta) * Eigen::MatrixXd::Identity(rank, rank) +
        (theta / static_cast<double>(rank)) * Eigen::MatrixXd::Ones(rank, rank);

    std::vector<double> history{nsnmf_cost(
        input, result.basis, result.activation, result.smoothing)};
    for (std::size_t iteration = 1;
         iteration <= options.pretrain_max_iterations;
         ++iteration) {
        const Eigen::MatrixXd smoothed_activation =
            result.smoothing * result.activation;
        const Eigen::MatrixXd basis_numerator =
            input * smoothed_activation.transpose();
        const Eigen::MatrixXd basis_gram =
            smoothed_activation * smoothed_activation.transpose();
        for (std::size_t update = 0; update < options.inner_updates; ++update) {
            const Eigen::MatrixXd denominator = result.basis * basis_gram;
            multiplicative_update(
                result.basis, basis_numerator, denominator, options.epsilon);
        }

        const Eigen::MatrixXd effective_basis = result.basis * result.smoothing;
        const Eigen::MatrixXd activation_numerator =
            effective_basis.transpose() * input;
        const Eigen::MatrixXd basis_effective_gram =
            effective_basis.transpose() * effective_basis;
        for (std::size_t update = 0; update < options.inner_updates; ++update) {
            const Eigen::MatrixXd denominator =
                basis_effective_gram * result.activation;
            multiplicative_update(
                result.activation,
                activation_numerator,
                denominator,
                options.epsilon);
        }

        if (iteration % 5 == 0) {
            history.push_back(nsnmf_cost(
                input, result.basis, result.activation, result.smoothing));
        }
        if (history.size() >= 20) {
            const double previous = history[history.size() - 2];
            const double current = history.back();
            if (previous >= current &&
                previous - current <= options.tolerance * std::max(1.0, previous)) {
                break;
            }
        }
    }
    return result;
}

Eigen::MatrixXd left_chain(
    const std::vector<Eigen::MatrixXd>& basis,
    const std::vector<Eigen::MatrixXd>& smoothing,
    const std::size_t exclusive_end) {
    Eigen::MatrixXd result = basis[0] * smoothing[0];
    for (std::size_t layer = 1; layer < exclusive_end; ++layer) {
        result = result * basis[layer] * smoothing[layer];
    }
    return result;
}

Eigen::MatrixXd right_chain(
    const DnmfResult& result,
    const std::size_t layer) {
    Eigen::MatrixXd right = result.smoothing[layer];
    for (std::size_t next = layer + 1; next < result.basis.size(); ++next) {
        right = right * result.basis[next] * result.smoothing[next];
    }
    return right * result.activation.back();
}

struct RepetitionResult {
    double value = 0.0;
    Eigen::MatrixXd gradient;
};

RepetitionResult repetition_value_and_gradient(
    const Eigen::Ref<const Eigen::MatrixXd>& activation,
    const DnmfOptions& options,
    const bool compute_gradient) {
    RepetitionResult result;
    result.gradient = compute_gradient
                          ? Eigen::MatrixXd::Zero(activation.rows(), activation.cols())
                          : Eigen::MatrixXd{};
    const std::size_t sample_count = static_cast<std::size_t>(activation.cols());
    if (sample_count < 2 || options.harmonic_count == 0) {
        return result;
    }

    std::vector<std::size_t> lags;
    for (std::size_t lag = std::max<std::size_t>(1, options.repetition_lag_min);
         lag <= options.repetition_lag_max && lag < sample_count;
         ++lag) {
        if (options.harmonic_count * lag < sample_count) {
            lags.push_back(lag);
        }
    }
    if (lags.empty()) {
        return result;
    }

    std::vector<double> harmonic_weights(options.harmonic_count);
    for (std::size_t harmonic = 0; harmonic < options.harmonic_count; ++harmonic) {
        harmonic_weights[harmonic] = 1.0 / static_cast<double>(harmonic + 1);
    }
    const double weight_sum = std::accumulate(
        harmonic_weights.begin(), harmonic_weights.end(), 0.0);
    for (double& weight : harmonic_weights) {
        weight /= weight_sum;
    }

    const double temperature = std::max(options.repetition_temperature, options.epsilon);
    const double prior = 1.0 / static_cast<double>(lags.size());
    for (Eigen::Index component = 0; component < activation.rows(); ++component) {
        std::vector<double> correlations(lags.size(), 0.0);
        std::vector<Eigen::RowVectorXd> gradients;
        if (compute_gradient) {
            gradients.reserve(lags.size());
        }

        for (std::size_t lag_index = 0; lag_index < lags.size(); ++lag_index) {
            const std::size_t lag = lags[lag_index];
            std::vector<double> harmonic_values(options.harmonic_count, 0.0);
            std::vector<Eigen::RowVectorXd> harmonic_gradients;
            if (compute_gradient) {
                harmonic_gradients.assign(
                    options.harmonic_count,
                    Eigen::RowVectorXd::Zero(activation.cols()));
            }

            for (std::size_t harmonic = 0;
                 harmonic < options.harmonic_count;
                 ++harmonic) {
                const std::size_t delay = (harmonic + 1) * lag;
                const std::size_t pair_count = sample_count - delay;
                double value = 0.0;
                for (std::size_t index = 0; index < pair_count; ++index) {
                    const double first = activation(
                        component, static_cast<Eigen::Index>(index));
                    const double second = activation(
                        component, static_cast<Eigen::Index>(index + delay));
                    value += first * second;
                    if (compute_gradient) {
                        const double scale = 1.0 / static_cast<double>(pair_count);
                        harmonic_gradients[harmonic][static_cast<Eigen::Index>(index)] +=
                            scale * second;
                        harmonic_gradients[harmonic][static_cast<Eigen::Index>(index + delay)] +=
                            scale * first;
                    }
                }
                harmonic_values[harmonic] = value / static_cast<double>(pair_count);
            }

            double logarithm = 0.0;
            for (std::size_t harmonic = 0;
                 harmonic < options.harmonic_count;
                 ++harmonic) {
                logarithm += harmonic_weights[harmonic] *
                             std::log(options.epsilon + harmonic_values[harmonic]);
            }
            const double correlation = std::exp(logarithm);
            correlations[lag_index] = correlation;

            if (compute_gradient) {
                Eigen::RowVectorXd gradient =
                    Eigen::RowVectorXd::Zero(activation.cols());
                for (std::size_t harmonic = 0;
                     harmonic < options.harmonic_count;
                     ++harmonic) {
                    gradient +=
                        correlation * harmonic_weights[harmonic] /
                        (options.epsilon + harmonic_values[harmonic]) *
                        harmonic_gradients[harmonic];
                }
                gradients.push_back(std::move(gradient));
            }
        }

        std::vector<double> logits(lags.size());
        double largest_logit = -std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < lags.size(); ++index) {
            logits[index] = std::log(prior + options.epsilon) +
                            correlations[index] / temperature;
            largest_logit = std::max(largest_logit, logits[index]);
        }
        double probability_sum = 0.0;
        for (double& logit : logits) {
            logit = std::exp(logit - largest_logit);
            probability_sum += logit;
        }
        for (double& probability : logits) {
            probability /= probability_sum + options.epsilon;
        }

        double component_value = 0.0;
        for (std::size_t index = 0; index < lags.size(); ++index) {
            component_value += logits[index] * correlations[index];
        }
        result.value += component_value;

        if (compute_gradient) {
            for (std::size_t index = 0; index < lags.size(); ++index) {
                const double coefficient = logits[index] *
                    (1.0 + (correlations[index] - component_value) / temperature);
                result.gradient.row(component) += coefficient * gradients[index];
            }
        }
    }
    return result;
}

double total_objective(
    const Eigen::Ref<const Eigen::MatrixXd>& input,
    const DnmfResult& result,
    const DnmfOptions& options) {
    const double reconstruction_error =
        (input - reconstruct_dnmf(result)).squaredNorm();
    const double repetition = options.lambda_repetition > 0.0
                                  ? repetition_value_and_gradient(
                                        result.activation.back(), options, false)
                                        .value
                                  : 0.0;
    return reconstruction_error - options.lambda_repetition * repetition;
}

}  // namespace

NndsvdResult nndsvd(
    const Eigen::Ref<const Eigen::MatrixXd>& input,
    const std::size_t rank) {
    validate_nonnegative_finite(input);
    if (rank == 0 || rank > static_cast<std::size_t>(std::min(input.rows(), input.cols()))) {
        throw std::invalid_argument("NNDSVD rank is outside the matrix dimensions");
    }

    Eigen::BDCSVD<Eigen::MatrixXd> decomposition(
        input, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (decomposition.info() != Eigen::Success) {
        throw std::runtime_error("NNDSVD singular-value decomposition failed");
    }

    NndsvdResult result;
    result.basis = Eigen::MatrixXd::Zero(input.rows(), static_cast<Eigen::Index>(rank));
    result.activation = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(rank), input.cols());
    const auto& singular_values = decomposition.singularValues();
    const auto& left = decomposition.matrixU();
    const auto& right = decomposition.matrixV();

    result.basis.col(0) =
        std::sqrt(singular_values[0]) * left.col(0).cwiseAbs();
    result.activation.row(0) =
        std::sqrt(singular_values[0]) * right.col(0).cwiseAbs().transpose();

    for (std::size_t component = 1; component < rank; ++component) {
        const Eigen::VectorXd u = left.col(static_cast<Eigen::Index>(component));
        const Eigen::VectorXd v = right.col(static_cast<Eigen::Index>(component));
        const Eigen::VectorXd u_positive = u.cwiseMax(0.0);
        const Eigen::VectorXd u_negative = (-u).cwiseMax(0.0);
        const Eigen::VectorXd v_positive = v.cwiseMax(0.0);
        const Eigen::VectorXd v_negative = (-v).cwiseMax(0.0);
        const double positive_term = u_positive.norm() * v_positive.norm();
        const double negative_term = u_negative.norm() * v_negative.norm();

        const Eigen::VectorXd* chosen_u = &u_positive;
        const Eigen::VectorXd* chosen_v = &v_positive;
        double chosen_term = positive_term;
        if (negative_term > positive_term) {
            chosen_u = &u_negative;
            chosen_v = &v_negative;
            chosen_term = negative_term;
        }
        if (chosen_term > 0.0 && chosen_u->norm() > 0.0 && chosen_v->norm() > 0.0) {
            const double scale = std::sqrt(
                singular_values[static_cast<Eigen::Index>(component)] * chosen_term);
            result.basis.col(static_cast<Eigen::Index>(component)) =
                scale * *chosen_u / chosen_u->norm();
            result.activation.row(static_cast<Eigen::Index>(component)) =
                scale * chosen_v->transpose() / chosen_v->norm();
        }
    }

    result.basis = (result.basis.array() < 1e-10)
                       .select(0.1, result.basis.array())
                       .matrix();
    result.activation = (result.activation.array() < 1e-10)
                            .select(0.1, result.activation.array())
                            .matrix();
    return result;
}

ThetaSchedule configure_theta(
    const double q_nsd,
    const std::vector<double>& manual_theta) {
    if (manual_theta.empty()) {
        throw std::invalid_argument("theta schedule must not be empty");
    }
    if (!std::isfinite(q_nsd)) {
        throw std::invalid_argument("q_NSD must be finite");
    }
    ThetaSchedule result;
    constexpr double q_min = 0.381664;
    constexpr double q_max = 0.863117;
    result.normalized_difficulty = std::clamp(
        (q_nsd - q_min) / (q_max - q_min + std::numeric_limits<double>::epsilon()),
        0.0,
        1.0);
    result.adaptive.resize(manual_theta.size());
    for (std::size_t layer = 0; layer < manual_theta.size(); ++layer) {
        const double depth = manual_theta.size() <= 1
                                 ? 0.0
                                 : static_cast<double>(layer) /
                                       static_cast<double>(manual_theta.size() - 1);
        const double middle = 4.0 * depth * (1.0 - depth);
        const double eta = -2.378484 + 3.330215 * depth +
                           0.244229 * result.normalized_difficulty +
                           0.303702 * result.normalized_difficulty * middle;
        result.adaptive[layer] = std::clamp(
            1.0 / (1.0 + std::exp(-eta)), 1e-4, 1.0 - 1e-4);
    }
    result.selected = manual_theta;
    return result;
}

DnmfResult deep_nsnmf(
    const Eigen::Ref<const Eigen::MatrixXd>& input,
    const DnmfOptions& options) {
    validate_nonnegative_finite(input);
    if (options.ranks.empty() || options.theta.size() != options.ranks.size()) {
        throw std::invalid_argument("ranks and theta must have the same nonzero length");
    }
    if (options.inner_updates == 0 || options.epsilon <= 0.0) {
        throw std::invalid_argument("DNMF iteration settings must be positive");
    }

    DnmfResult result;
    result.basis.reserve(options.ranks.size());
    result.activation.reserve(options.ranks.size());
    result.smoothing.reserve(options.ranks.size());

    Eigen::MatrixXd layer_input = input;
    for (std::size_t layer = 0; layer < options.ranks.size(); ++layer) {
        if (options.ranks[layer] == 0 ||
            options.ranks[layer] >
                static_cast<std::size_t>(std::min(layer_input.rows(), layer_input.cols()))) {
            throw std::invalid_argument("a DNMF rank is outside its layer dimensions");
        }
        if (options.theta[layer] < 0.0 || options.theta[layer] >= 1.0) {
            throw std::invalid_argument("theta values must be in [0, 1)");
        }
        auto trained = train_single_layer(
            layer_input, options.ranks[layer], options.theta[layer], options);
        result.basis.push_back(std::move(trained.basis));
        result.activation.push_back(std::move(trained.activation));
        result.smoothing.push_back(std::move(trained.smoothing));
        layer_input = result.activation.back();
    }

    result.objective_history.push_back(total_objective(input, result, options));
    for (std::size_t iteration = 1;
         iteration <= options.finetune_max_iterations;
         ++iteration) {
        for (std::size_t layer = 0; layer < result.basis.size(); ++layer) {
            const Eigen::MatrixXd right = right_chain(result, layer);
            Eigen::MatrixXd numerator;
            Eigen::MatrixXd denominator_left;
            if (layer == 0) {
                numerator = input * right.transpose();
                denominator_left = Eigen::MatrixXd::Identity(
                    result.basis[layer].rows(), result.basis[layer].rows());
            } else {
                const Eigen::MatrixXd left =
                    left_chain(result.basis, result.smoothing, layer);
                numerator = left.transpose() * input * right.transpose();
                denominator_left = left.transpose() * left;
            }
            const Eigen::MatrixXd right_gram = right * right.transpose();
            for (std::size_t update = 0; update < options.inner_updates; ++update) {
                Eigen::MatrixXd denominator;
                if (layer == 0) {
                    denominator = result.basis[layer] * right_gram;
                } else {
                    denominator =
                        denominator_left * result.basis[layer] * right_gram;
                }
                multiplicative_update(
                    result.basis[layer], numerator, denominator, options.epsilon);
            }

            if (layer + 1 == result.basis.size()) {
                const Eigen::MatrixXd complete_basis =
                    left_chain(result.basis, result.smoothing, result.basis.size());
                const Eigen::MatrixXd activation_numerator =
                    complete_basis.transpose() * input;
                const Eigen::MatrixXd complete_gram =
                    complete_basis.transpose() * complete_basis;
                for (std::size_t update = 0;
                     update < options.inner_updates;
                     ++update) {
                    const Eigen::MatrixXd denominator =
                        complete_gram * result.activation.back();
                    multiplicative_update(
                        result.activation.back(),
                        activation_numerator,
                        denominator,
                        options.epsilon);
                }

                if (options.lambda_repetition > 0.0) {
                    const auto repetition = repetition_value_and_gradient(
                        result.activation.back(), options, true);
                    const double gradient_norm = repetition.gradient.norm();
                    const double activation_norm = result.activation.back().norm();
                    if (gradient_norm > options.epsilon &&
                        activation_norm > options.epsilon) {
                        const double step = options.lambda_repetition * activation_norm /
                                            (gradient_norm + options.epsilon);
                        result.activation.back() =
                            (result.activation.back() + step * repetition.gradient)
                                .cwiseMax(0.0);
                    }
                }
            }
        }

        if (iteration == 1 || iteration % 5 == 0) {
            result.objective_history.push_back(total_objective(input, result, options));
            if (result.objective_history.size() >= 20) {
                const double previous =
                    result.objective_history[result.objective_history.size() - 2];
                const double current = result.objective_history.back();
                if (previous >= current &&
                    previous - current <=
                        options.tolerance * std::max(1.0, std::abs(previous))) {
                    break;
                }
            }
        }
    }

    for (std::size_t layer = result.basis.size(); layer-- > 1;) {
        result.activation[layer - 1] = result.basis[layer] *
                                       result.smoothing[layer] *
                                       result.activation[layer];
    }
    return result;
}

Eigen::MatrixXd reconstruct_dnmf(const DnmfResult& result) {
    if (result.basis.empty() ||
        result.basis.size() != result.activation.size() ||
        result.basis.size() != result.smoothing.size()) {
        throw std::invalid_argument("DNMF factors are incomplete");
    }
    Eigen::MatrixXd reconstruction = result.activation.back();
    for (std::size_t layer = result.basis.size(); layer-- > 0;) {
        reconstruction =
            result.basis[layer] * result.smoothing[layer] * reconstruction;
    }
    return reconstruction;
}

std::vector<Eigen::MatrixXd> cumulative_bases(const DnmfResult& result) {
    if (result.basis.empty() || result.basis.size() != result.smoothing.size()) {
        throw std::invalid_argument("DNMF factors are incomplete");
    }
    std::vector<Eigen::MatrixXd> bases;
    bases.reserve(result.basis.size());
    // Keep MATLAB build_layer_reconstructions.m semantics: S participates in
    // optimization, while the interpretable/exported basis is the Z chain.
    Eigen::MatrixXd cumulative = result.basis[0];
    bases.push_back(cumulative);
    for (std::size_t layer = 1; layer < result.basis.size(); ++layer) {
        cumulative = cumulative * result.basis[layer];
        bases.push_back(cumulative);
    }
    return bases;
}

std::vector<Eigen::MatrixXd> layer_reconstructions(const DnmfResult& result) {
    const auto bases = cumulative_bases(result);
    if (bases.size() != result.activation.size()) {
        throw std::invalid_argument("DNMF factors are incomplete");
    }
    std::vector<Eigen::MatrixXd> reconstructions;
    reconstructions.reserve(bases.size());
    for (std::size_t layer = 0; layer < bases.size(); ++layer) {
        reconstructions.push_back(bases[layer] * result.activation[layer]);
    }
    return reconstructions;
}

double relative_frobenius_error(
    const Eigen::Ref<const Eigen::MatrixXd>& expected,
    const Eigen::Ref<const Eigen::MatrixXd>& actual) {
    if (expected.rows() != actual.rows() || expected.cols() != actual.cols()) {
        throw std::invalid_argument("matrices must have identical dimensions");
    }
    const double denominator = expected.norm();
    return denominator == 0.0 ? actual.norm() : (expected - actual).norm() / denominator;
}

}  // namespace cppnmf
