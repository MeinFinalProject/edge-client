#include "bytetrack/kalman_filter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace vision_runtime::bytetrack {
namespace {

constexpr std::size_t kStateDim = 8;
constexpr std::size_t kMeasureDim = 4;

inline double& at(KalmanFilter::Covariance& m, std::size_t r, std::size_t c) {
    return m[r * kStateDim + c];
}
inline double at(KalmanFilter::Covariance const& m, std::size_t r, std::size_t c) {
    return m[r * kStateDim + c];
}

using Matrix4 = std::array<double, 16>;
inline double& at4(Matrix4& m, std::size_t r, std::size_t c) {
    return m[r * kMeasureDim + c];
}
inline double at4(Matrix4 const& m, std::size_t r, std::size_t c) {
    return m[r * kMeasureDim + c];
}

Matrix4 invert4(Matrix4 input) {
    std::array<double, 32> aug{};
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            aug[r * 8 + c] = at4(input, r, c);
            aug[r * 8 + 4 + c] = (r == c) ? 1.0 : 0.0;
        }
    }

    for (std::size_t col = 0; col < 4; ++col) {
        std::size_t pivot = col;
        double pivot_abs = std::abs(aug[pivot * 8 + col]);
        for (std::size_t r = col + 1; r < 4; ++r) {
            const double value = std::abs(aug[r * 8 + col]);
            if (value > pivot_abs) {
                pivot = r;
                pivot_abs = value;
            }
        }
        if (pivot_abs < 1e-12) {
            throw std::runtime_error("ByteTrack Kalman projected covariance singular");
        }
        if (pivot != col) {
            for (std::size_t c = 0; c < 8; ++c) {
                std::swap(aug[col * 8 + c], aug[pivot * 8 + c]);
            }
        }

        const double div = aug[col * 8 + col];
        for (std::size_t c = 0; c < 8; ++c) {
            aug[col * 8 + c] /= div;
        }
        for (std::size_t r = 0; r < 4; ++r) {
            if (r == col) continue;
            const double factor = aug[r * 8 + col];
            for (std::size_t c = 0; c < 8; ++c) {
                aug[r * 8 + c] -= factor * aug[col * 8 + c];
            }
        }
    }

    Matrix4 inv{};
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            at4(inv, r, c) = aug[r * 8 + 4 + c];
        }
    }
    return inv;
}

} // namespace

KalmanFilter::State KalmanFilter::initiate(Measurement const& measurement) const {
    State out{};
    for (std::size_t i = 0; i < 4; ++i) out.mean[i] = measurement[i];
    for (std::size_t i = 4; i < 8; ++i) out.mean[i] = 0.0;

    const double h = std::max(1e-6, measurement[3]);
    const std::array<double, 8> stddev{
        2.0 * kStdWeightPosition * h,
        2.0 * kStdWeightPosition * h,
        1e-2,
        2.0 * kStdWeightPosition * h,
        10.0 * kStdWeightVelocity * h,
        10.0 * kStdWeightVelocity * h,
        1e-5,
        10.0 * kStdWeightVelocity * h,
    };
    for (std::size_t i = 0; i < 8; ++i) {
        at(out.covariance, i, i) = stddev[i] * stddev[i];
    }
    return out;
}

KalmanFilter::State KalmanFilter::predict(Mean const& mean, Covariance const& covariance) const {
    State out{};
    out.mean = mean;
    for (std::size_t i = 0; i < 4; ++i) out.mean[i] += mean[i + 4];

    // F = [I I; 0 I]. Compute F * P * F^T explicitly.
    double F[8][8]{};
    for (std::size_t i = 0; i < 8; ++i) F[i][i] = 1.0;
    for (std::size_t i = 0; i < 4; ++i) F[i][i + 4] = 1.0;

    double tmp[8][8]{};
    for (std::size_t r = 0; r < 8; ++r) {
        for (std::size_t c = 0; c < 8; ++c) {
            double sum = 0.0;
            for (std::size_t k = 0; k < 8; ++k) sum += F[r][k] * at(covariance, k, c);
            tmp[r][c] = sum;
        }
    }
    for (std::size_t r = 0; r < 8; ++r) {
        for (std::size_t c = 0; c < 8; ++c) {
            double sum = 0.0;
            for (std::size_t k = 0; k < 8; ++k) sum += tmp[r][k] * F[c][k];
            at(out.covariance, r, c) = sum;
        }
    }

    const double h = std::max(1e-6, mean[3]);
    const std::array<double, 8> stddev{
        kStdWeightPosition * h,
        kStdWeightPosition * h,
        1e-2,
        kStdWeightPosition * h,
        kStdWeightVelocity * h,
        kStdWeightVelocity * h,
        1e-5,
        kStdWeightVelocity * h,
    };
    for (std::size_t i = 0; i < 8; ++i) {
        at(out.covariance, i, i) += stddev[i] * stddev[i];
    }
    return out;
}

KalmanFilter::State KalmanFilter::update(
    Mean const& mean,
    Covariance const& covariance,
    Measurement const& measurement
) const {
    std::array<double, 4> projected_mean{};
    for (std::size_t i = 0; i < 4; ++i) projected_mean[i] = mean[i];

    Matrix4 projected_cov{};
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            at4(projected_cov, r, c) = at(covariance, r, c);
        }
    }

    const double h = std::max(1e-6, mean[3]);
    const std::array<double, 4> stddev{
        kStdWeightPosition * h,
        kStdWeightPosition * h,
        1e-1,
        kStdWeightPosition * h,
    };
    for (std::size_t i = 0; i < 4; ++i) {
        at4(projected_cov, i, i) += stddev[i] * stddev[i];
    }

    const Matrix4 inv = invert4(projected_cov);

    // Cross covariance C = P * H^T = first 4 columns of P.
    double gain[8][4]{};
    for (std::size_t r = 0; r < 8; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            double sum = 0.0;
            for (std::size_t k = 0; k < 4; ++k) {
                sum += at(covariance, r, k) * at4(inv, k, c);
            }
            gain[r][c] = sum;
        }
    }

    std::array<double, 4> innovation{};
    for (std::size_t i = 0; i < 4; ++i) innovation[i] = measurement[i] - projected_mean[i];

    State out{};
    out.mean = mean;
    for (std::size_t r = 0; r < 8; ++r) {
        for (std::size_t k = 0; k < 4; ++k) out.mean[r] += gain[r][k] * innovation[k];
    }

    for (std::size_t r = 0; r < 8; ++r) {
        for (std::size_t c = 0; c < 8; ++c) {
            double correction = 0.0;
            for (std::size_t k = 0; k < 4; ++k) {
                correction += gain[r][k] * at(covariance, c, k);
            }
            at(out.covariance, r, c) = at(covariance, r, c) - correction;
        }
    }
    // Remove tiny asymmetry caused by floating-point arithmetic.
    for (std::size_t r = 0; r < 8; ++r) {
        for (std::size_t c = r + 1; c < 8; ++c) {
            const double v = 0.5 * (at(out.covariance, r, c) + at(out.covariance, c, r));
            at(out.covariance, r, c) = v;
            at(out.covariance, c, r) = v;
        }
    }
    return out;
}

} // namespace vision_runtime::bytetrack
