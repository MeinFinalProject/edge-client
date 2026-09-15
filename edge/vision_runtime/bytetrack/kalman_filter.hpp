#pragma once

#include <array>

namespace vision_runtime::bytetrack {

class KalmanFilter {
public:
    using Mean = std::array<double, 8>;
    using Covariance = std::array<double, 64>;
    using Measurement = std::array<double, 4>;

    struct State {
        Mean mean{};
        Covariance covariance{};
    };

    [[nodiscard]] State initiate(Measurement const& measurement) const;
    [[nodiscard]] State predict(Mean const& mean, Covariance const& covariance) const;
    [[nodiscard]] State update(
        Mean const& mean,
        Covariance const& covariance,
        Measurement const& measurement
    ) const;

private:
    static constexpr double kStdWeightPosition = 1.0 / 20.0;
    static constexpr double kStdWeightVelocity = 1.0 / 160.0;
};

} // namespace vision_runtime::bytetrack
