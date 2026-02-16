#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

double calc_temperature(const std::vector<double>& prices, const std::vector<double>& volumes) {
    const std::size_t n = std::min(prices.size(), volumes.size());
    if (n < 2) {
        return 0.0;
    }

    double kinetic_energy = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
        const double dp = prices[i] - prices[i - 1];
        const double vm = volumes[i];
        const double x = dp * vm;
        kinetic_energy += x * x;
    }

    return kinetic_energy / static_cast<double>(n);
}

double calc_entropy(const std::vector<double>& volumes) {
    double total = 0.0;
    for (double v : volumes) {
        if (v > 0.0) {
            total += v;
        }
    }

    if (total <= 0.0) {
        return 0.0;
    }

    double entropy = 0.0;
    for (double v : volumes) {
        if (v <= 0.0) {
            continue;
        }

        const double p = v / total;
        entropy -= p * std::log(p);
    }
    return entropy;
}

std::string detect_phase(double temperature, double entropy) {
    if (temperature < 400.0 && entropy < 2.5) {
        return "ICE";
    }
    if (temperature >= 800.0 && entropy >= 3.5) {
        return "VAPOR";
    }
    return "WATER";
}

PYBIND11_MODULE(physics_cpp, m) {
    m.doc() = "HEAN physics compute kernels (C++, pybind11)";
    m.def("calc_temperature", &calc_temperature, py::arg("prices"), py::arg("volumes"));
    m.def("calc_entropy", &calc_entropy, py::arg("volumes"));
    m.def("detect_phase", &detect_phase, py::arg("temperature"), py::arg("entropy"));
}

