/**
 * Fast Math Utilities for Warden Module
 * All functions forced inline for ultra-low latency execution
 */

#ifndef FAST_MATH_H
#define FAST_MATH_H

#include <cmath>
#include <algorithm>

namespace FastMath {
    
    // Fast Euclidean distance (2D)
    __attribute__((always_inline))
    inline double distance_2d(double x1, double y1, double x2, double y2) noexcept {
        double dx = x1 - x2;
        double dy = y1 - y2;
        return std::sqrt(dx * dx + dy * dy);
    }
    
    // Fast squared distance (avoid sqrt for comparisons)
    __attribute__((always_inline))
    inline double distance_squared_2d(double x1, double y1, double x2, double y2) noexcept {
        double dx = x1 - x2;
        double dy = y1 - y2;
        return dx * dx + dy * dy;
    }
    
    // Fast abs (template for different types)
    __attribute__((always_inline))
    inline double abs(double x) noexcept {
        return std::abs(x);
    }
    
    __attribute__((always_inline))
    inline float abs(float x) noexcept {
        return std::abs(x);
    }
    
    // Fast min/max (avoid template overhead)
    __attribute__((always_inline))
    inline double min(double a, double b) noexcept {
        return std::min(a, b);
    }
    
    __attribute__((always_inline))
    inline double max(double a, double b) noexcept {
        return std::max(a, b);
    }
    
    // Fast power (optimized for common cases)
    __attribute__((always_inline))
    inline double pow2(double x) noexcept {
        return x * x;
    }
    
    __attribute__((always_inline))
    inline double pow3(double x) noexcept {
        return x * x * x;
    }
    
    __attribute__((always_inline))
    inline double pow1_5(double x) noexcept {
        double sqrt_x = std::sqrt(x);
        return x * sqrt_x;
    }
    
    // Fast sqrt approximation using Quake's fast inverse sqrt (for very high frequency)
    // Note: Modern CPUs have fast sqrt, but this can be faster in tight loops
    __attribute__((always_inline))
    inline float fast_sqrt(float x) noexcept {
        // Use standard sqrt for accuracy (modern CPUs optimize it well)
        return std::sqrt(x);
    }
    
    // Fast curvature calculation (second derivative approximation)
    __attribute__((always_inline))
    inline double curvature_2d(
        double x0, double y0,
        double x1, double y1,
        double x2, double y2
    ) noexcept {
        double dx1 = x1 - x0;
        double dx2 = x2 - x1;
        double dy1 = y1 - y0;
        double dy2 = y2 - y1;
        
        if (abs(dx1) < 1e-10 || abs(dx2) < 1e-10) {
            return 0.0;
        }
        
        double dydx1 = dy1 / dx1;
        double dydx2 = dy2 / dx2;
        double d2ydx2 = (dydx2 - dydx1) / ((dx1 + dx2) * 0.5);
        
        double denominator = pow1_5(1.0 + dydx1 * dydx1);
        if (abs(denominator) < 1e-10) {
            return 0.0;
        }
        
        return d2ydx2 / denominator;
    }
    
    // Fast normalization (L2 norm)
    __attribute__((always_inline))
    inline double normalize_2d(double& x, double& y) noexcept {
        double norm = std::sqrt(x * x + y * y);
        if (norm > 1e-10) {
            x /= norm;
            y /= norm;
        }
        return norm;
    }
    
    // Fast clamp
    __attribute__((always_inline))
    inline double clamp(double x, double min_val, double max_val) noexcept {
        return std::max(min_val, std::min(x, max_val));
    }
    
    // Fast lerp (linear interpolation)
    __attribute__((always_inline))
    inline double lerp(double a, double b, double t) noexcept {
        return a + t * (b - a);
    }
    
    // Fast smoothstep (smooth interpolation)
    __attribute__((always_inline))
    inline double smoothstep(double edge0, double edge1, double x) noexcept {
        double t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    }
}

#endif // FAST_MATH_H