/**
 * ELM Regressor Implementation for Order Flow Imbalance Prediction
 */

#include "ELM_Regressor.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <random>
#include <cmath>

// Simple matrix operations for Moore-Penrose pseudo-inverse
namespace {
    // Matrix multiplication: C = A * B
    void mat_mult(const std::vector<std::vector<double>>& A,
                  const std::vector<std::vector<double>>& B,
                  std::vector<std::vector<double>>& C) {
        int rows_A = A.size();
        int cols_A = A[0].size();
        int cols_B = B[0].size();
        
        C.resize(rows_A);
        for (int i = 0; i < rows_A; i++) {
            C[i].resize(cols_B, 0.0);
            for (int j = 0; j < cols_B; j++) {
                for (int k = 0; k < cols_A; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
    
    // Transpose matrix
    std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& A) {
        if (A.empty()) return {};
        
        int rows = A.size();
        int cols = A[0].size();
        std::vector<std::vector<double>> AT(cols, std::vector<double>(rows));
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                AT[j][i] = A[i][j];
            }
        }
        return AT;
    }
    
    // Simplified pseudo-inverse using SVD approximation (for small matrices)
    // For production, use LAPACK or Eigen, but this works for prototyping
    std::vector<std::vector<double>> pseudo_inverse(const std::vector<std::vector<double>>& A) {
        // Simple implementation: (A^T * A)^(-1) * A^T
        // For ELM, we can use ridge regression regularization
        auto AT = transpose(A);
        int rows = A.size();
        int cols = A[0].size();
        
        // Compute A^T * A
        std::vector<std::vector<double>> ATA(cols, std::vector<double>(cols, 0.0));
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < rows; k++) {
                    ATA[i][j] += AT[i][k] * A[k][j];
                }
                // Add regularization (ridge regression)
                if (i == j) {
                    ATA[i][j] += 0.001;  // Small regularization term
                }
            }
        }
        
        // Simple inverse for 2x2 or small matrices
        // For larger, use iterative method or external library
        // Here we'll use a simplified approach
        
        // Compute (A^T * A)^(-1) * A^T
        std::vector<std::vector<double>> pinv(cols, std::vector<double>(rows, 0.0));
        
        // For simplicity, use direct computation for small matrices
        // In production, use Cholesky decomposition or SVD
        if (cols <= 10) {
            // Compute inverse using Gaussian elimination (simplified)
            // For now, use approximation
            for (int i = 0; i < cols; i++) {
                for (int j = 0; j < rows; j++) {
                    for (int k = 0; k < cols; k++) {
                        double inv_ATA = (i == k ? 1.0 : 0.0) / ATA[i][i];  // Simplified
                        pinv[i][j] += inv_ATA * AT[k][j];
                    }
                }
            }
        }
        
        return pinv;
    }
}

ELM_Regressor::ELM_Regressor(int input_size, int hidden_size)
    : input_size_(input_size)
    , hidden_size_(hidden_size)
    , rng_(std::random_device{}())
{
    input_weights_.resize(hidden_size, std::vector<double>(input_size));
    hidden_biases_.resize(hidden_size);
    output_weights_.resize(hidden_size, 0.0);
    
    initialize_weights();
}

ELM_Regressor::~ELM_Regressor() {
}

void ELM_Regressor::initialize_weights() {
    // Xavier/Glorot initialization for better convergence
    std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / (input_size_ + hidden_size_)));
    
    for (int i = 0; i < hidden_size_; i++) {
        for (int j = 0; j < input_size_; j++) {
            input_weights_[i][j] = dist(rng_);
        }
        hidden_biases_[i] = dist(rng_);
    }
}

double ELM_Regressor::predict(const std::vector<double>& features) const {
    if (features.size() != static_cast<size_t>(input_size_)) {
        return 0.0;  // Invalid input
    }
    
    return predict_raw(features.data(), features.size());
}

double ELM_Regressor::predict_raw(const double* features, int feature_size) const {
    if (feature_size != input_size_) {
        return 0.0;
    }
    
    // Forward pass: input -> hidden -> output
    std::vector<double> hidden_output(hidden_size_);
    
    for (int i = 0; i < hidden_size_; i++) {
        double sum = hidden_biases_[i];
        for (int j = 0; j < input_size_; j++) {
            sum += input_weights_[i][j] * features[j];
        }
        hidden_output[i] = relu(sum);  // ReLU activation
    }
    
    // Output layer: weighted sum
    double output = 0.0;
    for (int i = 0; i < hidden_size_; i++) {
        output += output_weights_[i] * hidden_output[i];
    }
    
    return output;
}

void ELM_Regressor::train(const std::vector<std::vector<double>>& X,
                          const std::vector<double>& y) {
    if (X.empty() || X.size() != y.size()) {
        return;
    }
    
    int num_samples = X.size();
    
    // Extract features as raw arrays
    std::vector<double> X_flat(num_samples * input_size_);
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < input_size_; j++) {
            X_flat[i * input_size_ + j] = X[i][j];
        }
    }
    
    train_raw(X_flat.data(), y.data(), num_samples, input_size_);
}

void ELM_Regressor::train_raw(const double* X, const double* y, 
                               int num_samples, int feature_size) {
    if (feature_size != input_size_ || num_samples == 0) {
        return;
    }
    
    // Compute hidden layer outputs for all samples
    std::vector<std::vector<double>> H(num_samples, std::vector<double>(hidden_size_));
    
    for (int s = 0; s < num_samples; s++) {
        for (int i = 0; i < hidden_size_; i++) {
            double sum = hidden_biases_[i];
            for (int j = 0; j < input_size_; j++) {
                sum += input_weights_[i][j] * X[s * input_size_ + j];
            }
            H[s][i] = relu(sum);
        }
    }
    
    // Solve for output weights: H * beta = y
    // beta = H^+ * y (Moore-Penrose pseudo-inverse)
    
    // Compute H^T * H for regularization
    std::vector<std::vector<double>> HTH(hidden_size_, std::vector<double>(hidden_size_, 0.0));
    for (int i = 0; i < hidden_size_; i++) {
        for (int j = 0; j < hidden_size_; j++) {
            for (int s = 0; s < num_samples; s++) {
                HTH[i][j] += H[s][i] * H[s][j];
            }
            // Ridge regularization
            if (i == j) {
                HTH[i][j] += 0.001;
            }
        }
    }
    
    // Compute H^T * y
    std::vector<double> HTy(hidden_size_, 0.0);
    for (int i = 0; i < hidden_size_; i++) {
        for (int s = 0; s < num_samples; s++) {
            HTy[i] += H[s][i] * y[s];
        }
    }
    
    // Solve HTH * beta = HTy using iterative method
    // For simplicity, use gradient descent for small hidden_size
    std::vector<double> beta(hidden_size_, 0.0);
    double learning_rate = 0.01;
    int max_iterations = 1000;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        std::vector<double> gradient(hidden_size_, 0.0);
        
        for (int i = 0; i < hidden_size_; i++) {
            double residual = -HTy[i];
            for (int j = 0; j < hidden_size_; j++) {
                residual += HTH[i][j] * beta[j];
            }
            gradient[i] = residual;
        }
        
        // Update beta
        double max_grad = 0.0;
        for (int i = 0; i < hidden_size_; i++) {
            beta[i] -= learning_rate * gradient[i];
            max_grad = std::max(max_grad, std::abs(gradient[i]));
        }
        
        // Check convergence
        if (max_grad < 1e-6) {
            break;
        }
    }
    
    output_weights_ = beta;
}

double ELM_Regressor::detect_spoofing(double predicted_movement, double actual_movement,
                                     double threshold) const {
    // Spoofing detection: if predicted movement is large but actual is small/opposite,
    // likely spoofing
    double discrepancy = std::abs(predicted_movement - actual_movement);
    
    // If large predicted movement but small actual = spoofing
    if (std::abs(predicted_movement) > threshold && std::abs(actual_movement) < threshold * 0.5) {
        return std::min(1.0, discrepancy / threshold);  // Spoofing probability
    }
    
    return 0.0;  // No spoofing detected
}

double ELM_Regressor::calculate_ofi(const double* bid_sizes, const double* ask_sizes,
                                    int num_levels) {
    double ofi = 0.0;
    
    // OFI = sum over levels of [I(event) * size]
    // I = +1 for bid events, -1 for ask events
    for (int i = 0; i < num_levels; i++) {
        // Simplified: use bid/ask size difference as proxy for order flow
        ofi += bid_sizes[i] - ask_sizes[i];
    }
    
    // Normalize by total depth
    double total_depth = 0.0;
    for (int i = 0; i < num_levels; i++) {
        total_depth += bid_sizes[i] + ask_sizes[i];
    }
    
    if (total_depth > 0.0) {
        ofi /= total_depth;
    }
    
    return ofi;
}

std::vector<double> ELM_Regressor::extract_features(
    const double* bid_prices, const double* bid_sizes, int num_bids,
    const double* ask_prices, const double* ask_sizes, int num_asks,
    double prev_mid_price) {
    
    std::vector<double> features;
    
    if (num_bids == 0 || num_asks == 0) {
        return features;
    }
    
    // Feature 1: Order Flow Imbalance (OFI)
    int num_levels = std::min(num_bids, num_asks);
    double ofi = calculate_ofi(bid_sizes, ask_sizes, num_levels);
    features.push_back(ofi);
    
    // Feature 2: Bid depth (total size on bid side)
    double bid_depth = 0.0;
    for (int i = 0; i < num_bids; i++) {
        bid_depth += bid_sizes[i];
    }
    features.push_back(bid_depth);
    
    // Feature 3: Ask depth (total size on ask side)
    double ask_depth = 0.0;
    for (int i = 0; i < num_asks; i++) {
        ask_depth += ask_sizes[i];
    }
    features.push_back(ask_depth);
    
    // Feature 4: Spread (normalized)
    double best_bid = bid_prices[0];
    double best_ask = ask_prices[0];
    double mid_price = (best_bid + best_ask) / 2.0;
    double spread = (best_ask - best_bid) / mid_price;
    features.push_back(spread);
    
    // Feature 5: Mid price change (if previous available)
    if (prev_mid_price > 0.0) {
        double mid_change = (mid_price - prev_mid_price) / prev_mid_price;
        features.push_back(mid_change);
    } else {
        features.push_back(0.0);
    }
    
    // Feature 6: Weighted average bid/ask imbalance
    double weighted_bid = 0.0, weighted_ask = 0.0;
    double total_bid_weight = 0.0, total_ask_weight = 0.0;
    
    for (int i = 0; i < num_bids && i < 5; i++) {  // Top 5 levels
        double weight = 1.0 / (i + 1.0);  // Higher weight for closer levels
        weighted_bid += bid_sizes[i] * weight;
        total_bid_weight += weight;
    }
    
    for (int i = 0; i < num_asks && i < 5; i++) {
        double weight = 1.0 / (i + 1.0);
        weighted_ask += ask_sizes[i] * weight;
        total_ask_weight += weight;
    }
    
    double weighted_imbalance = 0.0;
    if (total_bid_weight > 0.0 && total_ask_weight > 0.0) {
        weighted_imbalance = (weighted_bid / total_bid_weight - weighted_ask / total_ask_weight) /
                            (weighted_bid / total_bid_weight + weighted_ask / total_ask_weight);
    }
    features.push_back(weighted_imbalance);
    
    return features;
}

// C interface for Python bindings
extern "C" {
    static ELM_Regressor* g_elm = nullptr;
    
    void elm_init(int input_size, int hidden_size) {
        if (g_elm == nullptr) {
            g_elm = new ELM_Regressor(input_size, hidden_size);
        }
    }
    
    double elm_predict(const double* features, int feature_size) {
        if (g_elm) {
            return g_elm->predict_raw(features, feature_size);
        }
        return 0.0;
    }
    
    void elm_train(const double* X, const double* y, int num_samples, int feature_size) {
        if (g_elm) {
            g_elm->train_raw(X, y, num_samples, feature_size);
        }
    }
    
    double elm_detect_spoofing(double predicted, double actual, double threshold) {
        if (g_elm) {
            return g_elm->detect_spoofing(predicted, actual, threshold);
        }
        return 0.0;
    }
    
    double elm_calculate_ofi(const double* bid_sizes, const double* ask_sizes, int num_levels) {
        return ELM_Regressor::calculate_ofi(bid_sizes, ask_sizes, num_levels);
    }
    
    void elm_cleanup() {
        if (g_elm) {
            delete g_elm;
            g_elm = nullptr;
        }
    }
}
