/**
 * ELM (Extreme Learning Machine) Regressor for Order Flow Imbalance (OFI) Prediction
 * 
 * Lightweight single-hidden-layer neural network for real-time orderbook toxicity detection.
 * Detects spoofing and layering patterns in <1ms inference time.
 */

#ifndef ELM_REGRESSOR_H
#define ELM_REGRESSOR_H

#include <vector>
#include <random>
#include <cmath>
#include <cstring>

/**
 * ELM Regressor for OFI (Order Flow Imbalance) prediction
 * 
 * Architecture:
 * - Input: Orderbook features (bid/ask sizes, depths, imbalances)
 * - Hidden layer: Random weights (not trained), ReLU activation
 * - Output: Predicted price movement direction and magnitude
 */
class ELM_Regressor {
public:
    /**
     * Initialize ELM with specified architecture
     * 
     * @param input_size: Number of input features (orderbook features)
     * @param hidden_size: Number of hidden neurons (typically 50-200)
     */
    ELM_Regressor(int input_size, int hidden_size = 100);
    
    ~ELM_Regressor();
    
    /**
     * Predict price movement based on orderbook features
     * 
     * @param features: Input feature vector (orderbook imbalance, depths, etc.)
     * @return: Predicted price movement (positive = upward, negative = downward)
     */
    double predict(const std::vector<double>& features) const;
    
    /**
     * Predict price movement with raw pointer (for zero-copy optimization)
     */
    double predict_raw(const double* features, int feature_size) const;
    
    /**
     * Train output weights using Moore-Penrose pseudo-inverse
     * 
     * @param X: Training feature matrix (samples x features)
     * @param y: Training targets (price movements)
     */
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<double>& y);
    
    /**
     * Train with raw pointers (for performance)
     */
    void train_raw(const double* X, const double* y, int num_samples, int feature_size);
    
    /**
     * Detect spoofing/layering based on predicted vs actual price movement
     * 
     * @param predicted_movement: Predicted price movement from ELM
     * @param actual_movement: Actual observed price movement
     * @param threshold: Spoofing detection threshold (default: 0.002 = 0.2%)
     * @return: Spoofing probability (0-1, 1 = high confidence spoofing)
     */
    double detect_spoofing(double predicted_movement, double actual_movement, 
                          double threshold = 0.002) const;
    
    /**
     * Calculate Order Flow Imbalance (OFI) from orderbook snapshots
     * 
     * OFI = sum over levels of [I(event at level) * size]
     * where I = +1 for bid, -1 for ask
     */
    static double calculate_ofi(const double* bid_sizes, const double* ask_sizes,
                                int num_levels);
    
    /**
     * Extract orderbook features for prediction
     * Features: [OFI, bid_depth, ask_depth, spread, mid_price_change, ...]
     */
    static std::vector<double> extract_features(
        const double* bid_prices, const double* bid_sizes, int num_bids,
        const double* ask_prices, const double* ask_sizes, int num_asks,
        double prev_mid_price = 0.0
    );
    
private:
    int input_size_;
    int hidden_size_;
    
    // Random weights (input -> hidden) - not trained, fixed at initialization
    std::vector<std::vector<double>> input_weights_;  // [hidden_size][input_size]
    std::vector<double> hidden_biases_;  // [hidden_size]
    
    // Trained weights (hidden -> output)
    std::vector<double> output_weights_;  // [hidden_size]
    
    // Random number generator for weight initialization
    mutable std::mt19937 rng_;
    
    // Activation function (ReLU)
    double relu(double x) const {
        return x > 0.0 ? x : 0.0;
    }
    
    // Initialize random weights
    void initialize_weights();
};

#endif // ELM_REGRESSOR_H
