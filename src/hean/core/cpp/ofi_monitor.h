/**
 * Order-Flow Imbalance (OFI) Monitor with ML Prediction
 * 
 * Features:
 * - Real-time OFI calculation at each price level
 * - ML-based prediction of next 3 ticks with >75% accuracy target
 * - Lightweight LSTM/XGBoost model optimized for C++
 * - Integration with swarm intelligence system
 */

#ifndef OFI_MONITOR_H
#define OFI_MONITOR_H

#include <vector>
#include <string>
#include <map>
#include <deque>
#include <mutex>
#include <atomic>
#include <memory>

// Price level order flow data
struct PriceLevelFlow {
    double price;
    double bid_size;
    double ask_size;
    double bid_orders;      // Number of bid orders
    double ask_orders;      // Number of ask orders
    double bid_canceled;    // Canceled bid volume
    double ask_canceled;    // Canceled ask volume
    int64_t timestamp_ns;
    
    PriceLevelFlow() : price(0.0), bid_size(0.0), ask_size(0.0),
                      bid_orders(0.0), ask_orders(0.0), bid_canceled(0.0),
                      ask_canceled(0.0), timestamp_ns(0) {}
};

// OFI calculation result
struct OFIResult {
    double ofi_value;               // Net order flow imbalance (-1.0 to 1.0)
    double delta;                   // Net buy volume - sell volume
    double buy_pressure;            // Normalized buying pressure (0.0 to 1.0)
    double sell_pressure;           // Normalized selling pressure (0.0 to 1.0)
    double imbalance_strength;      // Strength of imbalance (0.0 to 1.0)
    std::vector<double> price_level_ofi;  // OFI at each price level
    
    OFIResult() : ofi_value(0.0), delta(0.0), buy_pressure(0.0),
                 sell_pressure(0.0), imbalance_strength(0.0) {}
};

// Price movement prediction (next 3 ticks)
struct PricePrediction {
    std::vector<double> predicted_prices;    // Predicted prices for next 3 ticks
    std::vector<double> probabilities;       // Confidence for each prediction
    double overall_confidence;               // Overall prediction confidence
    bool is_bullish;                         // True if predicted upward movement
    double expected_movement;                // Expected price change
    double accuracy_estimate;                // Estimated accuracy (target >75%)
    
    PricePrediction() : overall_confidence(0.0), is_bullish(false),
                       expected_movement(0.0), accuracy_estimate(0.0) {
        predicted_prices.resize(3, 0.0);
        probabilities.resize(3, 0.0);
    }
};

// Lightweight ML model interface (can be replaced with ONNX model)
class LightweightPredictor {
private:
    // Simple feedforward model weights (can be replaced with ONNX)
    std::vector<std::vector<double>> weights_hidden1_;
    std::vector<double> bias_hidden1_;
    std::vector<std::vector<double>> weights_output_;
    std::vector<double> bias_output_;
    bool model_loaded_;
    
    // Feature normalization
    std::vector<double> feature_mean_;
    std::vector<double> feature_std_;
    
    // Initialize with default weights (can load from file or ONNX)
    void initialize_default_model();
    
public:
    LightweightPredictor();
    ~LightweightPredictor();
    
    // Load model from ONNX file (if available)
    bool load_onnx_model(const std::string& model_path);
    
    // Load model from weights file
    bool load_weights(const std::string& weights_path);
    
    // Predict next 3 ticks given feature vector
    PricePrediction predict(const std::vector<double>& features);
    
    // Get feature vector size expected by model
    int get_feature_size() const { return 20; }  // Default: 20 features
};

// OFI Monitor class
class OFIMonitor {
private:
    std::map<std::string, std::deque<PriceLevelFlow>> price_level_history_;
    std::map<std::string, OFIResult> current_ofi_;
    std::mutex data_mutex_;
    std::unique_ptr<LightweightPredictor> predictor_;
    
    // Configuration
    int lookback_window_;           // Number of price levels to analyze
    double price_level_size_;       // Price increment for level calculation
    bool use_ml_prediction_;        // Enable ML prediction
    
    // OFI calculation methods
    OFIResult calculate_ofi(const std::string& symbol, const std::vector<PriceLevelFlow>& levels);
    double calculate_delta(const std::vector<PriceLevelFlow>& levels);
    double calculate_vpin(const std::vector<PriceLevelFlow>& levels);
    
    // Extract features for ML model
    std::vector<double> extract_features(const std::string& symbol, double current_price);
    
    // Update price level data
    void update_price_levels(const std::string& symbol, const PriceLevelFlow& flow);
    
public:
    OFIMonitor(int lookback_window = 20, double price_level_size = 0.01, bool use_ml = true);
    ~OFIMonitor();
    
    // Update orderbook data
    void update_orderbook(
        const std::string& symbol,
        const std::vector<std::pair<double, double>>& bids,  // (price, size)
        const std::vector<std::pair<double, double>>& asks,  // (price, size)
        int64_t timestamp_ns
    );
    
    // Update trade data (for delta calculation)
    void update_trade(
        const std::string& symbol,
        double price,
        double size,
        bool is_buy,  // True if buy, false if sell
        int64_t timestamp_ns
    );
    
    // Get current OFI for symbol
    OFIResult get_ofi(const std::string& symbol);
    
    // Predict next 3 ticks
    PricePrediction predict_next_ticks(const std::string& symbol, double current_price);
    
    // Get OFI at specific price level
    double get_price_level_ofi(const std::string& symbol, double price);
    
    // Get delta (net buy - sell volume)
    double get_delta(const std::string& symbol);
    
    // Reset data for symbol
    void reset_symbol(const std::string& symbol);
    
    // Enable/disable ML prediction
    void set_ml_prediction(bool enabled) { use_ml_prediction_ = enabled; }
    
    // Load ML model
    bool load_model(const std::string& model_path);
};

#endif // OFI_MONITOR_H
