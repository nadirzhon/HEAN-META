/**
 * Python bindings for GraphEngine using pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <chrono>
#include <vector>
#include <string>

// Include HFT engine headers
#include "Sniper.h"
#include "ELM_Regressor.h"
#include "ToxicityDetector.h"
#include "Scalper.h"
#include "swarm_manager.h"
#include "ofi_monitor.h"
#include "fingerprinter.h"
#include "metamorphic_engine.h"

extern "C" {
    void graph_engine_init(int window_size);
    int graph_engine_add_asset(const char* symbol);
    int graph_engine_update_price(const char* symbol, double price, int64_t timestamp_ns);
    void graph_engine_recalculate();
    void graph_engine_get_feature_vector(double* output, int size);
    const char* graph_engine_get_leader();
    double graph_engine_get_correlation(const char* symbol_a, const char* symbol_b);
    double graph_engine_get_lead_lag(const char* symbol_a, const char* symbol_b);
    int graph_engine_get_asset_count();
    void graph_engine_cleanup();
    
    // Algorithmic Fingerprinting functions
    void algo_fingerprinter_init();
    void algo_fingerprinter_update_order(const char* order_id, const char* symbol, double price, double size, int64_t timestamp_ns, int is_limit);
    void algo_fingerprinter_remove_order(const char* order_id);
    int algo_fingerprinter_get_predictive_alpha(const char* symbol, double* alpha_signal, double* confidence, char* bot_id_out, int max_bot_id_len);
    int algo_fingerprinter_get_active_orders_count();
    int algo_fingerprinter_get_identified_bots_count();
    void algo_fingerprinter_cleanup();
    
    // FastWarden TDA functions
    void fast_warden_init();
    void fast_warden_update_orderbook(
        const char* symbol,
        const double* bid_prices,
        const double* bid_sizes,
        int num_bids,
        const double* ask_prices,
        const double* ask_sizes,
        int num_asks
    );
    double fast_warden_get_market_topology_score();
    double fast_warden_predict_slippage(const char* symbol, double order_size, int is_buy);
    int fast_warden_is_market_disconnected();
    // Phase 19: Global Heartbeat functions
    void fast_warden_update_master_heartbeat(int64_t timestamp_ns);
    int fast_warden_is_master_online();
    int64_t fast_warden_get_last_master_heartbeat_ns();
    void fast_warden_set_heartbeat_timeout_ns(int64_t timeout_ns);
    void fast_warden_set_is_master_node(int is_master);
    int fast_warden_should_takeover_master();
    void fast_warden_cleanup();
    
    // Triangular Arbitrage Scanner functions
    void triangular_scanner_init(double fee_buffer, double min_profit_bps);
    void triangular_scanner_update_pair(const char* symbol, double bid_price, double ask_price,
                                        double bid_size, double ask_size, int64_t timestamp_ns);
    int triangular_scanner_scan_top_cycle(char* pair_a, char* pair_b, char* pair_c,
                                          double* profit_bps, double* max_size);
    void triangular_scanner_get_stats(int64_t* total_scans, int64_t* cycles_found,
                                     double* avg_latency_us);
    int triangular_scanner_get_active_pair_count();
    void triangular_scanner_cleanup();
    
    // Phase 16: Feed Handler shared memory bridge functions
    #ifdef ENABLE_SHARED_MEMORY
    void feed_handler_init();
    int feed_handler_push_tick(const char* symbol, double price, double bid, double ask, int64_t timestamp_ns);
    uint32_t feed_handler_get_dropped_ticks();
    uint64_t feed_handler_get_write_index();
    void feed_handler_cleanup();
    #endif
    
    // The Sniper (HFT Arbitrage Engine)
    void sniper_init();
    void sniper_set_delta_threshold(double threshold);
    int sniper_start();
    void sniper_stop();
    void sniper_update_binance_price(const char* symbol, double price, double bid, double ask, int64_t timestamp_ns);
    void sniper_update_bybit_price(const char* symbol, double price, double bid, double ask, int64_t timestamp_ns);
    void sniper_subscribe_symbol(const char* symbol);
    int64_t sniper_get_total_signals();
    int64_t sniper_get_executed_trades();
    double sniper_get_total_profit();
    int64_t sniper_get_avg_execution_time_ns();
    void sniper_cleanup();
    
    // ELM Regressor (OFI Prediction)
    void elm_init(int input_size, int hidden_size);
    double elm_predict(const double* features, int feature_size);
    void elm_train(const double* X, const double* y, int num_samples, int feature_size);
    double elm_detect_spoofing(double predicted, double actual, double threshold);
    double elm_calculate_ofi(const double* bid_sizes, const double* ask_sizes, int num_levels);
    void elm_cleanup();
    
    // Toxicity Detector
    void toxicity_detector_init();
    void toxicity_detector_update_orderbook(
        const char* symbol,
        const double* bid_prices, const double* bid_sizes, int num_bids,
        const double* ask_prices, const double* ask_sizes, int num_asks,
        int64_t timestamp_ns
    );
    int toxicity_detector_is_fake_order(const char* symbol, double price, double size, int is_bid);
    void toxicity_detector_cleanup();
    
    // Scalper (Profit-Extraction Mode)
    void scalper_init();
    void scalper_set_profit_target_pct(double target);
    void scalper_update_price(const char* symbol, double price, double bid, double ask, int64_t timestamp_ns);
    void scalper_set_hard_stop_ticks(const char* symbol, double stop_ticks);
    int scalper_execute_scalp(const char* symbol, int is_long, double entry_price, double target_price, double stop_loss_tick);
    int64_t scalper_get_total_trades();
    double scalper_get_total_profit();
    double scalper_get_win_rate();
    void scalper_cleanup();
}

#ifdef ENABLE_ONNX
#include <onnxruntime_cxx_api.h>

class VolatilityPredictor {
private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;
    bool model_loaded_;
    
public:
    VolatilityPredictor() : 
        env_(ORT_LOGGING_LEVEL_WARNING, "HEAN_VolatilityPredictor"),
        session_options_(Ort::SessionOptions::default_instance()),
        memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        model_loaded_(false) {
        
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }
    
    bool load_model(const std::string& model_path) {
        try {
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
            
            // Get input/output names and shapes
            Ort::AllocatorWithDefaultOptions allocator;
            size_t num_input_nodes = session_->GetInputCount();
            size_t num_output_nodes = session_->GetOutputCount();
            
            if (num_input_nodes > 0) {
                auto input_name = session_->GetInputNameAllocated(0, allocator);
                input_names_.push_back(input_name.get());
                
                Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
                auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
                input_shape_ = tensor_info.GetShape();
            }
            
            if (num_output_nodes > 0) {
                auto output_name = session_->GetOutputNameAllocated(0, allocator);
                output_names_.push_back(output_name.get());
            }
            
            model_loaded_ = true;
            return true;
        } catch (const std::exception& e) {
            model_loaded_ = false;
            return false;
        }
    }
    
    std::pair<bool, double> predict_volatility_spike(const std::vector<double>& feature_vector) {
        if (!model_loaded_ || session_ == nullptr) {
            return {false, 0.0};
        }
        
        try {
            // Prepare input tensor
            std::vector<int64_t> input_shape = {1, static_cast<int64_t>(feature_vector.size())};
            std::vector<float> input_data(feature_vector.begin(), feature_vector.end());
            
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info_,
                input_data.data(),
                input_data.size(),
                input_shape.data(),
                input_shape.size()
            );
            
            // Run inference
            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names_.data(),
                &input_tensor,
                1,
                output_names_.data(),
                1
            );
            
            // Extract probability
            float* float_array = output_tensors.front().GetTensorMutableData<float>();
            double probability = static_cast<double>(float_array[0]);
            
            return {true, probability};
        } catch (const std::exception& e) {
            return {false, 0.0};
        }
    }
    
    bool is_loaded() const {
        return model_loaded_;
    }
};
#endif

namespace py = pybind11;

// Wrapper class that manages a GraphEngine instance
class GraphEngineWrapper {
private:
    int window_size_;
    
public:
    GraphEngineWrapper(int window_size = 100) : window_size_(window_size) {
        graph_engine_init(window_size);
    }
    
    ~GraphEngineWrapper() {
        graph_engine_cleanup();
    }
    
    // Prevent copying
    GraphEngineWrapper(const GraphEngineWrapper&) = delete;
    GraphEngineWrapper& operator=(const GraphEngineWrapper&) = delete;
    
    int add_asset(const std::string& symbol) {
        return graph_engine_add_asset(symbol.c_str());
    }
    
    bool update_price(const std::string& symbol, double price, int64_t timestamp_ns = 0) {
        if (timestamp_ns == 0) {
            auto now = std::chrono::system_clock::now();
            timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()
            ).count();
        }
        return graph_engine_update_price(symbol.c_str(), price, timestamp_ns) != 0;
    }
    
    void recalculate() {
        graph_engine_recalculate();
    }
    
    py::array_t<double> get_feature_vector(int size = 5000) {
        auto result = py::array_t<double>(size);
        auto buffer = result.mutable_unchecked<1>();
        std::vector<double> temp(size, 0.0);
        graph_engine_get_feature_vector(temp.data(), size);
        for (int i = 0; i < size; i++) {
            buffer(i) = temp[i];
        }
        return result;
    }
    
    std::string get_current_leader() {
        const char* leader = graph_engine_get_leader();
        return leader ? std::string(leader) : std::string("");
    }
    
    double get_correlation(const std::string& symbol_a, const std::string& symbol_b) {
        return graph_engine_get_correlation(symbol_a.c_str(), symbol_b.c_str());
    }
    
    double get_lead_lag(const std::string& symbol_a, const std::string& symbol_b) {
        return graph_engine_get_lead_lag(symbol_a.c_str(), symbol_b.c_str());
    }
    
    int get_asset_count() {
        return graph_engine_get_asset_count();
    }
};

// FastWarden wrapper for TDA integration
class FastWardenWrapper {
public:
    FastWardenWrapper() {
        fast_warden_init();
    }
    
    ~FastWardenWrapper() {
        // Note: cleanup is handled at module level, not per instance
    }
    
    // Prevent copying
    FastWardenWrapper(const FastWardenWrapper&) = delete;
    FastWardenWrapper& operator=(const FastWardenWrapper&) = delete;
    
    void update_orderbook(
        const std::string& symbol,
        const std::vector<double>& bid_prices,
        const std::vector<double>& bid_sizes,
        const std::vector<double>& ask_prices,
        const std::vector<double>& ask_sizes
    ) {
        if (bid_prices.size() != bid_sizes.size() || ask_prices.size() != ask_sizes.size()) {
            throw std::runtime_error("Price and size vectors must have same length");
        }
        
        fast_warden_update_orderbook(
            symbol.c_str(),
            bid_prices.data(),
            bid_sizes.data(),
            static_cast<int>(bid_prices.size()),
            ask_prices.data(),
            ask_sizes.data(),
            static_cast<int>(ask_prices.size())
        );
    }
    
    double get_market_topology_score() {
        return fast_warden_get_market_topology_score();
    }
    
    double predict_slippage(const std::string& symbol, double order_size, bool is_buy) {
        return fast_warden_predict_slippage(symbol.c_str(), order_size, is_buy ? 1 : 0);
    }
    
    bool is_market_disconnected() {
        return fast_warden_is_market_disconnected() != 0;
    }
    
    // Phase 19: Global Heartbeat methods
    void update_master_heartbeat(int64_t timestamp_ns) {
        fast_warden_update_master_heartbeat(timestamp_ns);
    }
    
    bool is_master_online() {
        return fast_warden_is_master_online() != 0;
    }
    
    int64_t get_last_master_heartbeat_ns() {
        return fast_warden_get_last_master_heartbeat_ns();
    }
    
    void set_heartbeat_timeout_ns(int64_t timeout_ns) {
        fast_warden_set_heartbeat_timeout_ns(timeout_ns);
    }
    
    void set_is_master_node(bool is_master) {
        fast_warden_set_is_master_node(is_master ? 1 : 0);
    }
    
    bool should_takeover_master() {
        return fast_warden_should_takeover_master() != 0;
    }
};

// TriangularScanner wrapper for Python bindings
class TriangularScannerWrapper {
private:
    bool initialized_;
    
public:
    TriangularScannerWrapper(double fee_buffer = 0.001, double min_profit_bps = 5.0) 
        : initialized_(false) {
        triangular_scanner_init(fee_buffer, min_profit_bps);
        initialized_ = true;
    }
    
    ~TriangularScannerWrapper() {
        if (initialized_) {
            triangular_scanner_cleanup();
        }
    }
    
    TriangularScannerWrapper(const TriangularScannerWrapper&) = delete;
    TriangularScannerWrapper& operator=(const TriangularScannerWrapper&) = delete;
    
    void update_pair(const std::string& symbol, double bid_price, double ask_price,
                    double bid_size, double ask_size, int64_t timestamp_ns) {
        triangular_scanner_update_pair(symbol.c_str(), bid_price, ask_price,
                                      bid_size, ask_size, timestamp_ns);
    }
    
    py::dict scan_top_cycle() {
        char pair_a[32] = {0};
        char pair_b[32] = {0};
        char pair_c[32] = {0};
        double profit_bps = 0.0;
        double max_size = 0.0;
        
        int found = triangular_scanner_scan_top_cycle(pair_a, pair_b, pair_c,
                                                      &profit_bps, &max_size);
        
        if (found) {
            return py::dict(
                py::arg("found") = true,
                py::arg("pair_a") = std::string(pair_a),
                py::arg("pair_b") = std::string(pair_b),
                py::arg("pair_c") = std::string(pair_c),
                py::arg("profit_bps") = profit_bps,
                py::arg("max_size") = max_size
            );
        } else {
            return py::dict(py::arg("found") = false);
        }
    }
    
    py::dict get_stats() {
        int64_t total_scans = 0;
        int64_t cycles_found = 0;
        double avg_latency_us = 0.0;
        
        triangular_scanner_get_stats(&total_scans, &cycles_found, &avg_latency_us);
        
        return py::dict(
            py::arg("total_scans") = total_scans,
            py::arg("cycles_found") = cycles_found,
            py::arg("avg_latency_us") = avg_latency_us,
            py::arg("active_pairs") = triangular_scanner_get_active_pair_count()
        );
    }
    
    int get_active_pair_count() {
        return triangular_scanner_get_active_pair_count();
    }
};

PYBIND11_MODULE(graph_engine_py, m) {
    m.doc() = "HEAN Graph Engine - Real-time adjacency matrix and lead-lag detection + TDA";
    
    py::class_<GraphEngineWrapper>(m, "GraphEngine")
        .def(py::init<int>(), py::arg("window_size") = 100)
        .def("add_asset", &GraphEngineWrapper::add_asset, "Add or get asset index")
        .def("update_price", &GraphEngineWrapper::update_price, 
             py::arg("symbol"), py::arg("price"), py::arg("timestamp_ns") = 0,
             "Update price for an asset")
        .def("recalculate", &GraphEngineWrapper::recalculate,
             "Recalculate adjacency matrix and lead-lag relationships")
        .def("get_feature_vector", &GraphEngineWrapper::get_feature_vector,
             py::arg("size") = 5000,
             "Get high-dimensional feature vector")
        .def("get_current_leader", &GraphEngineWrapper::get_current_leader,
             "Get current market leader asset")
        .def("get_correlation", &GraphEngineWrapper::get_correlation,
             "Get correlation between two assets")
        .def("get_lead_lag", &GraphEngineWrapper::get_lead_lag,
             "Get lead-lag relationship (positive = a leads b)")
        .def("get_asset_count", &GraphEngineWrapper::get_asset_count,
             "Get number of tracked assets");
    
    // FastWarden TDA bindings
    py::class_<FastWardenWrapper>(m, "FastWarden")
        .def(py::init<>())
        .def("update_orderbook", &FastWardenWrapper::update_orderbook,
             py::arg("symbol"), py::arg("bid_prices"), py::arg("bid_sizes"),
             py::arg("ask_prices"), py::arg("ask_sizes"),
             "Update L2 orderbook for topological analysis")
        .def("get_market_topology_score", &FastWardenWrapper::get_market_topology_score,
             "Get market structural stability score (0-1, 1=stable, 0=collapsing)")
        .def("predict_slippage", &FastWardenWrapper::predict_slippage,
             py::arg("symbol"), py::arg("order_size"), py::arg("is_buy"),
             "Predict slippage using Riemannian curvature (returns percentage, e.g., 0.005 = 0.5%)")
        .def("is_market_disconnected", &FastWardenWrapper::is_market_disconnected,
             "Check if market manifold is disconnected (watchdog for flash-crash detection)")
        // Phase 19: Global Heartbeat bindings
        .def("update_master_heartbeat", &FastWardenWrapper::update_master_heartbeat,
             py::arg("timestamp_ns"),
             "Update master node heartbeat timestamp")
        .def("is_master_online", &FastWardenWrapper::is_master_online,
             "Check if master node is online (based on heartbeat)")
        .def("get_last_master_heartbeat_ns", &FastWardenWrapper::get_last_master_heartbeat_ns,
             "Get last master node heartbeat timestamp in nanoseconds")
        .def("set_heartbeat_timeout_ns", &FastWardenWrapper::set_heartbeat_timeout_ns,
             py::arg("timeout_ns"),
             "Set heartbeat timeout in nanoseconds (default: 10ms)")
        .def("set_is_master_node", &FastWardenWrapper::set_is_master_node,
             py::arg("is_master"),
             "Set whether this node is the master node")
        .def("should_takeover_master", &FastWardenWrapper::should_takeover_master,
             "Check if this node should take over master role (<10ms failover)");
    
    // Triangular Arbitrage Scanner bindings
    py::class_<TriangularScannerWrapper>(m, "TriangularScanner")
        .def(py::init<double, double>(), py::arg("fee_buffer") = 0.001, py::arg("min_profit_bps") = 5.0,
             "Initialize Triangular Arbitrage Scanner")
        .def("update_pair", &TriangularScannerWrapper::update_pair,
             py::arg("symbol"), py::arg("bid_price"), py::arg("ask_price"),
             py::arg("bid_size"), py::arg("ask_size"), py::arg("timestamp_ns"),
             "Update trading pair prices (ultra-low latency)")
        .def("scan_top_cycle", &TriangularScannerWrapper::scan_top_cycle,
             "Scan for top triangular arbitrage opportunity (< 500μs latency)")
        .def("get_stats", &TriangularScannerWrapper::get_stats,
             "Get scanner statistics (scans, cycles found, avg latency)")
        .def("get_active_pair_count", &TriangularScannerWrapper::get_active_pair_count,
             "Get number of active trading pairs being monitored");
    
#ifdef ENABLE_ONNX
    py::class_<VolatilityPredictor>(m, "VolatilityPredictor")
        .def(py::init<>())
        .def("load_model", &VolatilityPredictor::load_model,
             "Load ONNX model for volatility prediction")
        .def("predict_volatility_spike", &VolatilityPredictor::predict_volatility_spike,
             "Predict volatility spike probability (returns success, probability)")
        .def("is_loaded", &VolatilityPredictor::is_loaded,
             "Check if model is loaded");
#else
    m.attr("ENABLE_ONNX") = false;
    m.def("create_volatility_predictor", []() {
        throw std::runtime_error("ONNX Runtime not available. Rebuild with ONNX support.");
    });
#endif
    
    // The Sniper (HFT Arbitrage Engine) bindings
    m.def("sniper_init", []() { sniper_init(); }, "Initialize The Sniper arbitrage engine");
    m.def("sniper_set_delta_threshold", [](double threshold) { sniper_set_delta_threshold(threshold); },
          py::arg("threshold"), "Set price delta threshold (0.05% = 0.0005)");
    m.def("sniper_start", []() { return sniper_start() != 0; }, "Start The Sniper engine");
    m.def("sniper_stop", []() { sniper_stop(); }, "Stop The Sniper engine");
    m.def("sniper_update_binance_price", [](const std::string& symbol, double price, double bid, double ask, int64_t timestamp_ns) {
        sniper_update_binance_price(symbol.c_str(), price, bid, ask, timestamp_ns);
    }, py::arg("symbol"), py::arg("price"), py::arg("bid"), py::arg("ask"), py::arg("timestamp_ns"),
       "Update Binance price for arbitrage detection");
    m.def("sniper_update_bybit_price", [](const std::string& symbol, double price, double bid, double ask, int64_t timestamp_ns) {
        sniper_update_bybit_price(symbol.c_str(), price, bid, ask, timestamp_ns);
    }, py::arg("symbol"), py::arg("price"), py::arg("bid"), py::arg("ask"), py::arg("timestamp_ns"),
       "Update Bybit price for arbitrage detection");
    m.def("sniper_subscribe_symbol", [](const std::string& symbol) {
        sniper_subscribe_symbol(symbol.c_str());
    }, py::arg("symbol"), "Subscribe to symbol for arbitrage detection");
    m.def("sniper_get_total_signals", []() { return sniper_get_total_signals(); },
          "Get total arbitrage signals detected");
    m.def("sniper_get_executed_trades", []() { return sniper_get_executed_trades(); },
          "Get total executed trades");
    m.def("sniper_get_total_profit", []() { return sniper_get_total_profit(); },
          "Get total profit from arbitrage");
    m.def("sniper_get_avg_execution_time_ns", []() { return sniper_get_avg_execution_time_ns(); },
          "Get average execution time in nanoseconds");
    
    // ELM Regressor bindings
    m.def("elm_init", [](int input_size, int hidden_size) { elm_init(input_size, hidden_size); },
          py::arg("input_size"), py::arg("hidden_size") = 100, "Initialize ELM regressor");
    m.def("elm_predict", [](const py::array_t<double>& features) {
        auto buf = features.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Features must be 1D array");
        }
        return elm_predict(static_cast<double*>(buf.ptr), buf.size);
    }, py::arg("features"), "Predict price movement from orderbook features");
    m.def("elm_train", [](const py::array_t<double>& X, const py::array_t<double>& y) {
        auto X_buf = X.request();
        auto y_buf = y.request();
        if (X_buf.ndim != 2 || y_buf.ndim != 1) {
            throw std::runtime_error("X must be 2D, y must be 1D");
        }
        if (X_buf.shape[0] != y_buf.size) {
            throw std::runtime_error("X and y must have same number of samples");
        }
        elm_train(static_cast<double*>(X_buf.ptr), static_cast<double*>(y_buf.ptr),
                  X_buf.shape[0], X_buf.shape[1]);
    }, py::arg("X"), py::arg("y"), "Train ELM regressor");
    m.def("elm_detect_spoofing", [](double predicted, double actual, double threshold = 0.002) {
        return elm_detect_spoofing(predicted, actual, threshold);
    }, py::arg("predicted"), py::arg("actual"), py::arg("threshold") = 0.002,
       "Detect spoofing probability (0-1)");
    m.def("elm_calculate_ofi", [](const py::array_t<double>& bid_sizes, const py::array_t<double>& ask_sizes) {
        auto bid_buf = bid_sizes.request();
        auto ask_buf = ask_sizes.request();
        if (bid_buf.ndim != 1 || ask_buf.ndim != 1) {
            throw std::runtime_error("Sizes must be 1D arrays");
        }
        int num_levels = std::min(bid_buf.size, ask_buf.size);
        return elm_calculate_ofi(static_cast<double*>(bid_buf.ptr),
                                static_cast<double*>(ask_buf.ptr), num_levels);
    }, py::arg("bid_sizes"), py::arg("ask_sizes"), "Calculate Order Flow Imbalance (OFI)");
    
    // Toxicity Detector bindings
    m.def("toxicity_detector_init", []() { toxicity_detector_init(); }, "Initialize toxicity detector");
    m.def("toxicity_detector_update_orderbook", [](
        const std::string& symbol,
        const py::array_t<double>& bid_prices, const py::array_t<double>& bid_sizes,
        const py::array_t<double>& ask_prices, const py::array_t<double>& ask_sizes,
        int64_t timestamp_ns
    ) {
        auto bid_p_buf = bid_prices.request();
        auto bid_s_buf = bid_sizes.request();
        auto ask_p_buf = ask_prices.request();
        auto ask_s_buf = ask_sizes.request();
        toxicity_detector_update_orderbook(
            symbol.c_str(),
            static_cast<double*>(bid_p_buf.ptr), static_cast<double*>(bid_s_buf.ptr), bid_p_buf.size,
            static_cast<double*>(ask_p_buf.ptr), static_cast<double*>(ask_s_buf.ptr), ask_s_buf.size,
            timestamp_ns
        );
    }, py::arg("symbol"), py::arg("bid_prices"), py::arg("bid_sizes"),
       py::arg("ask_prices"), py::arg("ask_sizes"), py::arg("timestamp_ns"),
       "Update orderbook for toxicity detection");
    m.def("toxicity_detector_is_fake_order", [](const std::string& symbol, double price, double size, bool is_bid) {
        return toxicity_detector_is_fake_order(symbol.c_str(), price, size, is_bid ? 1 : 0) != 0;
    }, py::arg("symbol"), py::arg("price"), py::arg("size"), py::arg("is_bid"),
       "Check if order is fake (spoofing/layering)");
    
    // Scalper bindings
    m.def("scalper_init", []() { scalper_init(); }, "Initialize scalper (profit-extraction mode)");
    m.def("scalper_set_profit_target_pct", [](double target) { scalper_set_profit_target_pct(target); },
          py::arg("target"), "Set profit target percentage (0.02% = 0.0002)");
    m.def("scalper_update_price", [](const std::string& symbol, double price, double bid, double ask, int64_t timestamp_ns) {
        scalper_update_price(symbol.c_str(), price, bid, ask, timestamp_ns);
    }, py::arg("symbol"), py::arg("price"), py::arg("bid"), py::arg("ask"), py::arg("timestamp_ns"),
       "Update price for scalping");
    m.def("scalper_set_hard_stop_ticks", [](const std::string& symbol, double stop_ticks) {
        scalper_set_hard_stop_ticks(symbol.c_str(), stop_ticks);
    }, py::arg("symbol"), py::arg("stop_ticks"), "Set hard-stop in ticks (not percentage)");
    m.def("scalper_execute_scalp", [](const std::string& symbol, bool is_long, double entry_price,
                                     double target_price, double stop_loss_tick) {
        return scalper_execute_scalp(symbol.c_str(), is_long ? 1 : 0, entry_price, target_price, stop_loss_tick) != 0;
    }, py::arg("symbol"), py::arg("is_long"), py::arg("entry_price"),
       py::arg("target_price"), py::arg("stop_loss_tick"), "Execute scalping trade");
    m.def("scalper_get_total_trades", []() { return scalper_get_total_trades(); },
          "Get total scalping trades");
    m.def("scalper_get_total_profit", []() { return scalper_get_total_profit(); },
          "Get total profit from scalping");
    m.def("scalper_get_win_rate", []() { return scalper_get_win_rate(); },
          "Get win rate (0-1)");
    
    // Algorithmic Fingerprinting bindings
    m.def("algo_fingerprinter_init", []() {
        algo_fingerprinter_init();
    }, "Initialize Algorithmic Fingerprinting Engine");
    
    m.def("algo_fingerprinter_update_order", [](
        const std::string& order_id,
        const std::string& symbol,
        double price,
        double size,
        int64_t timestamp_ns,
        bool is_limit
    ) {
        algo_fingerprinter_update_order(
            order_id.c_str(),
            symbol.c_str(),
            price,
            size,
            timestamp_ns,
            is_limit ? 1 : 0
        );
    }, py::arg("order_id"), py::arg("symbol"), py::arg("price"), py::arg("size"), 
       py::arg("timestamp_ns"), py::arg("is_limit"),
       "Update order information for fingerprinting");
    
    m.def("algo_fingerprinter_remove_order", [](
        const std::string& order_id
    ) {
        algo_fingerprinter_remove_order(order_id.c_str());
    }, py::arg("order_id"), "Remove order from fingerprinting");
    
    m.def("algo_fingerprinter_get_predictive_alpha", [](
        const std::string& symbol
    ) -> py::dict {
        double alpha_signal = 0.0;
        double confidence = 0.0;
        char bot_id[256] = {0};
        
        int result = algo_fingerprinter_get_predictive_alpha(
            symbol.c_str(),
            &alpha_signal,
            &confidence,
            bot_id,
            sizeof(bot_id)
        );
        
        if (result == 0) {
            return py::dict("signal_available"_a=false);
        }
        
        return py::dict(
            "signal_available"_a=true,
            "alpha_signal"_a=alpha_signal,
            "confidence"_a=confidence,
            "bot_id"_a=std::string(bot_id)
        );
    }, py::arg("symbol"), "Get predictive alpha signal for symbol");
    
    m.def("algo_fingerprinter_get_active_orders_count", []() -> int {
        return algo_fingerprinter_get_active_orders_count();
    }, "Get count of active orders being fingerprinted");
    
    m.def("algo_fingerprinter_get_identified_bots_count", []() -> int {
        return algo_fingerprinter_get_identified_bots_count();
    }, "Get count of identified bot signatures");
    
    // Metamorphic Engine bindings
    m.def("metamorphic_engine_init", []() {
        metamorphic_engine_init();
    }, "Initialize Metamorphic Engine");
    
    m.def("metamorphic_engine_register_strategy", [](
        const std::string& strategy_id,
        int model_type
    ) {
        metamorphic_engine_register_strategy(strategy_id.c_str(), model_type);
    }, py::arg("strategy_id"), py::arg("model_type"),
       "Register a strategy for profiling");
    
    m.def("metamorphic_engine_record_trade", [](
        const std::string& strategy_id,
        double pnl,
        int64_t timestamp_ns,
        bool is_win
    ) {
        metamorphic_engine_record_trade(strategy_id.c_str(), pnl, timestamp_ns, is_win ? 1 : 0);
    }, py::arg("strategy_id"), py::arg("pnl"), py::arg("timestamp_ns"), py::arg("is_win"),
       "Record a trade result for alpha decay detection");
    
    m.def("metamorphic_engine_get_evolution_status", [](
        const std::string& strategy_id
    ) -> py::dict {
        int current_model = 0;
        int proposed_model = 0;
        double alpha_decay_rate = 0.0;
        int evolution_cycle = 0;
        
        int triggered = metamorphic_engine_get_evolution_status(
            strategy_id.c_str(), &current_model, &proposed_model,
            &alpha_decay_rate, &evolution_cycle);
        
        return py::dict(
            "evolution_triggered"_a=(triggered != 0),
            "current_model"_a=current_model,
            "proposed_model"_a=proposed_model,
            "alpha_decay_rate"_a=alpha_decay_rate,
            "evolution_cycle"_a=evolution_cycle
        );
    }, py::arg("strategy_id"), "Get evolution status for a strategy");
    
    m.def("metamorphic_engine_apply_evolution", [](
        const std::string& strategy_id
    ) {
        metamorphic_engine_apply_evolution(strategy_id.c_str());
    }, py::arg("strategy_id"), "Apply evolutionary model change");
    
    m.def("metamorphic_engine_get_sel", []() -> double {
        return metamorphic_engine_get_sel();
    }, "Get System Evolution Level (SEL) - overall system intelligence metric");
    
    m.def("metamorphic_engine_cleanup", []() {
        metamorphic_engine_cleanup();
    }, "Cleanup Metamorphic Engine");
    
    // Module-level cleanup function
    m.def("cleanup", []() {
        algo_fingerprinter_cleanup();
        scalper_cleanup();
        toxicity_detector_cleanup();
        elm_cleanup();
        sniper_cleanup();
        fast_warden_cleanup();
        graph_engine_cleanup();
        metamorphic_engine_cleanup();
        #ifdef ENABLE_SHARED_MEMORY
        feed_handler_cleanup();
        #endif
    }, "Cleanup all C++ engines");
    
#ifdef ENABLE_SHARED_MEMORY
    // Phase 16: Feed Handler bindings for shared memory bridge
    m.def("feed_handler_init", []() {
        feed_handler_init();
    }, "Initialize Feed Handler shared memory bridge");
    
    m.def("feed_handler_push_tick", [](
        const std::string& symbol,
        double price,
        double bid,
        double ask,
        int64_t timestamp_ns
    ) -> int {
        return feed_handler_push_tick(
            symbol.c_str(),
            price,
            bid,
            ask,
            timestamp_ns
        );
    }, py::arg("symbol"), py::arg("price"), py::arg("bid"), py::arg("ask"), py::arg("timestamp_ns"),
       "Push tick data to shared memory ring buffer (zero-copy)");
    
    m.def("feed_handler_get_dropped_ticks", []() -> uint32_t {
        return feed_handler_get_dropped_ticks();
    }, "Get number of dropped ticks (buffer full)");
    
    m.def("feed_handler_get_write_index", []() -> uint64_t {
        return feed_handler_get_write_index();
    }, "Get current write index in ring buffer");
    
    m.def("feed_handler_cleanup", []() {
        feed_handler_cleanup();
    }, "Cleanup Feed Handler shared memory");
#else
    m.attr("ENABLE_SHARED_MEMORY") = false;
    m.def("feed_handler_init", []() {
        throw std::runtime_error("Shared memory bridge not available. Rebuild with Boost.Interprocess support.");
    });
#endif
}
