/**
 * MetamorphicCore.h
 * 
 * The C++ "Body" of the HEAN Absolute+ system.
 * Receives Logic Mutation signals from Python CausalBrain every 100ms
 * and dynamically adapts its execution logic.
 */

#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <mutex>
#include <chrono>
#include <thread>
#include <queue>

namespace hean {

// Logic mutation signal from Python CausalBrain
struct LogicMutation {
    uint64_t timestamp_ns;
    uint32_t mutation_id;
    double adaptation_rate;
    std::vector<double> mutation_params;  // Parameters for logic adaptation
    std::string mutation_type;            // "strategy", "risk", "execution", "correlation"
    bool is_priority;
    
    LogicMutation() : timestamp_ns(0), mutation_id(0), adaptation_rate(0.0), is_priority(false) {}
};

// Execution context that can be mutated
struct ExecutionContext {
    double risk_multiplier;
    double spread_threshold;
    double min_edge_confidence;
    double execution_speed_factor;
    std::map<std::string, double> strategy_params;
    uint64_t last_mutation_ns;
    
    ExecutionContext() : risk_multiplier(1.0), spread_threshold(0.0001),
                        min_edge_confidence(0.7), execution_speed_factor(1.0),
                        last_mutation_ns(0) {}
};

class MetamorphicCore {
public:
    MetamorphicCore();
    ~MetamorphicCore();
    
    // Initialize the core with ZeroMQ or shared memory connection
    bool initialize(const std::string& connection_type = "zmq", 
                    const std::string& endpoint = "ipc:///tmp/hean_metamorphic");
    
    // Process a logic mutation signal from Python Brain
    void processMutationSignal(const LogicMutation& mutation);
    
    // Get current execution context (read-only for external use)
    ExecutionContext getExecutionContext() const;
    
    // Apply mutated logic to a trading decision
    bool shouldExecuteTrade(const std::string& symbol, 
                           double edge_confidence,
                           double spread,
                           double risk_score) const;
    
    // Get current mutation statistics
    struct MutationStats {
        uint64_t total_mutations;
        uint64_t mutations_applied;
        double avg_adaptation_rate;
        uint64_t last_mutation_time_ns;
    };
    MutationStats getMutationStats() const;
    
    // Start the mutation signal receiver thread
    void startMutationReceiver();
    
    // Stop the mutation signal receiver
    void stopMutationReceiver();
    
    // Check if core is ready
    bool isReady() const { return is_initialized_.load(); }
    
private:
    // Internal mutation processing
    void applyMutation(const LogicMutation& mutation);
    void evolveExecutionContext(const LogicMutation& mutation);
    
    // Signal receiver thread
    void mutationReceiverThread();
    
    // Connection management
    void* zmq_context_;
    void* zmq_socket_;
    std::string connection_type_;
    std::string endpoint_;
    
    // Core state
    std::atomic<bool> is_initialized_;
    std::atomic<bool> should_run_;
    mutable std::mutex context_mutex_;
    ExecutionContext current_context_;
    
    // Mutation statistics
    mutable std::mutex stats_mutex_;
    MutationStats stats_;
    
    // Receiver thread
    std::thread receiver_thread_;
    
    // Mutation queue
    std::queue<LogicMutation> mutation_queue_;
    std::mutex queue_mutex_;
};

} // namespace hean
