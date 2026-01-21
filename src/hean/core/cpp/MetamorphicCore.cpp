/**
 * MetamorphicCore.cpp
 * 
 * Implementation of the metamorphic execution engine that adapts in real-time.
 */

#include "MetamorphicCore.h"
#include <zmq.h>
#include <cstring>
#include <iostream>
#include <cmath>
#include <sstream>

#ifdef __APPLE__
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace hean {

MetamorphicCore::MetamorphicCore() 
    : zmq_context_(nullptr), zmq_socket_(nullptr),
      is_initialized_(false), should_run_(false) {
    stats_.total_mutations = 0;
    stats_.mutations_applied = 0;
    stats_.avg_adaptation_rate = 0.0;
    stats_.last_mutation_time_ns = 0;
}

MetamorphicCore::~MetamorphicCore() {
    stopMutationReceiver();
    
    if (zmq_socket_) {
        zmq_close(zmq_socket_);
        zmq_socket_ = nullptr;
    }
    
    if (zmq_context_) {
        zmq_ctx_destroy(zmq_context_);
        zmq_context_ = nullptr;
    }
}

bool MetamorphicCore::initialize(const std::string& connection_type,
                                 const std::string& endpoint) {
    connection_type_ = connection_type;
    endpoint_ = endpoint;
    
    if (connection_type == "zmq") {
        zmq_context_ = zmq_ctx_new();
        if (!zmq_context_) {
            std::cerr << "MetamorphicCore: Failed to create ZMQ context" << std::endl;
            return false;
        }
        
        zmq_socket_ = zmq_socket(zmq_context_, ZMQ_SUB);
        if (!zmq_socket_) {
            std::cerr << "MetamorphicCore: Failed to create ZMQ socket" << std::endl;
            return false;
        }
        
        // Subscribe to all messages
        zmq_setsockopt(zmq_socket_, ZMQ_SUBSCRIBE, "", 0);
        
        int rc = zmq_connect(zmq_socket_, endpoint.c_str());
        if (rc != 0) {
            std::cerr << "MetamorphicCore: Failed to connect to " << endpoint << std::endl;
            return false;
        }
        
        // Set receive timeout to 100ms for non-blocking check
        int timeout_ms = 100;
        zmq_setsockopt(zmq_socket_, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
    }
    
    is_initialized_.store(true);
    return true;
}

void MetamorphicCore::processMutationSignal(const LogicMutation& mutation) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        mutation_queue_.push(mutation);
    }
    
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_mutations++;
    }
    
    // Process immediately if queue is small
    if (mutation_queue_.size() < 10) {
        applyMutation(mutation);
    }
}

void MetamorphicCore::applyMutation(const LogicMutation& mutation) {
    std::lock_guard<std::mutex> lock(context_mutex_);
    
    auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    
    evolveExecutionContext(mutation);
    
    current_context_.last_mutation_ns = now;
    
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.mutations_applied++;
        stats_.last_mutation_time_ns = now;
        
        // Update average adaptation rate (exponential moving average)
        double alpha = 0.1;
        stats_.avg_adaptation_rate = alpha * mutation.adaptation_rate + 
                                     (1.0 - alpha) * stats_.avg_adaptation_rate;
    }
}

void MetamorphicCore::evolveExecutionContext(const LogicMutation& mutation) {
    // Apply mutation based on type
    if (mutation.mutation_type == "strategy") {
        if (mutation.mutation_params.size() >= 2) {
            current_context_.min_edge_confidence = mutation.mutation_params[0];
            current_context_.execution_speed_factor = mutation.mutation_params[1];
        }
    }
    else if (mutation.mutation_type == "risk") {
        if (mutation.mutation_params.size() >= 1) {
            current_context_.risk_multiplier = mutation.mutation_params[0];
        }
    }
    else if (mutation.mutation_type == "execution") {
        if (mutation.mutation_params.size() >= 2) {
            current_context_.spread_threshold = mutation.mutation_params[0];
            current_context_.execution_speed_factor = mutation.mutation_params[1];
        }
    }
    else if (mutation.mutation_type == "correlation") {
        // Update correlation-based parameters
        if (mutation.mutation_params.size() >= 3) {
            current_context_.min_edge_confidence = mutation.mutation_params[0];
            current_context_.risk_multiplier = mutation.mutation_params[1];
            current_context_.execution_speed_factor = mutation.mutation_params[2];
        }
    }
    
    // Apply adaptation rate as a global modifier
    current_context_.min_edge_confidence *= (1.0 + mutation.adaptation_rate * 0.1);
    current_context_.risk_multiplier *= (1.0 + mutation.adaptation_rate * 0.05);
    current_context_.execution_speed_factor *= (1.0 + mutation.adaptation_rate * 0.15);
    
    // Ensure bounds
    current_context_.min_edge_confidence = std::max(0.1, std::min(1.0, current_context_.min_edge_confidence));
    current_context_.risk_multiplier = std::max(0.1, std::min(5.0, current_context_.risk_multiplier));
    current_context_.execution_speed_factor = std::max(0.1, std::min(10.0, current_context_.execution_speed_factor));
}

ExecutionContext MetamorphicCore::getExecutionContext() const {
    std::lock_guard<std::mutex> lock(context_mutex_);
    return current_context_;
}

bool MetamorphicCore::shouldExecuteTrade(const std::string& symbol,
                                         double edge_confidence,
                                         double spread,
                                         double risk_score) const {
    std::lock_guard<std::mutex> lock(context_mutex_);
    
    // Apply mutated logic to trading decisions
    if (edge_confidence < current_context_.min_edge_confidence) {
        return false;
    }
    
    if (spread > current_context_.spread_threshold) {
        return false;
    }
    
    // Risk-adjusted decision
    double adjusted_risk = risk_score * current_context_.risk_multiplier;
    if (adjusted_risk > 0.8) {
        return false;
    }
    
    return true;
}

MetamorphicCore::MutationStats MetamorphicCore::getMutationStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void MetamorphicCore::mutationReceiverThread() {
    if (!zmq_socket_) {
        return;
    }
    
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    
    while (should_run_.load()) {
        int rc = zmq_recv(zmq_socket_, &msg, ZMQ_DONTWAIT);
        
        if (rc > 0) {
            // Parse mutation signal from Python Brain
            // Format: "mutation_id|timestamp|type|rate|param1,param2,..."
            std::string data(static_cast<char*>(zmq_msg_data(&msg)), zmq_msg_size(&msg));
            
            LogicMutation mutation;
            std::istringstream iss(data);
            std::string token;
            std::vector<std::string> tokens;
            
            while (std::getline(iss, token, '|')) {
                tokens.push_back(token);
            }
            
            if (tokens.size() >= 5) {
                mutation.mutation_id = std::stoul(tokens[0]);
                mutation.timestamp_ns = std::stoull(tokens[1]);
                mutation.mutation_type = tokens[2];
                mutation.adaptation_rate = std::stod(tokens[3]);
                
                // Parse parameters
                std::istringstream param_stream(tokens[4]);
                std::string param;
                while (std::getline(param_stream, param, ',')) {
                    mutation.mutation_params.push_back(std::stod(param));
                }
                
                processMutationSignal(mutation);
            }
        }
        
        // Sleep for 10ms to allow other processing
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    zmq_msg_close(&msg);
}

void MetamorphicCore::startMutationReceiver() {
    if (should_run_.load()) {
        return;
    }
    
    should_run_.store(true);
    receiver_thread_ = std::thread(&MetamorphicCore::mutationReceiverThread, this);
}

void MetamorphicCore::stopMutationReceiver() {
    should_run_.store(false);
    if (receiver_thread_.joinable()) {
        receiver_thread_.join();
    }
}

} // namespace hean
