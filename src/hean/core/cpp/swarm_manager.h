/**
 * Swarm Intelligence Consensus Engine
 * Multi-Agent Decision System with Fast-Voting Mechanism
 * 
 * Features:
 * - 100+ lightweight Decision Agents analyzing Orderflow sub-features
 * - Fast-Voting: >80% consensus triggers execution
 * - Specialized agents for Delta, OFI, VPIN, Micro-Momentum
 */

#ifndef SWARM_MANAGER_H
#define SWARM_MANAGER_H

#include <vector>
#include <string>
#include <map>
#include <deque>
#include <mutex>
#include <atomic>
#include <chrono>
#include <memory>

// Agent decision types
enum class AgentDecision {
    BUY = 1,
    SELL = -1,
    NEUTRAL = 0
};

// Agent specialization types
enum class AgentType {
    DELTA_ANALYZER = 0,      // Analyzes delta (buy vs sell volume)
    OFI_ANALYZER = 1,        // Analyzes Order-Flow Imbalance
    VPIN_ANALYZER = 2,       // Analyzes Volume-synchronized Probability of Informed trading
    MICRO_MOMENTUM = 3,      // Analyzes micro-momentum (tick-level patterns)
    MOMENTUM_COMBINER = 4    // Combines all signals
};

// Orderflow feature snapshot
struct OrderflowSnapshot {
    double delta;                    // Net buy volume - sell volume
    double ofi;                      // Order-Flow Imbalance
    double vpin;                     // Volume-synchronized PIN
    double micro_momentum;           // Micro-momentum score
    double price;                    // Current price
    double bid_volume;               // Total bid volume
    double ask_volume;               // Total ask volume
    int64_t timestamp_ns;            // Nanosecond timestamp
    
    OrderflowSnapshot() : delta(0.0), ofi(0.0), vpin(0.0), micro_momentum(0.0),
                         price(0.0), bid_volume(0.0), ask_volume(0.0), timestamp_ns(0) {}
};

// Individual Decision Agent
struct DecisionAgent {
    int agent_id;
    AgentType agent_type;
    AgentDecision current_decision;
    double confidence;               // 0.0 to 1.0
    double signal_strength;          // Normalized signal strength
    std::deque<OrderflowSnapshot> feature_history;  // Rolling window of features
    int window_size;
    std::chrono::steady_clock::time_point last_update;
    
    DecisionAgent(int id, AgentType type, int window = 50)
        : agent_id(id), agent_type(type), current_decision(AgentDecision::NEUTRAL),
          confidence(0.0), signal_strength(0.0), window_size(window) {
        last_update = std::chrono::steady_clock::now();
    }
};

// Consensus result
struct ConsensusResult {
    AgentDecision consensus;         // BUY, SELL, or NEUTRAL
    double buy_vote_percentage;      // Percentage voting BUY
    double sell_vote_percentage;     // Percentage voting SELL
    double average_confidence;       // Average confidence of voting agents
    int total_agents;
    int buy_votes;
    int sell_votes;
    bool consensus_reached;          // True if >80% consensus achieved
    double execution_signal_strength; // Combined signal strength for execution
    
    ConsensusResult() : consensus(AgentDecision::NEUTRAL), buy_vote_percentage(0.0),
                       sell_vote_percentage(0.0), average_confidence(0.0), total_agents(0),
                       buy_votes(0), sell_votes(0), consensus_reached(false),
                       execution_signal_strength(0.0) {}
};

// Swarm Manager class
class SwarmManager {
private:
    std::vector<std::unique_ptr<DecisionAgent>> agents_;
    std::mutex agents_mutex_;
    std::atomic<int> agent_counter_;
    double consensus_threshold_;      // Default: 0.80 (80%)
    int max_agents_;
    
    // Agent decision making
    AgentDecision analyze_delta(DecisionAgent* agent, const OrderflowSnapshot& snapshot);
    AgentDecision analyze_ofi(DecisionAgent* agent, const OrderflowSnapshot& snapshot);
    AgentDecision analyze_vpin(DecisionAgent* agent, const OrderflowSnapshot& snapshot);
    AgentDecision analyze_micro_momentum(DecisionAgent* agent, const OrderflowSnapshot& snapshot);
    
    // Calculate confidence based on signal strength and consistency
    double calculate_confidence(DecisionAgent* agent, AgentDecision decision);
    
    // Fast voting mechanism
    ConsensusResult fast_vote(const std::string& symbol);
    
public:
    SwarmManager(int num_agents = 100, double consensus_threshold = 0.80);
    ~SwarmManager();
    
    // Initialize swarm with specialized agents
    void initialize_swarm();
    
    // Update orderflow features for all agents
    void update_orderflow(const std::string& symbol, const OrderflowSnapshot& snapshot);
    
    // Get consensus decision (Fast-Voting)
    ConsensusResult get_consensus(const std::string& symbol);
    
    // Get agent statistics
    std::map<AgentType, int> get_agent_distribution() const;
    
    // Get consensus confidence level (0.0 to 1.0)
    double get_consensus_confidence(const std::string& symbol);
    
    // Reset agents (for new trading session)
    void reset_agents();
    
    // Get detailed consensus breakdown
    ConsensusResult get_detailed_consensus(const std::string& symbol);
};

#endif // SWARM_MANAGER_H
