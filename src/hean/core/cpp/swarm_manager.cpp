/**
 * Swarm Intelligence Consensus Engine Implementation
 * Multi-Agent Decision System with Fast-Voting Mechanism
 */

#include "swarm_manager.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>
#include <string>

SwarmManager::SwarmManager(int num_agents, double consensus_threshold)
    : agent_counter_(0), consensus_threshold_(consensus_threshold), max_agents_(num_agents) {
    initialize_swarm();
}

SwarmManager::~SwarmManager() {
    agents_.clear();
}

void SwarmManager::initialize_swarm() {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    agents_.clear();
    agent_counter_ = 0;
    
    // Allocate agents by specialization type
    // 25% Delta Analyzers
    int delta_agents = static_cast<int>(max_agents_ * 0.25);
    // 30% OFI Analyzers (most important for order flow)
    int ofi_agents = static_cast<int>(max_agents_ * 0.30);
    // 25% VPIN Analyzers
    int vpin_agents = static_cast<int>(max_agents_ * 0.25);
    // 15% Micro-Momentum Analyzers
    int momentum_agents = static_cast<int>(max_agents_ * 0.15);
    // 5% Combiners (synthesize all signals)
    int combiner_agents = max_agents_ - (delta_agents + ofi_agents + vpin_agents + momentum_agents);
    
    // Create Delta Analyzers
    for (int i = 0; i < delta_agents; i++) {
        agents_.push_back(std::make_unique<DecisionAgent>(
            agent_counter_++, AgentType::DELTA_ANALYZER, 30 + (i % 20)
        ));
    }
    
    // Create OFI Analyzers (varying window sizes)
    for (int i = 0; i < ofi_agents; i++) {
        agents_.push_back(std::make_unique<DecisionAgent>(
            agent_counter_++, AgentType::OFI_ANALYZER, 20 + (i % 30)
        ));
    }
    
    // Create VPIN Analyzers
    for (int i = 0; i < vpin_agents; i++) {
        agents_.push_back(std::make_unique<DecisionAgent>(
            agent_counter_++, AgentType::VPIN_ANALYZER, 50 + (i % 50)
        ));
    }
    
    // Create Micro-Momentum Analyzers (shorter windows)
    for (int i = 0; i < momentum_agents; i++) {
        agents_.push_back(std::make_unique<DecisionAgent>(
            agent_counter_++, AgentType::MICRO_MOMENTUM, 10 + (i % 15)
        ));
    }
    
    // Create Combiners
    for (int i = 0; i < combiner_agents; i++) {
        agents_.push_back(std::make_unique<DecisionAgent>(
            agent_counter_++, AgentType::MOMENTUM_COMBINER, 100
        ));
    }
}

void SwarmManager::update_orderflow(const std::string& symbol, const OrderflowSnapshot& snapshot) {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    auto now = std::chrono::steady_clock::now();
    
    for (auto& agent : agents_) {
        // Update feature history
        agent->feature_history.push_back(snapshot);
        if (agent->feature_history.size() > static_cast<size_t>(agent->window_size)) {
            agent->feature_history.pop_front();
        }
        
        // Make decision based on agent type
        AgentDecision decision = AgentDecision::NEUTRAL;
        
        if (agent->feature_history.size() >= 5) {  // Need minimum history
            switch (agent->agent_type) {
                case AgentType::DELTA_ANALYZER:
                    decision = analyze_delta(agent.get(), snapshot);
                    break;
                case AgentType::OFI_ANALYZER:
                    decision = analyze_ofi(agent.get(), snapshot);
                    break;
                case AgentType::VPIN_ANALYZER:
                    decision = analyze_vpin(agent.get(), snapshot);
                    break;
                case AgentType::MICRO_MOMENTUM:
                    decision = analyze_micro_momentum(agent.get(), snapshot);
                    break;
                case AgentType::MOMENTUM_COMBINER:
                    // Combiner: average all other signals
                    {
                        double avg_delta = 0.0, avg_ofi = 0.0, avg_vpin = 0.0, avg_mom = 0.0;
                        int count = 0;
                        for (const auto& snap : agent->feature_history) {
                            avg_delta += snap.delta;
                            avg_ofi += snap.ofi;
                            avg_vpin += snap.vpin;
                            avg_mom += snap.micro_momentum;
                            count++;
                        }
                        if (count > 0) {
                            avg_delta /= count;
                            avg_ofi /= count;
                            avg_vpin /= count;
                            avg_mom /= count;
                            
                            // Weighted combination
                            double combined = (avg_delta * 0.3) + (avg_ofi * 0.4) + 
                                            (avg_vpin * 0.2) + (avg_mom * 0.1);
                            decision = (combined > 0.1) ? AgentDecision::BUY : 
                                      (combined < -0.1) ? AgentDecision::SELL : AgentDecision::NEUTRAL;
                        }
                    }
                    break;
            }
            
            // Calculate confidence
            agent->confidence = calculate_confidence(agent.get(), decision);
            agent->current_decision = decision;
            agent->last_update = now;
        }
    }
}

AgentDecision SwarmManager::analyze_delta(DecisionAgent* agent, const OrderflowSnapshot& snapshot) {
    if (agent->feature_history.size() < 3) {
        return AgentDecision::NEUTRAL;
    }
    
    // Calculate rolling average delta
    double sum_delta = 0.0;
    int count = 0;
    for (const auto& snap : agent->feature_history) {
        sum_delta += snap.delta;
        count++;
    }
    double avg_delta = sum_delta / count;
    
    // Calculate delta momentum (change in delta)
    double recent_delta = snapshot.delta;
    double momentum = recent_delta - avg_delta;
    
    // Normalize by total volume
    double total_volume = snapshot.bid_volume + snapshot.ask_volume;
    if (total_volume > 0) {
        momentum = momentum / total_volume;
    }
    
    // Threshold-based decision
    if (momentum > 0.05) {  // Significant buying pressure
        agent->signal_strength = std::min(1.0, std::abs(momentum) * 10.0);
        return AgentDecision::BUY;
    } else if (momentum < -0.05) {  // Significant selling pressure
        agent->signal_strength = std::min(1.0, std::abs(momentum) * 10.0);
        return AgentDecision::SELL;
    }
    
    agent->signal_strength = 0.0;
    return AgentDecision::NEUTRAL;
}

AgentDecision SwarmManager::analyze_ofi(DecisionAgent* agent, const OrderflowSnapshot& snapshot) {
    if (agent->feature_history.size() < 3) {
        return AgentDecision::NEUTRAL;
    }
    
    // OFI is already calculated, analyze trend
    std::vector<double> ofi_values;
    for (const auto& snap : agent->feature_history) {
        ofi_values.push_back(snap.ofi);
    }
    
    // Calculate OFI momentum (slope)
    double recent_ofi = snapshot.ofi;
    double avg_ofi = std::accumulate(ofi_values.begin(), ofi_values.end(), 0.0) / ofi_values.size();
    
    // OFI threshold: positive = buying pressure, negative = selling pressure
    double ofi_strength = recent_ofi;
    
    // Normalize to [-1, 1] range
    double max_ofi = *std::max_element(ofi_values.begin(), ofi_values.end());
    double min_ofi = *std::min_element(ofi_values.begin(), ofi_values.end());
    double ofi_range = max_ofi - min_ofi;
    
    if (ofi_range > 0) {
        ofi_strength = (recent_ofi - min_ofi) / ofi_range * 2.0 - 1.0;  // Map to [-1, 1]
    }
    
    if (ofi_strength > 0.3) {  // Strong buying pressure
        agent->signal_strength = std::min(1.0, std::abs(ofi_strength));
        return AgentDecision::BUY;
    } else if (ofi_strength < -0.3) {  // Strong selling pressure
        agent->signal_strength = std::min(1.0, std::abs(ofi_strength));
        return AgentDecision::SELL;
    }
    
    agent->signal_strength = 0.0;
    return AgentDecision::NEUTRAL;
}

AgentDecision SwarmManager::analyze_vpin(DecisionAgent* agent, const OrderflowSnapshot& snapshot) {
    if (agent->feature_history.size() < 10) {
        return AgentDecision::NEUTRAL;
    }
    
    // VPIN analysis: high VPIN indicates informed trading
    std::vector<double> vpin_values;
    for (const auto& snap : agent->feature_history) {
        vpin_values.push_back(snap.vpin);
    }
    
    double recent_vpin = snapshot.vpin;
    double avg_vpin = std::accumulate(vpin_values.begin(), vpin_values.end(), 0.0) / vpin_values.size();
    
    // High VPIN with positive delta = informed buying
    // High VPIN with negative delta = informed selling
    double vpin_spike = recent_vpin - avg_vpin;
    
    if (vpin_spike > 0.2 && snapshot.delta > 0) {  // Informed buying
        agent->signal_strength = std::min(1.0, vpin_spike * 2.0);
        return AgentDecision::BUY;
    } else if (vpin_spike > 0.2 && snapshot.delta < 0) {  // Informed selling
        agent->signal_strength = std::min(1.0, vpin_spike * 2.0);
        return AgentDecision::SELL;
    }
    
    agent->signal_strength = 0.0;
    return AgentDecision::NEUTRAL;
}

AgentDecision SwarmManager::analyze_micro_momentum(DecisionAgent* agent, const OrderflowSnapshot& snapshot) {
    if (agent->feature_history.size() < 3) {
        return AgentDecision::NEUTRAL;
    }
    
    // Micro-momentum: analyze price changes at tick level
    std::vector<double> prices;
    for (const auto& snap : agent->feature_history) {
        prices.push_back(snap.price);
    }
    
    // Calculate price momentum (acceleration)
    if (prices.size() < 3) {
        return AgentDecision::NEUTRAL;
    }
    
    double recent_price = prices.back();
    double prev_price = prices[prices.size() - 2];
    double prev_prev_price = prices[prices.size() - 3];
    
    double velocity = (recent_price - prev_price) / prev_price;
    double prev_velocity = (prev_price - prev_prev_price) / prev_prev_price;
    double acceleration = velocity - prev_velocity;
    
    // Combine with micro-momentum signal from snapshot
    double combined_momentum = (acceleration * 0.5) + (snapshot.micro_momentum * 0.5);
    
    if (combined_momentum > 0.0001) {  // Positive acceleration
        agent->signal_strength = std::min(1.0, std::abs(combined_momentum) * 1000.0);
        return AgentDecision::BUY;
    } else if (combined_momentum < -0.0001) {  // Negative acceleration
        agent->signal_strength = std::min(1.0, std::abs(combined_momentum) * 1000.0);
        return AgentDecision::SELL;
    }
    
    agent->signal_strength = 0.0;
    return AgentDecision::NEUTRAL;
}

double SwarmManager::calculate_confidence(DecisionAgent* agent, AgentDecision decision) {
    if (decision == AgentDecision::NEUTRAL) {
        return 0.0;
    }
    
    // Base confidence from signal strength
    double base_confidence = agent->signal_strength;
    
    // Increase confidence based on consistency (how often agent agrees with itself)
    if (agent->feature_history.size() >= 5) {
        int consistent_decisions = 0;
        AgentDecision prev_decision = agent->current_decision;
        
        // Check last 5 decisions (if we tracked them)
        // For now, use signal consistency over time
        double signal_variance = 0.0;
        std::vector<double> signals;
        
        for (const auto& snap : agent->feature_history) {
            double signal = 0.0;
            switch (agent->agent_type) {
                case AgentType::DELTA_ANALYZER:
                    signal = snap.delta;
                    break;
                case AgentType::OFI_ANALYZER:
                    signal = snap.ofi;
                    break;
                case AgentType::VPIN_ANALYZER:
                    signal = snap.vpin;
                    break;
                case AgentType::MICRO_MOMENTUM:
                    signal = snap.micro_momentum;
                    break;
                default:
                    break;
            }
            signals.push_back(signal);
        }
        
        if (signals.size() > 1) {
            double mean = std::accumulate(signals.begin(), signals.end(), 0.0) / signals.size();
            for (double s : signals) {
                signal_variance += (s - mean) * (s - mean);
            }
            signal_variance /= signals.size();
        }
        
        // Lower variance = higher confidence (more consistent)
        double consistency_boost = 1.0 / (1.0 + signal_variance * 10.0);
        base_confidence = base_confidence * (0.7 + 0.3 * consistency_boost);
    }
    
    return std::min(1.0, base_confidence);
}

ConsensusResult SwarmManager::fast_vote(const std::string& symbol) {
    ConsensusResult result;
    
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    result.total_agents = static_cast<int>(agents_.size());
    
    double total_buy_confidence = 0.0;
    double total_sell_confidence = 0.0;
    
    for (const auto& agent : agents_) {
        // Only count agents with sufficient history
        if (agent->feature_history.size() < 5) {
            continue;
        }
        
        if (agent->current_decision == AgentDecision::BUY) {
            result.buy_votes++;
            total_buy_confidence += agent->confidence;
        } else if (agent->current_decision == AgentDecision::SELL) {
            result.sell_votes++;
            total_sell_confidence += agent->confidence;
        }
    }
    
    int total_votes = result.buy_votes + result.sell_votes;
    
    if (total_votes > 0) {
        result.buy_vote_percentage = (static_cast<double>(result.buy_votes) / total_votes) * 100.0;
        result.sell_vote_percentage = (static_cast<double>(result.sell_votes) / total_votes) * 100.0;
        
        // Calculate average confidence of voting agents
        double total_conf = total_buy_confidence + total_sell_confidence;
        result.average_confidence = total_conf / total_votes;
        
        // Determine consensus
        if (result.buy_vote_percentage >= consensus_threshold_ * 100.0) {
            result.consensus = AgentDecision::BUY;
            result.consensus_reached = true;
            result.execution_signal_strength = (result.buy_vote_percentage / 100.0) * result.average_confidence;
        } else if (result.sell_vote_percentage >= consensus_threshold_ * 100.0) {
            result.consensus = AgentDecision::SELL;
            result.consensus_reached = true;
            result.execution_signal_strength = (result.sell_vote_percentage / 100.0) * result.average_confidence;
        } else {
            // No consensus - determine majority
            result.consensus_reached = false;
            if (result.buy_vote_percentage > result.sell_vote_percentage) {
                result.consensus = AgentDecision::BUY;
            } else if (result.sell_vote_percentage > result.buy_vote_percentage) {
                result.consensus = AgentDecision::SELL;
            } else {
                result.consensus = AgentDecision::NEUTRAL;
            }
            // Calculate signal strength even without consensus (weaker signal)
            double majority_percentage = std::max(result.buy_vote_percentage, result.sell_vote_percentage);
            result.execution_signal_strength = (majority_percentage / 100.0) * result.average_confidence * 0.5;  // 50% penalty for no consensus
        }
    }
    
    return result;
}

ConsensusResult SwarmManager::get_consensus(const std::string& symbol) {
    return fast_vote(symbol);
}

ConsensusResult SwarmManager::get_detailed_consensus(const std::string& symbol) {
    return fast_vote(symbol);
}

double SwarmManager::get_consensus_confidence(const std::string& symbol) {
    ConsensusResult result = fast_vote(symbol);
    return result.execution_signal_strength;
}

std::map<AgentType, int> SwarmManager::get_agent_distribution() const {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    std::map<AgentType, int> distribution;
    for (const auto& agent : agents_) {
        distribution[agent->agent_type]++;
    }
    
    return distribution;
}

void SwarmManager::reset_agents() {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    for (auto& agent : agents_) {
        agent->feature_history.clear();
        agent->current_decision = AgentDecision::NEUTRAL;
        agent->confidence = 0.0;
        agent->signal_strength = 0.0;
    }
}
