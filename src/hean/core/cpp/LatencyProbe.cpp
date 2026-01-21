/**
 * Ultra-Low Latency (ULL) Trading Engine - High-Precision Jitter Monitor
 * Phase 9: Network infrastructure exploitation for ultra-low latency execution
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// High-precision timing
using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::high_resolution_clock::time_point;
using Nanoseconds = std::chrono::nanoseconds;
using Microseconds = std::chrono::microseconds;
using Milliseconds = std::chrono::milliseconds;

// Network congestion detection parameters
#define CONGESTION_WINDOW_SIZE 100
#define CONGESTION_THRESHOLD_MS 10.0  // 10ms jitter threshold for congestion detection
#define CONGESTION_DEVIATION_MULTIPLIER 2.5  // Standard deviation multiplier
#define PULSE_INTERVAL_MS 50  // Send pulse every 50ms

// Packet pacing parameters
#define PACKET_PACING_WINDOW_MS 1000  // 1 second window
#define MAX_PACKETS_PER_WINDOW 50  // Max 50 packets per second to avoid burst limiters
#define MIN_PACKET_SPACING_MS 20  // Minimum 20ms between packets

// Latency measurement structure
struct LatencyMeasurement {
    int64_t send_timestamp_ns;
    int64_t receive_timestamp_ns;
    int64_t round_trip_time_ns;
    bool valid;
};

// Congestion window state
struct CongestionState {
    std::deque<double> rtt_history_ms;
    double mean_rtt_ms;
    double std_dev_rtt_ms;
    bool congestion_detected;
    int congestion_window_size;
    std::atomic<bool> preemptive_mode;
};

// Packet pacing state
struct PacingState {
    std::deque<int64_t> packet_timestamps_ns;
    std::atomic<int64_t> next_allowed_packet_ns;
    std::mutex pacing_mutex;
    int packets_in_window;
};

// Latency Probe class
class LatencyProbe {
private:
    std::string api_endpoint_;
    bool running_;
    std::thread probe_thread_;
    std::mutex state_mutex_;
    
    // Latency tracking
    std::deque<LatencyMeasurement> measurements_;
    std::atomic<int64_t> last_rtt_ns_;
    std::atomic<double> current_jitter_ms_;
    
    // Congestion detection
    CongestionState congestion_state_;
    
    // Packet pacing
    PacingState pacing_state_;
    
    // Statistics
    std::atomic<int64_t> total_pulses_sent_;
    std::atomic<int64_t> total_pulses_received_;
    std::atomic<double> min_rtt_ms_;
    std::atomic<double> max_rtt_ms_;
    std::atomic<double> avg_rtt_ms_;
    
    // Calculate RTT statistics
    void update_rtt_statistics() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        
        if (congestion_state_.rtt_history_ms.empty()) {
            return;
        }
        
        // Calculate mean
        double sum = 0.0;
        for (double rtt : congestion_state_.rtt_history_ms) {
            sum += rtt;
        }
        congestion_state_.mean_rtt_ms = sum / congestion_state_.rtt_history_ms.size();
        
        // Calculate standard deviation
        double variance = 0.0;
        for (double rtt : congestion_state_.rtt_history_ms) {
            double diff = rtt - congestion_state_.mean_rtt_ms;
            variance += diff * diff;
        }
        congestion_state_.std_dev_rtt_ms = std::sqrt(variance / congestion_state_.rtt_history_ms.size());
        
        // Detect congestion: RTT spikes beyond threshold or high deviation
        bool high_jitter = congestion_state_.std_dev_rtt_ms > CONGESTION_THRESHOLD_MS;
        bool rtt_spike = false;
        if (!congestion_state_.rtt_history_ms.empty()) {
            double latest_rtt = congestion_state_.rtt_history_ms.back();
            double threshold = congestion_state_.mean_rtt_ms + 
                              (CONGESTION_DEVIATION_MULTIPLIER * congestion_state_.std_dev_rtt_ms);
            rtt_spike = latest_rtt > threshold;
        }
        
        congestion_state_.congestion_detected = high_jitter || rtt_spike;
        
        // Update jitter metric
        current_jitter_ms_.store(congestion_state_.std_dev_rtt_ms);
        
        // Switch to pre-emptive mode if congestion detected
        if (congestion_state_.congestion_detected && !congestion_state_.preemptive_mode.load()) {
            congestion_state_.preemptive_mode.store(true);
        } else if (!congestion_state_.congestion_detected && congestion_state_.preemptive_mode.load()) {
            // Exit pre-emptive mode after stability period
            congestion_state_.preemptive_mode.store(false);
        }
    }
    
    // Simulate sending pulse to API endpoint (in real implementation, this would be an HTTP request)
    // Returns: receive timestamp in nanoseconds
    int64_t send_pulse_and_measure() {
        TimePoint send_time = Clock::now();
        int64_t send_ns = std::chrono::duration_cast<Nanoseconds>(
            send_time.time_since_epoch()
        ).count();
        
        // In production, this would make an actual HTTP request to the Bybit API endpoint
        // For now, simulate network delay with some jitter
        std::this_thread::sleep_for(Microseconds(1000 + (rand() % 500)));  // 1-1.5ms simulated delay
        
        TimePoint receive_time = Clock::now();
        int64_t receive_ns = std::chrono::duration_cast<Nanoseconds>(
            receive_time.time_since_epoch()
        ).count();
        
        int64_t rtt_ns = receive_ns - send_ns;
        
        // Store measurement
        LatencyMeasurement measurement;
        measurement.send_timestamp_ns = send_ns;
        measurement.receive_timestamp_ns = receive_ns;
        measurement.round_trip_time_ns = rtt_ns;
        measurement.valid = true;
        
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            measurements_.push_back(measurement);
            if (measurements_.size() > 1000) {
                measurements_.pop_front();
            }
            
            // Update congestion tracking
            double rtt_ms = rtt_ns / 1000000.0;  // Convert to milliseconds
            congestion_state_.rtt_history_ms.push_back(rtt_ms);
            if (congestion_state_.rtt_history_ms.size() > CONGESTION_WINDOW_SIZE) {
                congestion_state_.rtt_history_ms.pop_front();
            }
            
            // Update statistics
            last_rtt_ns_.store(rtt_ns);
            if (measurements_.size() == 1) {
                min_rtt_ms_.store(rtt_ms);
                max_rtt_ms_.store(rtt_ms);
                avg_rtt_ms_.store(rtt_ms);
            } else {
                if (rtt_ms < min_rtt_ms_.load()) {
                    min_rtt_ms_.store(rtt_ms);
                }
                if (rtt_ms > max_rtt_ms_.load()) {
                    max_rtt_ms_.store(rtt_ms);
                }
                // Update running average
                double current_avg = avg_rtt_ms_.load();
                int count = measurements_.size();
                double new_avg = current_avg + (rtt_ms - current_avg) / count;
                avg_rtt_ms_.store(new_avg);
            }
        }
        
        total_pulses_sent_.fetch_add(1);
        total_pulses_received_.fetch_add(1);
        
        return receive_ns;
    }
    
    // Packet pacing: ensure we don't exceed exchange burst limiters
    bool can_send_packet() {
        std::lock_guard<std::mutex> lock(pacing_state_.pacing_mutex);
        
        TimePoint now = Clock::now();
        int64_t now_ns = std::chrono::duration_cast<Nanoseconds>(
            now.time_since_epoch()
        ).count();
        
        // Check if enough time has passed since last packet
        if (now_ns < pacing_state_.next_allowed_packet_ns.load()) {
            return false;
        }
        
        // Clean old timestamps outside the window
        int64_t window_start_ns = now_ns - (PACKET_PACING_WINDOW_MS * 1000000);
        while (!pacing_state_.packet_timestamps_ns.empty() &&
               pacing_state_.packet_timestamps_ns.front() < window_start_ns) {
            pacing_state_.packet_timestamps_ns.pop_front();
            pacing_state_.packets_in_window--;
        }
        
        // Check if we're within rate limits
        if (pacing_state_.packets_in_window >= MAX_PACKETS_PER_WINDOW) {
            return false;
        }
        
        // Record this packet
        pacing_state_.packet_timestamps_ns.push_back(now_ns);
        pacing_state_.packets_in_window++;
        pacing_state_.next_allowed_packet_ns.store(
            now_ns + (MIN_PACKET_SPACING_MS * 1000000)
        );
        
        return true;
    }
    
    // Probe thread main loop
    void probe_loop() {
        while (running_) {
            try {
                // Check packet pacing before sending
                if (can_send_packet()) {
                    send_pulse_and_measure();
                    
                    // Update congestion statistics
                    update_rtt_statistics();
                } else {
                    // Wait until we can send the next packet
                    int64_t next_allowed = pacing_state_.next_allowed_packet_ns.load();
                    TimePoint now = Clock::now();
                    int64_t now_ns = std::chrono::duration_cast<Nanoseconds>(
                        now.time_since_epoch()
                    ).count();
                    
                    if (next_allowed > now_ns) {
                        int64_t wait_ns = next_allowed - now_ns;
                        std::this_thread::sleep_for(Nanoseconds(wait_ns));
                    }
                }
                
                // Sleep between pulses (pacing is handled above, but we still want some spacing)
                std::this_thread::sleep_for(Milliseconds(PULSE_INTERVAL_MS));
            } catch (...) {
                // Handle exceptions gracefully
                std::this_thread::sleep_for(Milliseconds(100));
            }
        }
    }
    
public:
    LatencyProbe(const std::string& api_endpoint) 
        : api_endpoint_(api_endpoint)
        , running_(false)
        , last_rtt_ns_(0)
        , current_jitter_ms_(0.0)
        , total_pulses_sent_(0)
        , total_pulses_received_(0)
        , min_rtt_ms_(0.0)
        , max_rtt_ms_(0.0)
        , avg_rtt_ms_(0.0)
    {
        congestion_state_.mean_rtt_ms = 0.0;
        congestion_state_.std_dev_rtt_ms = 0.0;
        congestion_state_.congestion_detected = false;
        congestion_state_.congestion_window_size = CONGESTION_WINDOW_SIZE;
        congestion_state_.preemptive_mode.store(false);
        
        pacing_state_.packets_in_window = 0;
        pacing_state_.next_allowed_packet_ns.store(0);
    }
    
    ~LatencyProbe() {
        stop();
    }
    
    void start() {
        if (running_) {
            return;
        }
        running_ = true;
        probe_thread_ = std::thread(&LatencyProbe::probe_loop, this);
    }
    
    void stop() {
        if (!running_) {
            return;
        }
        running_ = false;
        if (probe_thread_.joinable()) {
            probe_thread_.join();
        }
    }
    
    // Get current jitter in milliseconds
    double get_jitter_ms() const {
        return current_jitter_ms_.load();
    }
    
    // Get last RTT in nanoseconds
    int64_t get_last_rtt_ns() const {
        return last_rtt_ns_.load();
    }
    
    // Check if congestion is detected
    bool is_congestion_detected() const {
        return congestion_state_.congestion_detected;
    }
    
    // Check if in pre-emptive order mode
    bool is_preemptive_mode() const {
        return congestion_state_.preemptive_mode.load();
    }
    
    // Get statistics
    void get_statistics(double* min_rtt_ms, double* max_rtt_ms, double* avg_rtt_ms,
                       double* jitter_ms, int64_t* total_pulses) {
        if (min_rtt_ms) *min_rtt_ms = min_rtt_ms_.load();
        if (max_rtt_ms) *max_rtt_ms = max_rtt_ms_.load();
        if (avg_rtt_ms) *avg_rtt_ms = avg_rtt_ms_.load();
        if (jitter_ms) *jitter_ms = current_jitter_ms_.load();
        if (total_pulses) *total_pulses = total_pulses_sent_.load();
    }
    
    // Check if we can send a packet now (for external pacing control)
    bool check_packet_pacing() {
        return can_send_packet();
    }
};

// Global instance
static LatencyProbe* g_latency_probe = nullptr;

// C interface for Python bindings
extern "C" {
    void latency_probe_init(const char* api_endpoint) {
        if (g_latency_probe == nullptr) {
            g_latency_probe = new LatencyProbe(std::string(api_endpoint));
        }
    }
    
    void latency_probe_start() {
        if (g_latency_probe) {
            g_latency_probe->start();
        }
    }
    
    void latency_probe_stop() {
        if (g_latency_probe) {
            g_latency_probe->stop();
        }
    }
    
    double latency_probe_get_jitter_ms() {
        if (g_latency_probe) {
            return g_latency_probe->get_jitter_ms();
        }
        return 0.0;
    }
    
    int64_t latency_probe_get_last_rtt_ns() {
        if (g_latency_probe) {
            return g_latency_probe->get_last_rtt_ns();
        }
        return 0;
    }
    
    int latency_probe_is_congestion_detected() {
        if (g_latency_probe) {
            return g_latency_probe->is_congestion_detected() ? 1 : 0;
        }
        return 0;
    }
    
    int latency_probe_is_preemptive_mode() {
        if (g_latency_probe) {
            return g_latency_probe->is_preemptive_mode() ? 1 : 0;
        }
        return 0;
    }
    
    void latency_probe_get_statistics(double* min_rtt_ms, double* max_rtt_ms, 
                                     double* avg_rtt_ms, double* jitter_ms, 
                                     int64_t* total_pulses) {
        if (g_latency_probe) {
            g_latency_probe->get_statistics(min_rtt_ms, max_rtt_ms, avg_rtt_ms, 
                                           jitter_ms, total_pulses);
        }
    }
    
    int latency_probe_check_packet_pacing() {
        if (g_latency_probe) {
            return g_latency_probe->check_packet_pacing() ? 1 : 0;
        }
        return 0;
    }
    
    void latency_probe_cleanup() {
        if (g_latency_probe) {
            delete g_latency_probe;
            g_latency_probe = nullptr;
        }
    }
}

#ifdef __cplusplus
}
#endif
