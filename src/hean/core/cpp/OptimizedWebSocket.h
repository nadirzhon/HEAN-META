/**
 * Optimized WebSocket Client with TCP/IP Socket Optimizations
 * 
 * - TCP_NODELAY: Disable Nagle's algorithm for low latency
 * - SO_PRIORITY: Set socket priority for kernel scheduling
 * - Multi-threaded polling: Never let WebSocket sleep
 */

#ifndef OPTIMIZED_WEBSOCKET_H
#define OPTIMIZED_WEBSOCKET_H

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "MemoryPool.h"

#ifdef __linux__
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#elif defined(__APPLE__)
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <netinet/in.h>
#include <sys/event.h>
#endif

// Forward declaration
struct WebSocketMessage;

/**
 * Optimized WebSocket client with TCP/IP optimizations
 */
class OptimizedWebSocket {
public:
    typedef std::function<void(const std::string& message, int64_t timestamp_ns)> MessageCallback;
    
    OptimizedWebSocket();
    ~OptimizedWebSocket();
    
    // Connection
    bool connect(const std::string& url);
    void disconnect();
    bool is_connected() const { return connected_.load(); }
    
    // Message handling
    void set_message_callback(MessageCallback callback);
    
    // Sending
    bool send(const std::string& message);
    
    // TCP/IP optimizations
    bool set_tcp_nodelay(bool enable = true);
    bool set_socket_priority(int priority = 6);  // 0-6, 6 = highest
    bool set_receive_buffer_size(int size = 64 * 1024);  // 64KB
    bool set_send_buffer_size(int size = 64 * 1024);
    
    // Multi-threaded polling
    void start_polling_thread();
    void stop_polling_thread();
    
    // Statistics
    int64_t get_total_messages() const { return total_messages_.load(); }
    int64_t get_total_bytes() const { return total_bytes_.load(); }
    int64_t get_avg_latency_ns() const;
    int64_t get_max_latency_ns() const { return max_latency_ns_.load(); }
    
private:
    std::string url_;
    int socket_fd_;
    std::atomic<bool> connected_;
    std::atomic<bool> polling_running_;
    
    // Message callback
    MessageCallback message_callback_;
    std::mutex callback_mutex_;
    
    // Polling thread
    std::thread polling_thread_;
    
    // Statistics
    std::atomic<int64_t> total_messages_;
    std::atomic<int64_t> total_bytes_;
    std::atomic<int64_t> max_latency_ns_;
    std::atomic<int64_t> total_latency_ns_;
    std::atomic<int64_t> latency_count_;
    
    // Internal methods
    void polling_loop();
    bool perform_handshake();
    bool read_message(std::string& message);
    bool write_message(const std::string& message);
    int64_t get_timestamp_ns() const;
    
    // WebSocket frame parsing (simplified)
    bool parse_websocket_frame(const std::vector<uint8_t>& frame, std::string& message);
    std::vector<uint8_t> create_websocket_frame(const std::string& message);
    
    // Helper: base64 encoding (for handshake)
    std::string base64_encode(const std::string& data);
};

#endif // OPTIMIZED_WEBSOCKET_H
