/**
 * Transcendent Entity Interface Server
 * High-performance uWebSockets server for HTML5/WebGL dashboard
 * 
 * Serves a minimalist dashboard with:
 * - Three.js 3D Singularity visualization
 * - Force-directed Causal Mesh graph
 * - Real-time binary WebSocket data binding
 */

#ifndef TRANSCENDENT_ENTITY_SERVER_H
#define TRANSCENDENT_ENTITY_SERVER_H

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <functional>
#include <mutex>
#include <map>

// Forward declarations for uWebSockets
struct us_listen_socket_t;
typedef struct us_listen_socket_t us_listen_socket_t;

namespace uWS {
    template <bool SSL>
    struct TemplatedApp;
    typedef TemplatedApp<false> App;
    
    template <bool SSL>
    struct WebSocket;
    typedef WebSocket<false> WebSocket;
    
    template <bool SSL>
    struct HttpRequest;
    typedef HttpRequest<false> HttpRequest;
    
    template <bool SSL>
    struct HttpResponse;
    typedef HttpResponse<false> HttpResponse;
}

// Shared memory data structure (matches Python bridge)
struct TranscendentEntityData {
    double balance;              // Extraction capacity ($300)
    double realtime_pnl;         // Real-time extraction delta
    double volatility;           // Market entropy (0-1)
    double confidence;           // AI confidence (0-1)
    char logic_state[64];        // Current logic state string
    uint64_t timestamp_ns;       // Timestamp in nanoseconds
    
    // Causal Mesh data
    struct CausalNode {
        char symbol[16];         // Asset symbol
        double influence;        // Influence strength (0-1)
        double position_x;     // Graph position X
        double position_y;       // Graph position Y
    };
    
    struct CausalEdge {
        char source[16];         // Source symbol
        char target[16];         // Target symbol
        double strength;         // Edge strength (0-1)
        int lag_ms;              // Time lag in milliseconds
    };
    
    uint32_t node_count;         // Number of causal nodes
    uint32_t edge_count;         // Number of causal edges
    CausalNode nodes[100];       // Max 100 nodes
    CausalEdge edges[500];       // Max 500 edges
};

class TranscendentEntityServer {
public:
    typedef std::function<void(double balance, double pnl, double volatility, 
                              double confidence, const std::string& logic_state)> DataCallback;
    
    TranscendentEntityServer(int port = 8888);
    ~TranscendentEntityServer();
    
    // Server control
    bool start();
    void stop();
    bool is_running() const { return running_.load(); }
    
    // Data update (called from Python bridge)
    void update_data(const TranscendentEntityData& data);
    
    // Get current data snapshot
    TranscendentEntityData get_data_snapshot() const;
    
    // Set HTML content path (default: web/transcendent_entity.html)
    void set_html_path(const std::string& path) { html_path_ = path; }
    
private:
    int port_;
    std::string html_path_;
    std::atomic<bool> running_;
    std::atomic<bool> should_stop_;
    
    // uWebSockets app
    std::unique_ptr<uWS::App> app_;
    us_listen_socket_t* listen_socket_;
    std::thread server_thread_;
    
    // Shared data (protected by mutex)
    mutable std::mutex data_mutex_;
    TranscendentEntityData current_data_;
    
    // WebSocket connections
    mutable std::mutex ws_mutex_;
    std::map<void*, uWS::WebSocket<false>*> ws_connections_;
    
    // Internal methods
    void server_loop();
    void handle_http_request(uWS::HttpResponse<false>* res, uWS::HttpRequest<false>* req);
    void handle_websocket_open(uWS::WebSocket<false>* ws);
    void handle_websocket_message(uWS::WebSocket<false>* ws, std::string_view message, uWS::OpCode opcode);
    void handle_websocket_close(uWS::WebSocket<false>* ws, int code, std::string_view message);
    void broadcast_data();
    
    // HTML content loading
    std::string load_html_content() const;
    
    // Binary message encoding
    std::string encode_binary_message(const TranscendentEntityData& data) const;
};

#endif // TRANSCENDENT_ENTITY_SERVER_H
