/**
 * Optimized WebSocket Client Implementation with TCP/IP Optimizations
 */

#include "OptimizedWebSocket.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <fcntl.h>
#include <arpa/inet.h>

// Base64 encoding helper
namespace {
    static const char base64_chars[] = 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    std::string base64_encode(const std::string& data) {
        std::string encoded;
        int val = 0, valb = -6;
        for (unsigned char c : data) {
            val = (val << 8) + c;
            valb += 8;
            while (valb >= 0) {
                encoded.push_back(base64_chars[(val >> valb) & 0x3F]);
                valb -= 6;
            }
        }
        if (valb > -6) {
            encoded.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
        }
        while (encoded.size() % 4) {
            encoded.push_back('=');
        }
        return encoded;
    }
}

OptimizedWebSocket::OptimizedWebSocket()
    : socket_fd_(-1)
    , connected_(false)
    , polling_running_(false)
    , total_messages_(0)
    , total_bytes_(0)
    , max_latency_ns_(0)
    , total_latency_ns_(0)
    , latency_count_(0)
{
}

OptimizedWebSocket::~OptimizedWebSocket() {
    disconnect();
}

bool OptimizedWebSocket::connect(const std::string& url) {
    if (connected_.load()) {
        disconnect();
    }
    
    url_ = url;
    
    // Parse URL (simplified - assumes wss://host:port/path)
    std::string protocol, host, port, path;
    
    // Extract protocol
    size_t protocol_end = url.find("://");
    if (protocol_end == std::string::npos) {
        return false;
    }
    protocol = url.substr(0, protocol_end);
    
    // Extract host and port
    size_t host_start = protocol_end + 3;
    size_t path_start = url.find('/', host_start);
    if (path_start == std::string::npos) {
        path_start = url.length();
        path = "/";
    } else {
        path = url.substr(path_start);
    }
    
    std::string host_port = url.substr(host_start, path_start - host_start);
    size_t colon_pos = host_port.find(':');
    if (colon_pos == std::string::npos) {
        host = host_port;
        port = (protocol == "wss") ? "443" : "80";
    } else {
        host = host_port.substr(0, colon_pos);
        port = host_port.substr(colon_pos + 1);
    }
    
    // Create socket
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &result) != 0) {
        return false;
    }
    
    // Try to connect
    socket_fd_ = -1;
    for (struct addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
        socket_fd_ = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (socket_fd_ < 0) {
            continue;
        }
        
        // TCP/IP OPTIMIZATIONS
        // 1. TCP_NODELAY: Disable Nagle's algorithm
        int flag = 1;
        setsockopt(socket_fd_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
        
        // 2. SO_PRIORITY: Set socket priority (Linux)
        #ifdef __linux__
        int priority = 6;  // Highest priority
        setsockopt(socket_fd_, SOL_SOCKET, SO_PRIORITY, &priority, sizeof(priority));
        #endif
        
        // 3. Buffer sizes
        int recv_buf = 64 * 1024;  // 64KB
        setsockopt(socket_fd_, SOL_SOCKET, SO_RCVBUF, &recv_buf, sizeof(recv_buf));
        
        int send_buf = 64 * 1024;  // 64KB
        setsockopt(socket_fd_, SOL_SOCKET, SO_SNDBUF, &send_buf, sizeof(send_buf));
        
        // 4. Non-blocking mode for multi-threaded polling
        int flags = fcntl(socket_fd_, F_GETFL, 0);
        fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);
        
        // Connect
        if (::connect(socket_fd_, rp->ai_addr, rp->ai_addrlen) == 0) {
            break;  // Success
        }
        
        close(socket_fd_);
        socket_fd_ = -1;
    }
    
    freeaddrinfo(result);
    
    if (socket_fd_ < 0) {
        return false;
    }
    
    // Perform WebSocket handshake
    if (!perform_handshake()) {
        close(socket_fd_);
        socket_fd_ = -1;
        return false;
    }
    
    connected_.store(true);
    
    // Start polling thread
    start_polling_thread();
    
    return true;
}

void OptimizedWebSocket::disconnect() {
    if (!connected_.load()) {
        return;
    }
    
    connected_.store(false);
    stop_polling_thread();
    
    if (socket_fd_ >= 0) {
        close(socket_fd_);
        socket_fd_ = -1;
    }
}

bool OptimizedWebSocket::perform_handshake() {
    // Generate WebSocket key (simplified base64)
    std::string key = ::base64_encode(std::string(16, 'a'));  // Use anonymous namespace function
    
    // Extract path from URL
    size_t path_start = url_.find('/', url_.find("://") + 3);
    std::string path = (path_start != std::string::npos) ? url_.substr(path_start) : "/";
    
    // Extract host from URL
    size_t protocol_end = url_.find("://");
    if (protocol_end == std::string::npos) {
        return false;
    }
    size_t host_start = protocol_end + 3;
    size_t host_end = url_.find('/', host_start);
    std::string host = (host_end != std::string::npos) ? url_.substr(host_start, host_end - host_start) : url_.substr(host_start);
    
    // Build handshake request
    std::ostringstream request;
    request << "GET " << path << " HTTP/1.1\r\n";
    request << "Host: " << host << "\r\n";
    request << "Upgrade: websocket\r\n";
    request << "Connection: Upgrade\r\n";
    request << "Sec-WebSocket-Key: " << key << "\r\n";
    request << "Sec-WebSocket-Version: 13\r\n";
    request << "\r\n";
    
    std::string request_str = request.str();
    
    // Send handshake
    if (::send(socket_fd_, request_str.c_str(), request_str.length(), MSG_NOSIGNAL) < 0) {
        return false;
    }
    
    // Read response (simplified - in production, parse properly)
    char buffer[4096];
    int received = ::recv(socket_fd_, buffer, sizeof(buffer) - 1, 0);
    if (received <= 0) {
        return false;
    }
    
    buffer[received] = '\0';
    std::string response(buffer);
    
    // Check for "101 Switching Protocols"
    if (response.find("101") == std::string::npos) {
        return false;
    }
    
    return true;
}

bool OptimizedWebSocket::send(const std::string& message) {
    if (!connected_.load() || socket_fd_ < 0) {
        return false;
    }
    
    // Create WebSocket frame
    std::vector<uint8_t> frame = create_websocket_frame(message);
    
    // Send frame
    int sent = ::send(socket_fd_, frame.data(), frame.size(), MSG_NOSIGNAL);
    if (sent < 0) {
        return false;
    }
    
    return true;
}

bool OptimizedWebSocket::read_message(std::string& message) {
    if (socket_fd_ < 0) {
        return false;
    }
    
    // Read WebSocket frame header (2 bytes minimum)
    uint8_t header[2];
    int received = recv(socket_fd_, header, 2, MSG_DONTWAIT);
    if (received != 2) {
        return false;
    }
    
    // Parse frame (simplified - assumes unmasked text frame)
    bool fin = (header[0] & 0x80) != 0;
    uint8_t opcode = header[0] & 0x0F;
    bool masked = (header[1] & 0x80) != 0;
    uint64_t payload_len = header[1] & 0x7F;
    
    if (payload_len == 126) {
        // Extended payload length (16-bit)
        uint8_t len_bytes[2];
        if (recv(socket_fd_, len_bytes, 2, MSG_DONTWAIT) != 2) {
            return false;
        }
        payload_len = (len_bytes[0] << 8) | len_bytes[1];
    } else if (payload_len == 127) {
        // Extended payload length (64-bit)
        uint8_t len_bytes[8];
        if (recv(socket_fd_, len_bytes, 8, MSG_DONTWAIT) != 8) {
            return false;
        }
        payload_len = 0;
        for (int i = 0; i < 8; i++) {
            payload_len = (payload_len << 8) | len_bytes[i];
        }
    }
    
    // Read masking key (if masked)
    uint8_t masking_key[4] = {0};
    if (masked) {
        if (recv(socket_fd_, masking_key, 4, MSG_DONTWAIT) != 4) {
            return false;
        }
    }
    
    // Read payload
    std::vector<uint8_t> payload(payload_len);
    int total_received = 0;
    while (total_received < static_cast<int>(payload_len)) {
        int received = recv(socket_fd_, payload.data() + total_received, 
                           payload_len - total_received, MSG_DONTWAIT);
        if (received <= 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::microseconds(1));
                continue;
            }
            return false;
        }
        total_received += received;
    }
    
    // Unmask payload (if masked)
    if (masked) {
        for (size_t i = 0; i < payload.size(); i++) {
            payload[i] ^= masking_key[i % 4];
        }
    }
    
    // Convert to string
    message.assign(payload.begin(), payload.end());
    
    return true;
}

std::vector<uint8_t> OptimizedWebSocket::create_websocket_frame(const std::string& message) {
    std::vector<uint8_t> frame;
    
    // FIN + opcode (text frame)
    frame.push_back(0x81);  // FIN=1, opcode=1 (text)
    
    // Payload length
    size_t len = message.length();
    if (len < 126) {
        frame.push_back(static_cast<uint8_t>(len));
    } else if (len < 65536) {
        frame.push_back(126);
        frame.push_back(static_cast<uint8_t>((len >> 8) & 0xFF));
        frame.push_back(static_cast<uint8_t>(len & 0xFF));
    } else {
        frame.push_back(127);
        for (int i = 7; i >= 0; i--) {
            frame.push_back(static_cast<uint8_t>((len >> (i * 8)) & 0xFF));
        }
    }
    
    // Payload (unmasked - server to client)
    for (char c : message) {
        frame.push_back(static_cast<uint8_t>(c));
    }
    
    return frame;
}

void OptimizedWebSocket::set_message_callback(MessageCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    message_callback_ = callback;
}

void OptimizedWebSocket::start_polling_thread() {
    if (polling_running_.load()) {
        return;
    }
    
    polling_running_.store(true);
    polling_thread_ = std::thread(&OptimizedWebSocket::polling_loop, this);
    
    // Pin WebSocket thread to Core 0 for ultra-low latency
    CPUAffinity::pin_to_core(0);
    CPUAffinity::set_realtime_priority(99);
    CPUAffinity::set_performance_mode(0);
}

void OptimizedWebSocket::stop_polling_thread() {
    if (!polling_running_.load()) {
        return;
    }
    
    polling_running_.store(false);
    if (polling_thread_.joinable()) {
        polling_thread_.join();
    }
}

void OptimizedWebSocket::polling_loop() {
    // Multi-threaded polling: Never let WebSocket sleep
    while (polling_running_.load() && connected_.load()) {
        std::string message;
        int64_t receive_start = get_timestamp_ns();
        
        if (read_message(message)) {
            int64_t receive_latency = get_timestamp_ns() - receive_start;
            
            // Update statistics
            total_messages_.fetch_add(1);
            total_bytes_.fetch_add(message.length());
            
            // Track latency
            int64_t current_max = max_latency_ns_.load();
            while (receive_latency > current_max &&
                   !max_latency_ns_.compare_exchange_weak(current_max, receive_latency)) {
                current_max = max_latency_ns_.load();
            }
            
            total_latency_ns_.fetch_add(receive_latency);
            latency_count_.fetch_add(1);
            
            // Call callback
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (message_callback_) {
                message_callback_(message, get_timestamp_ns());
            }
        } else {
            // No message available - spin-wait with minimal sleep
            std::this_thread::sleep_for(std::chrono::microseconds(1));  // 1us
        }
    }
}

int64_t OptimizedWebSocket::get_timestamp_ns() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

int64_t OptimizedWebSocket::get_avg_latency_ns() const {
    int64_t count = latency_count_.load();
    if (count == 0) {
        return 0;
    }
    return total_latency_ns_.load() / count;
}

bool OptimizedWebSocket::set_tcp_nodelay(bool enable) {
    if (socket_fd_ < 0) {
        return false;
    }
    
    int flag = enable ? 1 : 0;
    return setsockopt(socket_fd_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag)) == 0;
}

bool OptimizedWebSocket::set_socket_priority(int priority) {
    if (socket_fd_ < 0) {
        return false;
    }
    
    #ifdef __linux__
    return setsockopt(socket_fd_, SOL_SOCKET, SO_PRIORITY, &priority, sizeof(priority)) == 0;
    #else
    return false;  // Not supported on non-Linux
    #endif
}

bool OptimizedWebSocket::set_receive_buffer_size(int size) {
    if (socket_fd_ < 0) {
        return false;
    }
    
    return setsockopt(socket_fd_, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size)) == 0;
}

bool OptimizedWebSocket::set_send_buffer_size(int size) {
    if (socket_fd_ < 0) {
        return false;
    }
    
    return setsockopt(socket_fd_, SOL_SOCKET, SO_SNDBUF, &size, sizeof(size)) == 0;
}

std::string OptimizedWebSocket::base64_encode(const std::string& data) {
    // Use the helper function from anonymous namespace
    return ::base64_encode(data);
}
