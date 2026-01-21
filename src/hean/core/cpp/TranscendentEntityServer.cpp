/**
 * Transcendent Entity Interface Server Implementation
 * High-performance uWebSockets server
 */

#include "TranscendentEntityServer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <cstring>

// uWebSockets
#define ASIO_STANDALONE
#include "uWebSockets.h"

TranscendentEntityServer::TranscendentEntityServer(int port)
    : port_(port)
    , html_path_("web/transcendent_entity.html")
    , running_(false)
    , should_stop_(false)
    , listen_socket_(nullptr)
{
    // Initialize data structure
    std::memset(&current_data_, 0, sizeof(current_data_));
    current_data_.balance = 300.0;  // Default $300
    current_data_.volatility = 0.5;
    current_data_.confidence = 0.7;
    std::strncpy(current_data_.logic_state, "INITIALIZING", sizeof(current_data_.logic_state) - 1);
}

TranscendentEntityServer::~TranscendentEntityServer() {
    stop();
}

bool TranscendentEntityServer::start() {
    if (running_.load()) {
        return false;
    }
    
    should_stop_ = false;
    
    // Start server in separate thread
    server_thread_ = std::thread(&TranscendentEntityServer::server_loop, this);
    
    // Wait a bit for server to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    return running_.load();
}

void TranscendentEntityServer::stop() {
    if (!running_.load()) {
        return;
    }
    
    should_stop_ = true;
    
    // Close all WebSocket connections
    {
        std::lock_guard<std::mutex> lock(ws_mutex_);
        for (auto& pair : ws_connections_) {
            // Close will be handled by uWS cleanup
        }
        ws_connections_.clear();
    }
    
    // Stop uWebSockets server
    if (listen_socket_) {
        us_listen_socket_close(0, listen_socket_);
        listen_socket_ = nullptr;
    }
    
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    
    running_ = false;
}

void TranscendentEntityServer::update_data(const TranscendentEntityData& data) {
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        current_data_ = data;
        current_data_.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    
    // Broadcast to all connected WebSocket clients
    broadcast_data();
}

TranscendentEntityData TranscendentEntityServer::get_data_snapshot() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return current_data_;
}

void TranscendentEntityServer::server_loop() {
    try {
        uWS::App::WebSocketBehavior<false> behavior;
        behavior.compression = uWS::SHARED_COMPRESSOR;
        behavior.maxPayloadLength = 16 * 1024 * 1024;  // 16MB
        behavior.idleTimeout = 16;
        behavior.maxBackpressure = 1 * 1024 * 1024;  // 1MB
        
        behavior.open = [this](uWS::WebSocket<false>* ws) {
            handle_websocket_open(ws);
        };
        
        behavior.message = [this](uWS::WebSocket<false>* ws, std::string_view message, uWS::OpCode opcode) {
            handle_websocket_message(ws, message, opcode);
        };
        
        behavior.close = [this](uWS::WebSocket<false>* ws, int code, std::string_view message) {
            handle_websocket_close(ws, code, message);
        };
        
        app_ = std::make_unique<uWS::App>();
        
        app_->ws<false>("/*", std::move(behavior))
            .get("/*", [this](uWS::HttpResponse<false>* res, uWS::HttpRequest<false>* req) {
                handle_http_request(res, req);
            })
            .listen(port_, [this](us_listen_socket_t* listen_socket) {
                if (listen_socket) {
                    listen_socket_ = listen_socket;
                    running_ = true;
                    std::cout << "[TranscendentEntity] Server listening on port " << port_ << std::endl;
                } else {
                    std::cerr << "[TranscendentEntity] Failed to start server on port " << port_ << std::endl;
                }
            })
            .run();
            
    } catch (const std::exception& e) {
        std::cerr << "[TranscendentEntity] Server error: " << e.what() << std::endl;
        running_ = false;
    }
}

void TranscendentEntityServer::handle_http_request(uWS::HttpResponse<false>* res, uWS::HttpRequest<false>* req) {
    std::string path(req->getUrl().data(), req->getUrl().length());
    
    if (path == "/" || path == "/transcendent_entity.html") {
        // Serve HTML dashboard
        std::string html = load_html_content();
        res->writeStatus("200 OK")
           ->writeHeader("Content-Type", "text/html; charset=utf-8")
           ->end(html);
    } else {
        // 404
        res->writeStatus("404 Not Found")
           ->end("Not Found");
    }
}

void TranscendentEntityServer::handle_websocket_open(uWS::WebSocket<false>* ws) {
    std::lock_guard<std::mutex> lock(ws_mutex_);
    ws_connections_[ws] = ws;
    
    // Send initial data snapshot
    TranscendentEntityData data = get_data_snapshot();
    std::string binary_msg = encode_binary_message(data);
    ws->send(binary_msg, uWS::BINARY);
}

void TranscendentEntityServer::handle_websocket_message(uWS::WebSocket<false>* ws, std::string_view message, uWS::OpCode opcode) {
    // Echo ping messages or handle commands
    if (opcode == uWS::TEXT) {
        std::string msg(message.data(), message.length());
        if (msg == "ping") {
            ws->send("pong", uWS::TEXT);
        }
    }
}

void TranscendentEntityServer::handle_websocket_close(uWS::WebSocket<false>* ws, int code, std::string_view message) {
    std::lock_guard<std::mutex> lock(ws_mutex_);
    ws_connections_.erase(ws);
}

void TranscendentEntityServer::broadcast_data() {
    std::string binary_msg = encode_binary_message(get_data_snapshot());
    
    std::lock_guard<std::mutex> lock(ws_mutex_);
    for (auto& pair : ws_connections_) {
        if (pair.second) {
            pair.second->send(binary_msg, uWS::BINARY);
        }
    }
}

std::string TranscendentEntityServer::load_html_content() const {
    std::ifstream file(html_path_);
    if (!file.is_open()) {
        // Return default HTML if file not found
        return R"(<!DOCTYPE html>
<html><head><title>Transcendent Entity</title></head>
<body><h1>Dashboard file not found</h1><p>Expected at: )" + html_path_ + R"(</p></body></html>)";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::string TranscendentEntityServer::encode_binary_message(const TranscendentEntityData& data) const {
    // Simple binary encoding: just serialize the struct
    // Note: This assumes same endianness and struct layout
    std::string msg(sizeof(TranscendentEntityData), '\0');
    std::memcpy(const_cast<char*>(msg.data()), &data, sizeof(TranscendentEntityData));
    return msg;
}
