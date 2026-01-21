/**
 * Phase 16: Zero-Copy Feed Handler using Shared Memory
 * Low-latency data transfer between C++ Feed Handler and Python Strategy
 * Uses Boost.Interprocess for cross-process shared memory communication
 */

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <cstring>
#include <chrono>
#include <atomic>
#include <cstdint>
#include <vector>
#include <string>

namespace bip = boost::interprocess;

// Shared memory data structure (must be POD - Plain Old Data)
struct TickData {
    char symbol[16];              // e.g., "BTCUSDT\0"
    double price;
    double bid;
    double ask;
    int64_t timestamp_ns;
    uint32_t sequence_id;         // For detecting missed ticks
    uint8_t valid;                // 1 if data is valid, 0 otherwise
    char padding[7];              // Padding to align to 64 bytes
};

// Shared memory ring buffer structure
struct SharedMemoryRing {
    static constexpr size_t RING_SIZE = 1024;  // 1024 ticks in ring buffer
    static constexpr size_t MEMORY_SIZE = sizeof(TickData) * RING_SIZE + 256; // Extra for metadata
    
    std::atomic<uint64_t> write_index;   // Next position to write
    std::atomic<uint64_t> read_index;    // Last position read by Python
    std::atomic<uint32_t> dropped_ticks; // Counter for dropped ticks
    TickData ticks[RING_SIZE];           // Ring buffer data
};

class FeedHandler {
private:
    bip::shared_memory_object shm_;
    bip::mapped_region region_;
    SharedMemoryRing* ring_;
    bip::named_mutex mutex_;
    uint64_t write_counter_;
    
public:
    FeedHandler() 
        : shm_(bip::open_or_create, "hean_feed_ring", bip::read_write)
        , mutex_(bip::open_or_create, "hean_feed_mutex")
        , write_counter_(0)
    {
        // Set size of shared memory
        shm_.truncate(SharedMemoryRing::MEMORY_SIZE);
        
        // Map the entire shared memory
        region_ = bip::mapped_region(shm_, bip::read_write);
        
        // Get pointer to ring buffer
        ring_ = static_cast<SharedMemoryRing*>(region_.get_address());
        
        // Initialize if this is the first time
        static bool initialized = false;
        if (!initialized) {
            ring_->write_index.store(0);
            ring_->read_index.store(0);
            ring_->dropped_ticks.store(0);
            std::memset(ring_->ticks, 0, sizeof(ring_->ticks));
            initialized = true;
        }
        
        // Sync write counter with current write index
        write_counter_ = ring_->write_index.load();
    }
    
    ~FeedHandler() {
        // Note: We don't remove shared memory on destruction
        // It persists until explicitly removed or system reboot
    }
    
    // Push a tick to shared memory (zero-copy write)
    bool push_tick(const char* symbol, double price, double bid, double ask, int64_t timestamp_ns) {
        // Lock for atomic operation
        bip::scoped_lock<bip::named_mutex> lock(mutex_);
        
        uint64_t next_index = write_counter_ % SharedMemoryRing::RING_SIZE;
        uint64_t read_index = ring_->read_index.load();
        
        // Check if ring buffer is full (write caught up to read)
        // Allow buffer to be almost full, but drop if truly full
        if ((write_counter_ - read_index) >= (SharedMemoryRing::RING_SIZE - 1)) {
            ring_->dropped_ticks.fetch_add(1);
            return false;  // Buffer full, tick dropped
        }
        
        // Write tick data (zero-copy, direct memory write)
        TickData* tick = &ring_->ticks[next_index];
        
        std::strncpy(tick->symbol, symbol, 15);
        tick->symbol[15] = '\0';
        tick->price = price;
        tick->bid = bid;
        tick->ask = ask;
        tick->timestamp_ns = timestamp_ns;
        tick->sequence_id = static_cast<uint32_t>(write_counter_);
        tick->valid = 1;
        
        // Memory fence to ensure writes are visible
        std::atomic_thread_fence(std::memory_order_release);
        
        // Update write index (atomic, lock-free)
        write_counter_++;
        ring_->write_index.store(write_counter_);
        
        return true;
    }
    
    // Get statistics
    uint32_t get_dropped_ticks() const {
        return ring_->dropped_ticks.load();
    }
    
    uint64_t get_write_index() const {
        return ring_->write_index.load();
    }
    
    uint64_t get_read_index() const {
        return ring_->read_index.load();
    }
    
    // Cleanup shared memory (call explicitly when shutting down)
    static void cleanup() {
        bip::shared_memory_object::remove("hean_feed_ring");
        bip::named_mutex::remove("hean_feed_mutex");
    }
};

// C interface for Python bindings
extern "C" {
    static FeedHandler* g_feed_handler = nullptr;
    
    void feed_handler_init() {
        if (g_feed_handler == nullptr) {
            g_feed_handler = new FeedHandler();
        }
    }
    
    int feed_handler_push_tick(const char* symbol, double price, double bid, double ask, int64_t timestamp_ns) {
        if (g_feed_handler == nullptr) {
            feed_handler_init();
        }
        return g_feed_handler->push_tick(symbol, price, bid, ask, timestamp_ns) ? 1 : 0;
    }
    
    uint32_t feed_handler_get_dropped_ticks() {
        if (g_feed_handler == nullptr) return 0;
        return g_feed_handler->get_dropped_ticks();
    }
    
    uint64_t feed_handler_get_write_index() {
        if (g_feed_handler == nullptr) return 0;
        return g_feed_handler->get_write_index();
    }
    
    void feed_handler_cleanup() {
        if (g_feed_handler != nullptr) {
            delete g_feed_handler;
            g_feed_handler = nullptr;
        }
        FeedHandler::cleanup();
    }
}
