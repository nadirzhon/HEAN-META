/**
 * Ultra-Low Latency Memory Pool System
 * Zero-allocation hot-path execution
 */

#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

// Fixed-size memory pool for hot-path allocations
template <size_t BlockSize, size_t BlockCount>
class FixedMemoryPool {
public:
    FixedMemoryPool() : free_blocks_(BlockCount), next_free_(0) {
        // Initialize free list: each block points to next
        for (size_t i = 0; i < BlockCount - 1; ++i) {
            *reinterpret_cast<size_t*>(&blocks_[i * BlockSize]) = i + 1;
        }
        *reinterpret_cast<size_t*>(&blocks_[(BlockCount - 1) * BlockSize]) = BlockCount; // End marker
    }
    
    void* allocate() {
        if (next_free_ >= BlockCount) {
            return nullptr; // Pool exhausted
        }
        
        void* block = &blocks_[next_free_ * BlockSize];
        next_free_ = *reinterpret_cast<size_t*>(block);
        return block;
    }
    
    void deallocate(void* ptr) {
        if (ptr < blocks_.data() || ptr >= blocks_.data() + blocks_.size()) {
            return; // Invalid pointer
        }
        
        size_t block_idx = (reinterpret_cast<uint8_t*>(ptr) - blocks_.data()) / BlockSize;
        *reinterpret_cast<size_t*>(ptr) = next_free_;
        next_free_ = block_idx;
    }
    
    void reset() {
        next_free_ = 0;
        for (size_t i = 0; i < BlockCount - 1; ++i) {
            *reinterpret_cast<size_t*>(&blocks_[i * BlockSize]) = i + 1;
        }
        *reinterpret_cast<size_t*>(&blocks_[(BlockCount - 1) * BlockSize]) = BlockCount;
    }
    
private:
    std::array<uint8_t, BlockSize * BlockCount> blocks_;
    std::atomic<size_t> next_free_;
    std::atomic<size_t> free_blocks_;
};

// Global memory pools for common sizes
extern FixedMemoryPool<64, 1000> g_pool_64;      // Small allocations
extern FixedMemoryPool<256, 500> g_pool_256;     // Medium allocations
extern FixedMemoryPool<1024, 100> g_pool_1024;   // Large allocations

// CPU Affinity utilities
namespace CPUAffinity {
    /**
     * Pin current thread to specific CPU core
     * @param core_id: CPU core ID (0-indexed)
     * @return true if successful
     */
    bool pin_to_core(int core_id);
    
    /**
     * Set thread to realtime priority (SCHED_FIFO)
     * @param priority: Priority level (1-99, higher = higher priority)
     * @return true if successful
     */
    bool set_realtime_priority(int priority = 99);
    
    /**
     * Disable CPU frequency scaling (performance mode)
     * @param core_id: CPU core ID
     * @return true if successful
     */
    bool set_performance_mode(int core_id);
}

#endif // MEMORY_POOL_H
