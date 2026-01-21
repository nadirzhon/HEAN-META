/**
 * Lock-Free Ring Buffer for Ultra-Low Latency Inter-Thread Communication
 * 
 * Single-Producer-Single-Consumer (SPSC) lock-free ring buffer
 * Optimized for sub-50 microsecond latency
 */

#ifndef LOCK_FREE_RING_BUFFER_H
#define LOCK_FREE_RING_BUFFER_H

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>

// Ensure power-of-2 size for fast modulo
template<typename T, size_t Size>
class LockFreeRingBuffer {
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    
private:
    alignas(64) std::atomic<size_t> write_pos_{0};  // Cache line aligned
    alignas(64) std::atomic<size_t> read_pos_{0};   // Cache line aligned
    
    T buffer_[Size];
    
    // Fast modulo using bit mask (only works for power-of-2)
    static constexpr size_t mask_ = Size - 1;
    
    // Force inline for hot-path operations
    __attribute__((always_inline)) 
    static inline size_t next_pos(size_t pos) noexcept {
        return (pos + 1) & mask_;
    }
    
    __attribute__((always_inline))
    static inline size_t pos_mod(size_t pos) noexcept {
        return pos & mask_;
    }
    
public:
    LockFreeRingBuffer() noexcept {
        // Zero-initialize buffer
        std::memset(buffer_, 0, sizeof(buffer_));
        write_pos_.store(0, std::memory_order_relaxed);
        read_pos_.store(0, std::memory_order_relaxed);
    }
    
    ~LockFreeRingBuffer() = default;
    
    // Non-copyable, non-movable
    LockFreeRingBuffer(const LockFreeRingBuffer&) = delete;
    LockFreeRingBuffer& operator=(const LockFreeRingBuffer&) = delete;
    
    /**
     * Try to push an item (non-blocking)
     * @return true if successful, false if buffer is full
     */
    __attribute__((always_inline))
    bool try_push(const T& item) noexcept {
        size_t current_write = write_pos_.load(std::memory_order_relaxed);
        size_t next_write = next_pos(current_write);
        size_t current_read = read_pos_.load(std::memory_order_acquire);
        
        // Check if buffer is full
        if (next_write == current_read) {
            return false;  // Full
        }
        
        // Copy item to buffer
        buffer_[pos_mod(current_write)] = item;
        
        // Update write position (release semantic ensures item is visible)
        write_pos_.store(next_write, std::memory_order_release);
        
        return true;
    }
    
    /**
     * Try to pop an item (non-blocking)
     * @return true if successful, false if buffer is empty
     */
    __attribute__((always_inline))
    bool try_pop(T& item) noexcept {
        size_t current_read = read_pos_.load(std::memory_order_relaxed);
        size_t current_write = write_pos_.load(std::memory_order_acquire);
        
        // Check if buffer is empty
        if (current_read == current_write) {
            return false;  // Empty
        }
        
        // Copy item from buffer
        item = buffer_[pos_mod(current_read)];
        
        // Update read position (release semantic ensures we've read the item)
        read_pos_.store(next_pos(current_read), std::memory_order_release);
        
        return true;
    }
    
    /**
     * Check if buffer is empty (non-blocking, approximate)
     */
    __attribute__((always_inline))
    bool empty() const noexcept {
        size_t r = read_pos_.load(std::memory_order_relaxed);
        size_t w = write_pos_.load(std::memory_order_relaxed);
        return r == w;
    }
    
    /**
     * Check if buffer is full (non-blocking, approximate)
     */
    __attribute__((always_inline))
    bool full() const noexcept {
        size_t r = read_pos_.load(std::memory_order_relaxed);
        size_t w = write_pos_.load(std::memory_order_relaxed);
        return next_pos(w) == r;
    }
    
    /**
     * Get approximate size (non-blocking)
     */
    __attribute__((always_inline))
    size_t size() const noexcept {
        size_t r = read_pos_.load(std::memory_order_relaxed);
        size_t w = write_pos_.load(std::memory_order_relaxed);
        
        if (w >= r) {
            return w - r;
        } else {
            return Size - (r - w);
        }
    }
    
    /**
     * Get capacity
     */
    static constexpr size_t capacity() noexcept {
        return Size - 1;  // One slot reserved for full/empty distinction
    }
    
    /**
     * Clear buffer (not thread-safe, use with caution)
     */
    void clear() noexcept {
        read_pos_.store(write_pos_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
};

#endif // LOCK_FREE_RING_BUFFER_H