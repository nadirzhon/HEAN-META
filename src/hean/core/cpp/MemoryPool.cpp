/**
 * Ultra-Low Latency Memory Pool Implementation
 */

#include "MemoryPool.h"

#ifdef __linux__
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <pthread.h>
#endif

// Global memory pool instances
FixedMemoryPool<64, 1000> g_pool_64;
FixedMemoryPool<256, 500> g_pool_256;
FixedMemoryPool<1024, 100> g_pool_1024;

namespace CPUAffinity {
    bool pin_to_core(int core_id) {
#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        
        pthread_t thread = pthread_self();
        int result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
        return result == 0;
#elif defined(__APPLE__)
        // macOS uses thread_affinity_policy (less reliable than Linux)
        thread_affinity_policy_data_t policy;
        policy.affinity_tag = core_id;
        
        thread_port_t thread = pthread_mach_thread_np(pthread_self());
        kern_return_t result = thread_policy_set(
            thread,
            THREAD_AFFINITY_POLICY,
            (thread_policy_t)&policy,
            THREAD_AFFINITY_POLICY_COUNT
        );
        return result == KERN_SUCCESS;
#else
        return false; // Unsupported platform
#endif
    }
    
    bool set_realtime_priority(int priority) {
#ifdef __linux__
        struct sched_param param;
        param.sched_priority = priority;
        int result = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
        return result == 0;
#elif defined(__APPLE__)
        // macOS uses thread_policy_set with THREAD_PRECEDENCE_POLICY
        thread_precedence_policy_data_t policy;
        policy.importance = priority;
        
        thread_port_t thread = pthread_mach_thread_np(pthread_self());
        kern_return_t result = thread_policy_set(
            thread,
            THREAD_PRECEDENCE_POLICY,
            (thread_policy_t)&policy,
            THREAD_PRECEDENCE_POLICY_COUNT
        );
        return result == KERN_SUCCESS;
#else
        return false;
#endif
    }
    
    bool set_performance_mode(int core_id) {
#ifdef __linux__
        // Set CPU governor to performance mode
        char path[256];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", core_id);
        
        int fd = open(path, O_WRONLY);
        if (fd < 0) {
            return false; // CPU scaling not available
        }
        
        const char* governor = "performance";
        ssize_t written = write(fd, governor, strlen(governor));
        close(fd);
        
        return written > 0;
#else
        // Not supported on macOS (would need Intel Power Gadget)
        return false;
#endif
    }
}
