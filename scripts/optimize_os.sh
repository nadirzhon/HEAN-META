#!/bin/bash
#
# Phase 16: CPU Affinity & Isolation Setup Script
# Pins C++ process to Core 0 and Python process to Core 1-3, disabling CPU throttling.
#
# This script must be run as root/sudo to modify CPU settings and set process affinity.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root/sudo to modify CPU settings"
        echo "Usage: sudo $0 [cpp_pid] [python_pid]"
        exit 1
    fi
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="darwin"
    else
        log_error "Unsupported OS: $OSTYPE"
        exit 1
    fi
    log_info "Detected OS: $OS"
}

# Disable CPU throttling for specific cores (Linux)
disable_cpu_throttling_linux() {
    local core=$1
    log_info "Disabling CPU throttling for core $core (Linux)"
    
    # Check if CPU governor exists
    if [ -f "/sys/devices/system/cpu/cpu${core}/cpufreq/scaling_governor" ]; then
        # Set governor to performance mode (disables frequency scaling)
        echo "performance" > "/sys/devices/system/cpu/cpu${core}/cpufreq/scaling_governor"
        log_info "Set CPU $core governor to performance mode"
    else
        log_warn "CPU $core scaling governor not found (may not support frequency scaling)"
    fi
    
    # Disable CPU idle states (deep sleep modes) for lower latency
    if [ -d "/sys/devices/system/cpu/cpu${core}/cpuidle" ]; then
        for state_dir in /sys/devices/system/cpu/cpu${core}/cpuidle/state*; do
            if [ -f "$state_dir/disable" ]; then
                echo 1 > "$state_dir/disable" 2>/dev/null || true
            fi
        done
        log_info "Disabled deep sleep states for CPU $core"
    fi
}

# Disable CPU throttling for specific cores (macOS)
disable_cpu_throttling_darwin() {
    local core=$1
    log_info "Disabling CPU throttling for core $core (macOS)"
    
    # macOS uses Intel Power Gadget or pmset
    # Note: macOS throttling control is more limited than Linux
    # We can use pmset to disable idle sleep and reduce CPU throttling
    
    # Disable idle sleep (prevents CPU from idling)
    pmset -a disablesleep 1 2>/dev/null || log_warn "Failed to disable idle sleep (pmset)"
    
    # Set performance mode (if available)
    # Note: This may require Intel Power Gadget or similar tools
    log_warn "macOS CPU throttling control is limited. Consider using Intel Power Gadget for advanced control."
}

# Pin process to specific CPU core(s)
pin_process_to_core() {
    local pid=$1
    local cores=$2
    
    if [ -z "$pid" ] || [ ! -d "/proc/$pid" ]; then
        log_error "Process PID $pid not found or not running"
        return 1
    fi
    
    log_info "Pinning process $pid to cores: $cores"
    
    if [ "$OS" == "linux" ]; then
        # Use taskset to set CPU affinity (Linux)
        if command -v taskset >/dev/null 2>&1; then
            taskset -cp "$cores" "$pid" >/dev/null 2>&1
            if [ $? -eq 0 ]; then
                log_info "Successfully pinned process $pid to cores $cores"
            else
                log_error "Failed to pin process $pid to cores $cores"
                return 1
            fi
        else
            log_error "taskset command not found. Install with: apt-get install util-linux"
            return 1
        fi
    elif [ "$OS" == "darwin" ]; then
        # macOS uses caffeinate or renice (affinity control is limited)
        # For process affinity on macOS, we need to use launchd or dtrace
        log_warn "macOS process affinity is limited. Using renice to increase priority."
        renice -n -20 -p "$pid" 2>/dev/null || log_warn "Failed to renice process $pid"
        
        # Note: For true CPU affinity on macOS, consider using dtrace or launchd
        # This is a simplified version
        log_info "Process $pid priority increased (macOS affinity control is limited)"
    fi
    
    return 0
}

# Setup isolcpus (Linux only - isolate CPUs from kernel scheduler)
setup_cpu_isolation_linux() {
    log_info "Setting up CPU isolation (Linux)"
    
    # Check if isolcpus is already set in kernel command line
    if [ -f /proc/cmdline ]; then
        CMDLINE=$(cat /proc/cmdline)
        if echo "$CMDLINE" | grep -q "isolcpus"; then
            log_info "isolcpus already configured in kernel command line"
            return 0
        fi
    fi
    
    log_warn "CPU isolation requires kernel boot parameter 'isolcpus'"
    log_warn "To fully isolate cores 0-3, add to GRUB boot parameters:"
    log_warn "  isolcpus=0,1,2,3 nohz_full=0,1,2,3 rcu_nocbs=0,1,2,3"
    log_warn "Then reboot the system."
    log_warn ""
    log_warn "For now, we'll set CPU governor to performance mode instead."
}

# Set process priority and scheduling
set_process_scheduling() {
    local pid=$1
    local priority_class=$2  # "realtime" or "high"
    
    if [ -z "$pid" ] || [ ! -d "/proc/$pid" ]; then
        log_error "Process PID $pid not found"
        return 1
    fi
    
    log_info "Setting process $pid scheduling class: $priority_class"
    
    if [ "$OS" == "linux" ]; then
        # Use chrt to set scheduling policy (Linux)
        if command -v chrt >/dev/null 2>&1; then
            if [ "$priority_class" == "realtime" ]; then
                # SCHED_FIFO with priority 99 (highest)
                chrt -f -p 99 "$pid" 2>/dev/null || {
                    log_warn "Failed to set realtime scheduling (may require CAP_SYS_NICE)"
                    # Fallback to SCHED_RR with priority 50
                    chrt -r -p 50 "$pid" 2>/dev/null || log_warn "Failed to set RR scheduling"
                }
            else
                # SCHED_RR (round-robin) with priority 50
                chrt -r -p 50 "$pid" 2>/dev/null || log_warn "Failed to set RR scheduling"
            fi
        else
            log_error "chrt command not found. Install with: apt-get install util-linux"
            return 1
        fi
    elif [ "$OS" == "darwin" ]; then
        # macOS uses renice (limited priority control)
        if [ "$priority_class" == "realtime" ]; then
            renice -n -20 -p "$pid" 2>/dev/null || log_warn "Failed to renice process $pid"
        else
            renice -n -10 -p "$pid" 2>/dev/null || log_warn "Failed to renice process $pid"
        fi
    fi
}

# Main optimization function
optimize_for_trading() {
    local cpp_pid=$1
    local python_pid=$2
    
    log_info "Starting Phase 16: CPU Affinity & Isolation Setup"
    log_info "C++ process PID: ${cpp_pid:-'not specified'}"
    log_info "Python process PID: ${python_pid:-'not specified'}"
    log_info ""
    
    # Disable CPU throttling for cores 0-3
    log_info "Disabling CPU throttling for cores 0-3..."
    for core in 0 1 2 3; do
        if [ "$OS" == "linux" ]; then
            disable_cpu_throttling_linux "$core"
        elif [ "$OS" == "darwin" ]; then
            disable_cpu_throttling_darwin "$core"
        fi
    done
    log_info ""
    
    # Setup CPU isolation (Linux only)
    if [ "$OS" == "linux" ]; then
        setup_cpu_isolation_linux
        log_info ""
    fi
    
    # Pin C++ process to Core 0
    if [ -n "$cpp_pid" ]; then
        log_info "Optimizing C++ process (PID: $cpp_pid)..."
        pin_process_to_core "$cpp_pid" "0"
        set_process_scheduling "$cpp_pid" "realtime"
        log_info ""
    fi
    
    # Pin Python process to Cores 1-3
    if [ -n "$python_pid" ]; then
        log_info "Optimizing Python process (PID: $python_pid)..."
        pin_process_to_core "$python_pid" "1-3"
        set_process_scheduling "$python_pid" "high"
        log_info ""
    fi
    
    log_info "Phase 16: CPU optimization complete!"
    log_info ""
    log_info "Summary:"
    log_info "  - CPU throttling disabled for cores 0-3"
    if [ -n "$cpp_pid" ]; then
        log_info "  - C++ process pinned to Core 0 with realtime scheduling"
    fi
    if [ -n "$python_pid" ]; then
        log_info "  - Python process pinned to Cores 1-3 with high priority"
    fi
    log_info ""
    log_warn "Note: Some optimizations require kernel boot parameters and system reboot."
}

# Main execution
main() {
    check_root
    detect_os
    
    CPP_PID=$1
    PYTHON_PID=$2
    
    if [ -z "$CPP_PID" ] && [ -z "$PYTHON_PID" ]; then
        log_warn "No PIDs provided. Searching for HEAN processes..."
        
        # Try to find HEAN processes
        if [ "$OS" == "linux" ]; then
            # Look for Python process running hean.main
            PYTHON_PID=$(pgrep -f "python.*hean.main" | head -1)
            # Look for C++ process (if it has a recognizable name)
            CPP_PID=$(pgrep -f "hean.*graph_engine" | head -1 || true)
        elif [ "$OS" == "darwin" ]; then
            PYTHON_PID=$(pgrep -f "python.*hean.main" | head -1)
            CPP_PID=$(pgrep -f "hean.*graph_engine" | head -1 || true)
        fi
        
        if [ -z "$PYTHON_PID" ]; then
            log_error "Could not find Python process. Please provide PIDs manually:"
            log_error "  Usage: sudo $0 [cpp_pid] [python_pid]"
            exit 1
        fi
        
        log_info "Found Python process: $PYTHON_PID"
        if [ -n "$CPP_PID" ]; then
            log_info "Found C++ process: $CPP_PID"
        fi
    fi
    
    optimize_for_trading "$CPP_PID" "$PYTHON_PID"
}

# Run main function
main "$@"
