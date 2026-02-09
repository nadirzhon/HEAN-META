"""C++ core modules - high-performance implementations.

This package contains C++ modules built with nanobind for maximum performance:
- indicators_cpp: Ultra-fast technical indicators (50-100x faster than pandas)
- order_router_cpp: Sub-microsecond order routing decisions

If C++ modules are not available, the system will automatically fall back
to pure Python implementations with a warning.
"""

import warnings
from typing import Any

# Try to import C++ modules, fall back to Python if not available
try:
    from . import indicators_cpp  # type: ignore

    INDICATORS_CPP_AVAILABLE = True
except ImportError:
    INDICATORS_CPP_AVAILABLE = False
    indicators_cpp = None
    warnings.warn(
        "C++ indicators module not available. Falling back to pure Python implementation. "
        "Performance will be reduced (50-100x slower). "
        "To build C++ modules, run: ./scripts/build_cpp_modules.sh",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from . import order_router_cpp  # type: ignore

    ORDER_ROUTER_CPP_AVAILABLE = True
except ImportError:
    ORDER_ROUTER_CPP_AVAILABLE = False
    order_router_cpp = None
    warnings.warn(
        "C++ order_router module not available. Falling back to pure Python implementation. "
        "Latency will be increased (microsecond → millisecond). "
        "To build C++ modules, run: ./scripts/build_cpp_modules.sh",
        RuntimeWarning,
        stacklevel=2,
    )


def get_cpp_status() -> dict[str, Any]:
    """Get status of C++ modules for diagnostics.

    Returns:
        dict with module availability and performance hints
    """
    return {
        "indicators_cpp_available": INDICATORS_CPP_AVAILABLE,
        "order_router_cpp_available": ORDER_ROUTER_CPP_AVAILABLE,
        "performance_hint": (
            "All C++ modules loaded - optimal performance"
            if INDICATORS_CPP_AVAILABLE and ORDER_ROUTER_CPP_AVAILABLE
            else "Some C++ modules missing - using Python fallback (slower)"
        ),
        "build_instructions": "Run: ./scripts/build_cpp_modules.sh",
    }


__all__ = [
    "indicators_cpp",
    "order_router_cpp",
    "INDICATORS_CPP_AVAILABLE",
    "ORDER_ROUTER_CPP_AVAILABLE",
    "get_cpp_status",
]
