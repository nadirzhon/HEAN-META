"""System-level integration components — registry, health, state, error analysis."""

from .component_registry import ComponentRegistry, get_component_registry, set_component_registry
from .error_analyzer import ErrorAnalyzer, ErrorContext, FixSuggestion, get_error_analyzer
from .health_monitor import HealthMonitor, ModuleHealth, ModuleStatus, get_health_monitor
from .redis_state import RedisStateManager, get_redis_state_manager

__all__ = [
    "ComponentRegistry",
    "ErrorAnalyzer",
    "ErrorContext",
    "FixSuggestion",
    "HealthMonitor",
    "ModuleHealth",
    "ModuleStatus",
    "RedisStateManager",
    "get_component_registry",
    "get_error_analyzer",
    "get_health_monitor",
    "get_redis_state_manager",
    "set_component_registry",
]
