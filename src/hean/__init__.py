"""HEAN — Event-driven crypto trading system for Bybit."""

__all__ = ["EventBus", "HEANSettings", "__version__"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "EventBus":
        from .core.bus import EventBus

        return EventBus
    if name == "HEANSettings":
        from .config import HEANSettings

        return HEANSettings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
