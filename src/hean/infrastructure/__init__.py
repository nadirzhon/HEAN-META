"""
Infrastructure module for HEAN trading system.

Provides:
- Redis caching
- Event streaming
- Analytics database
- Performance optimization
"""

from hean.infrastructure.cache import FeatureCache, CacheConfig

__all__ = ["FeatureCache", "CacheConfig"]
