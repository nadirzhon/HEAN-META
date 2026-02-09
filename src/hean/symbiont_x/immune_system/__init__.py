"""
Immune System - Иммунная система защиты

Три уровня защиты:
1. Constitution - неизменяемые правила риска
2. Reflexes - мгновенные реакции на опасность
3. Circuit Breakers - аварийные стопы
"""

from .circuit_breakers import CircuitBreakerState, CircuitBreakerSystem
from .constitution import ConstitutionViolation, RiskConstitution
from .reflexes import ReflexSystem, ReflexTrigger

__all__ = [
    'RiskConstitution',
    'ConstitutionViolation',
    'ReflexSystem',
    'ReflexTrigger',
    'CircuitBreakerSystem',
    'CircuitBreakerState',
]
