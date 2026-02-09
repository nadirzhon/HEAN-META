"""
Adversarial Digital Twin - Злой экзаменатор

Три мира тестирования:
- Replay World (исторические данные)
- Paper World (реальное время, виртуальный счёт)
- Micro-Real World (реальные деньги, micro позиции)
"""

from .stress_tests import StressTestSuite
from .survival_score import SurvivalScoreCalculator
from .test_worlds import MicroRealWorld, PaperWorld, ReplayWorld, TestWorld

__all__ = [
    'TestWorld',
    'ReplayWorld',
    'PaperWorld',
    'MicroRealWorld',
    'StressTestSuite',
    'SurvivalScoreCalculator',
]
