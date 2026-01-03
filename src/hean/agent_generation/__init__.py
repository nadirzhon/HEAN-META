"""Agent generation module for creating trading agents using LLM prompts."""

from hean.agent_generation.capital_optimizer import CapitalOptimizer
from hean.agent_generation.catalyst import ImprovementCatalyst
from hean.agent_generation.generator import AgentGenerator
from hean.agent_generation.parameter_optimizer import ParameterOptimizer
from hean.agent_generation.prompts import (
    PROMPT_ANALYTICAL,
    PROMPT_CREATIVE,
    PROMPT_EVALUATION,
    PROMPT_EVOLUTION,
    PROMPT_HYBRID,
    PROMPT_INITIAL,
    PROMPT_MARKET_CONDITIONS,
    PROMPT_MUTATION,
    PROMPT_PROBLEM_FOCUSED,
    SYSTEM_PROMPT,
    get_prompt,
)
from hean.agent_generation.report_generator import ReportGenerator

__all__ = [
    "AgentGenerator",
    "ImprovementCatalyst",
    "ParameterOptimizer",
    "CapitalOptimizer",
    "ReportGenerator",
    "SYSTEM_PROMPT",
    "PROMPT_INITIAL",
    "PROMPT_EVOLUTION",
    "PROMPT_ANALYTICAL",
    "PROMPT_MUTATION",
    "PROMPT_MARKET_CONDITIONS",
    "PROMPT_HYBRID",
    "PROMPT_PROBLEM_FOCUSED",
    "PROMPT_EVALUATION",
    "PROMPT_CREATIVE",
    "get_prompt",
]
