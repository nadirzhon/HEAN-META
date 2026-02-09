"""Prompts for agent generation."""

SYSTEM_PROMPT = """You are an expert Python developer specializing in algorithmic trading systems.
Generate clean, well-structured Python code for trading agents that:
- Follow the HEAN trading system architecture
- Use proper type hints
- Include error handling
- Follow best practices for algorithmic trading
- Are production-ready and tested"""

PROMPT_INITIAL = """Generate a new trading agent Python class that:
- Inherits from BaseStrategy
- Implements on_tick method
- Uses proper risk management
- Includes proper position sizing
- Has clear entry/exit logic"""

PROMPT_EVOLUTION = """Generate an evolved trading agent based on best/worst agents and market conditions."""

PROMPT_MUTATION = """Improve an existing trading agent code by fixing issues and optimizing performance."""

PROMPT_MARKET_CONDITIONS = """Generate a trading agent optimized for specific market conditions (volatility, trend, etc.)."""

PROMPT_HYBRID = """Generate a hybrid trading agent combining multiple strategies."""

PROMPT_PROBLEM_FOCUSED = """Generate a trading agent to solve a specific problem."""

PROMPT_EVALUATION = """Evaluate and improve the trading agent based on performance metrics."""

PROMPT_CREATIVE = """Generate a creative, innovative trading agent."""

PROMPT_ANALYTICAL = """Generate an analytical trading agent focused on data analysis."""


def get_prompt(prompt_type: str, **kwargs: dict) -> str:
    """Get prompt template by type.

    Args:
        prompt_type: Type of prompt (initial, evolution, mutation, etc.)
        **kwargs: Variables for prompt formatting

    Returns:
        Formatted prompt string
    """
    if prompt_type == "initial":
        return _get_initial_prompt(**kwargs)
    elif prompt_type == "evolution":
        return _get_evolution_prompt(**kwargs)
    elif prompt_type == "mutation":
        return _get_mutation_prompt(**kwargs)
    elif prompt_type == "market_conditions":
        return _get_market_conditions_prompt(**kwargs)
    elif prompt_type == "hybrid":
        return _get_hybrid_prompt(**kwargs)
    elif prompt_type == "problem_focused":
        return _get_problem_focused_prompt(**kwargs)
    elif prompt_type == "evaluation":
        return _get_evaluation_prompt(**kwargs)
    elif prompt_type == "creative":
        return _get_creative_prompt(**kwargs)
    elif prompt_type == "analytical":
        return _get_analytical_prompt(**kwargs)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


def _get_initial_prompt(**kwargs: dict) -> str:
    """Generate initial agent prompt."""
    return """Generate a new trading agent Python class that:
- Inherits from BaseStrategy
- Implements on_tick method
- Uses proper risk management
- Includes proper position sizing
- Has clear entry/exit logic"""


def _get_evolution_prompt(**kwargs: dict) -> str:
    """Generate evolution prompt."""
    best = kwargs.get("best_agents_info", "")
    worst = kwargs.get("worst_agents_info", "")
    market = kwargs.get("market_conditions", "")
    metrics = kwargs.get("performance_metrics", "")

    return f"""Generate an evolved trading agent based on:
Best agents: {best}
Worst agents: {worst}
Market conditions: {market}
Performance metrics: {metrics}

Create an improved agent that combines the best features."""


def _get_mutation_prompt(**kwargs: dict) -> str:
    """Generate mutation prompt."""
    code = kwargs.get("agent_code", "")
    issues = kwargs.get("issues", "")

    return f"""Improve this trading agent code:
{code}

Issues to fix: {issues}

Generate an improved version."""


def _get_market_conditions_prompt(**kwargs: dict) -> str:
    """Generate market conditions prompt."""
    vol = kwargs.get("volatility_level", "")
    trend = kwargs.get("trend_direction", "")

    return f"""Generate a trading agent optimized for:
Volatility: {vol}
Trend: {trend}

Create a specialized agent for these conditions."""


def _get_hybrid_prompt(**kwargs: dict) -> str:
    """Generate hybrid prompt."""
    return """Generate a hybrid trading agent combining multiple strategies."""


def _get_problem_focused_prompt(**kwargs: dict) -> str:
    """Generate problem-focused prompt."""
    problem = kwargs.get("problem", "")
    return f"""Generate a trading agent to solve this problem: {problem}"""


def _get_evaluation_prompt(**kwargs: dict) -> str:
    """Generate evaluation prompt."""
    return """Evaluate and improve the trading agent."""


def _get_creative_prompt(**kwargs: dict) -> str:
    """Generate creative prompt."""
    return """Generate a creative, innovative trading agent."""


def _get_analytical_prompt(**kwargs: dict) -> str:
    """Generate analytical prompt."""
    return """Generate an analytical trading agent focused on data analysis."""
