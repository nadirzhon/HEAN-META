"""Council member definitions with role-specific model assignments."""

from dataclasses import dataclass

from hean.council.prompts import (
    ARCHITECT_PROMPT,
    OPTIMIZER_PROMPT,
    QUANT_PROMPT,
    REVIEWER_PROMPT,
)


@dataclass
class CouncilMember:
    """Definition of a council member (AI model + role + perspective)."""

    role: str
    display_name: str
    model_id: str
    system_prompt: str
    max_tokens: int = 2000
    temperature: float = 0.3


DEFAULT_MEMBERS: list[CouncilMember] = [
    CouncilMember(
        role="architect",
        display_name="System Architect (Qwen)",
        model_id="qwen/qwen3-max",
        system_prompt=ARCHITECT_PROMPT,
        max_tokens=2500,
        temperature=0.2,
    ),
    CouncilMember(
        role="reviewer",
        display_name="Code Reviewer (Claude)",
        model_id="anthropic/claude-sonnet-4-5-20250929",
        system_prompt=REVIEWER_PROMPT,
        max_tokens=2000,
        temperature=0.2,
    ),
    CouncilMember(
        role="quant",
        display_name="Quant Analyst (GPT)",
        model_id="openai/gpt-4o",
        system_prompt=QUANT_PROMPT,
        max_tokens=2000,
        temperature=0.3,
    ),
    CouncilMember(
        role="optimizer",
        display_name="Performance Optimizer (DeepSeek)",
        model_id="deepseek/deepseek-r1",
        system_prompt=OPTIMIZER_PROMPT,
        max_tokens=2000,
        temperature=0.2,
    ),
]
