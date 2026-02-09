"""OpenAI integration for process generation with strict JSON validation."""

import json
from pathlib import Path
from typing import Any

from hean.logging import get_logger
from hean.process_factory.schemas import (
    BybitEnvironmentSnapshot,
    ProcessDefinition,
    ProcessPortfolioEntry,
)

logger = get_logger(__name__)


class OpenAIProcessFactory:
    """Generates process definitions using OpenAI with strict JSON validation."""

    def __init__(self, api_key: str | None = None, prompt_template_path: str | Path | None = None) -> None:
        """Initialize OpenAI process factory.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            prompt_template_path: Path to prompt template file
        """
        self.api_key = api_key
        self.prompt_template_path = prompt_template_path or Path(__file__).parent.parent.parent.parent / "templates" / "openai_process_factory_prompt.txt"

        # Try to import OpenAI client
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
            self._available = True
        except ImportError:
            logger.warning("OpenAI library not available. Install with: pip install openai")
            self._client = None
            self._available = False

    async def generate_process(
        self,
        snapshot: BybitEnvironmentSnapshot,
        portfolio: list[ProcessPortfolioEntry],
        top_failures: list[str] | None = None,
        capability_graph: dict[str, set[str]] | None = None,
        max_steps: int = 20,
        max_human_tasks: int = 5,
    ) -> ProcessDefinition | None:
        """Generate a process definition using OpenAI with strict validation.

        Args:
            snapshot: Environment snapshot
            portfolio: Current process portfolio
            top_failures: List of process IDs that failed (for learning)
            capability_graph: Capability graph (capability -> opportunities)
            max_steps: Maximum steps per process (default 20)
            max_human_tasks: Maximum human tasks per process (default 5)

        Returns:
            Generated process definition, or None if generation failed
        """
        if not self._available:
            logger.error("OpenAI not available")
            return None

        # Build prompt
        prompt = self._build_prompt(
            snapshot, portfolio, top_failures, capability_graph, max_steps, max_human_tasks
        )

        try:
            # Call OpenAI API with deterministic settings
            response = self._client.chat.completions.create(  # type: ignore
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a process definition generator. Return ONLY valid JSON matching the ProcessDefinition schema. No prose, no explanations, just JSON. The JSON must be valid and complete.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for more deterministic output
                response_format={"type": "json_object"},
                seed=42,  # Seed for reproducibility (if supported by model)
            )

            content = response.choices[0].message.content
            if not content:
                logger.error("Empty response from OpenAI")
                return None

            # Parse JSON with strict validation
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from OpenAI: {e}")
                logger.error(f"Response content: {content[:500]}")
                return None

            # Validate and create ProcessDefinition
            process = self._validate_and_create_process(data, max_steps, max_human_tasks)
            return process

        except Exception as e:
            logger.error(f"Failed to generate process: {e}", exc_info=True)
            return None

    def _build_prompt(
        self,
        snapshot: BybitEnvironmentSnapshot,
        portfolio: list[ProcessPortfolioEntry],
        top_failures: list[str] | None,
        capability_graph: dict[str, set[str]] | None,
        max_steps: int = 20,
        max_human_tasks: int = 5,
    ) -> str:
        """Build prompt for OpenAI.

        Args:
            snapshot: Environment snapshot
            portfolio: Current portfolio
            top_failures: Top failures
            capability_graph: Capability graph

        Returns:
            Prompt string
        """
        # Load template if available
        template = self._load_template()

        # Build context
        context = {
            "snapshot": {
                "balances": snapshot.balances,
                "positions_count": len(snapshot.positions),
                "funding_rates": snapshot.funding_rates,
            },
            "portfolio": [
                {
                    "process_id": p.process_id,
                    "state": p.state.value,
                    "pnl": p.pnl_sum,
                    "runs": p.runs_count,
                }
                for p in portfolio[:10]  # Limit to top 10
            ],
            "top_failures": top_failures or [],
            "capability_gaps": list(capability_graph.keys()) if capability_graph else [],
        }

        if template:
            # Use template with context
            prompt = template.format(
                snapshot=json.dumps(context["snapshot"], indent=2),
                portfolio=json.dumps(context["portfolio"], indent=2),
                failures=json.dumps(context["top_failures"], indent=2),
                capabilities=json.dumps(context["capability_gaps"], indent=2),
            )
        else:
            # Fallback prompt
            prompt = f"""Generate a new process definition based on this context:

Environment Snapshot:
{json.dumps(context['snapshot'], indent=2)}

Current Portfolio:
{json.dumps(context['portfolio'], indent=2)}

Top Failures:
{json.dumps(context['top_failures'], indent=2)}

Capability Gaps:
{json.dumps(context['capability_gaps'], indent=2)}

Return ONLY valid JSON matching the ProcessDefinition schema. Include:
- id: unique identifier
- name: human-readable name
- type: one of TRADING, EARN, CAMPAIGN, BONUS, DATA, ACCESS, OTHER
- description: process description
- requirements: dict with needs_bybit, needs_ui, etc.
- inputs_schema: JSON schema for inputs
- actions: list of ActionStep objects
- expected_outputs: list of output keys
- safety: SafetyPolicy object
- measurement: MeasurementSpec object
- kill_conditions: list of KillCondition objects
- scale_rules: list of ScaleRule objects

SAFETY: Do NOT generate processes that:
- Handle credentials directly
- Scrape UIs
- Violate Terms of Service
- Perform illegal actions
- Execute orders directly (use HUMAN_TASK or interfaces)

BUDGET GUARDRAILS:
- Maximum {max_steps} steps per process
- Maximum {max_human_tasks} human tasks per process
- Must declare automation feasibility in requirements

REQUIREMENTS:
- MUST include kill_conditions (at least one)
- MUST include measurement spec with metrics
- MUST declare if automation is feasible (needs_bybit, needs_ui, etc.)

Default to HUMAN_TASK for anything uncertain."""

        return prompt

    def _load_template(self) -> str | None:
        """Load prompt template from file.

        Returns:
            Template string or None if not found
        """
        try:
            if self.prompt_template_path and Path(self.prompt_template_path).exists():
                return Path(self.prompt_template_path).read_text()
        except Exception as e:
            logger.warning(f"Could not load prompt template: {e}")
        return None

    def _validate_and_create_process(
        self, data: dict[str, Any], max_steps: int = 20, max_human_tasks: int = 5
    ) -> ProcessDefinition:
        """Validate and create ProcessDefinition from JSON data with strict checks.

        Args:
            data: JSON data
            max_steps: Maximum allowed steps
            max_human_tasks: Maximum allowed human tasks

        Returns:
            ProcessDefinition

        Raises:
            ValueError: If data is invalid
        """
        # Safety filter: reject dangerous processes
        self._safety_filter(data)

        # Validate required fields
        required_fields = ["id", "name", "type", "description"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Validate kill_conditions (required)
        if "kill_conditions" not in data or not data["kill_conditions"]:
            raise ValueError("Missing kill_conditions (required for safety)")

        # Validate measurement spec (required)
        if "measurement" not in data:
            raise ValueError("Missing measurement spec (required)")

        # Validate budget guardrails
        actions = data.get("actions", [])
        if len(actions) > max_steps:
            raise ValueError(
                f"Too many steps: {len(actions)} > {max_steps} (budget limit)"
            )

        human_task_count = sum(
            1
            for action in actions
            if isinstance(action, dict)
            and action.get("kind") == "HUMAN_TASK"
        )
        if human_task_count > max_human_tasks:
            raise ValueError(
                f"Too many human tasks: {human_task_count} > {max_human_tasks} (budget limit)"
            )

        # Validate automation feasibility declaration
        requirements = data.get("requirements", {})
        if not isinstance(requirements, dict):
            raise ValueError("requirements must be a dictionary")

        # Check for disallowed actions
        for action in actions:
            if isinstance(action, dict):
                action_desc = action.get("description", "").lower()
                # Reject credential handling
                if any(
                    keyword in action_desc
                    for keyword in ["credential", "password", "api_key", "secret"]
                ):
                    raise ValueError(
                        f"Action contains credential handling: {action_desc}"
                    )
                # Reject UI scraping
                if any(
                    keyword in action_desc
                    for keyword in ["scrape", "selenium", "automated browser"]
                ):
                    raise ValueError(f"Action contains UI scraping: {action_desc}")

        # Create ProcessDefinition (Pydantic will validate)
        try:
            process = ProcessDefinition(**data)
            return process
        except Exception as e:
            raise ValueError(f"Invalid process definition: {e}") from e

    def _safety_filter(self, data: dict[str, Any]) -> None:
        """Filter out unsafe processes.

        Args:
            data: Process definition data

        Raises:
            ValueError: If process is unsafe
        """
        # Check for dangerous keywords
        description = data.get("description", "").lower()
        name = data.get("name", "").lower()

        dangerous_keywords = [
            "credential",
            "password",
            "api_key",
            "secret",
            "scrape",
            "selenium",
            "automated browser",
            "bypass",
            "hack",
            "exploit",
        ]

        for keyword in dangerous_keywords:
            if keyword in description or keyword in name:
                raise ValueError(f"Process contains unsafe keyword: {keyword}")

        # Check actions for unsafe operations
        actions = data.get("actions", [])
        for action in actions:
            action_desc = action.get("description", "").lower() if isinstance(action, dict) else ""
            if any(keyword in action_desc for keyword in dangerous_keywords):
                raise ValueError(f"Action contains unsafe keyword: {action_desc}")

