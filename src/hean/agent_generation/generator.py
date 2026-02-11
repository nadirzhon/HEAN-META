"""Agent generation system using LLM prompts."""

import ast
import re
from pathlib import Path
from typing import Any

from hean.agent_generation.prompts import SYSTEM_PROMPT, get_prompt
from hean.logging import get_logger

logger = get_logger(__name__)


class AgentGenerator:
    """Generator for trading agents using LLM prompts."""

    def __init__(self, llm_client: Any | None = None):
        """Initialize the agent generator.

        Args:
            llm_client: LLM client (OpenAI, Anthropic, etc.). If None,
                      will use environment variables or default.
        """
        self.llm_client = llm_client
        self._setup_llm_client()

    def _setup_llm_client(self) -> None:
        """Setup LLM client from environment or default."""
        import os
        from pathlib import Path

        # First, try to load from config (Pydantic settings loads from env automatically)
        try:
            from hean.config import settings
            gemini_key_from_config = settings.gemini_api_key
        except Exception:
            gemini_key_from_config = None

        # Try to load .env file if it exists (for local development)
        env_file = Path(".env")
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
            except ImportError:
                # If python-dotenv not installed, try manual parsing
                try:
                    with open(env_file) as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                key, value = line.split("=", 1)
                                os.environ[key.strip()] = value.strip()
                except Exception:
                    pass

        # Try OpenRouter FIRST (Qwen3-Max-Thinking — cheapest + best quality)
        try:
            import openai as _openai  # type: ignore
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key:
                self.llm_client = _openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
                self._is_openrouter = True
                logger.info("Using OpenRouter client (Qwen3-Max-Thinking)")
                return
        except ImportError:
            pass

        # Try OpenAI
        try:
            import openai  # type: ignore
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm_client = openai.OpenAI(api_key=api_key)
                logger.info("Using OpenAI client")
                return
        except ImportError:
            pass

        # Try Anthropic
        try:
            import anthropic  # type: ignore
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.llm_client = anthropic.Anthropic(api_key=api_key)
                logger.info("Using Anthropic client")
                return
        except ImportError:
            pass

        # Try Google Gemini (priority: env var > config > None)
        try:
            import google.generativeai as genai  # type: ignore

            # Try environment variable first, then config
            api_key = os.getenv("GEMINI_API_KEY") or gemini_key_from_config

            if api_key and api_key.strip():
                genai.configure(api_key=api_key)
                # Store genai module as client, but also store a flag
                self.llm_client = genai
                self._is_gemini = True
                logger.info("Using Google Gemini client")
                return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to configure Gemini: {e}")

        logger.warning("No LLM client configured. Set OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY")

    def generate_agent(
        self, prompt_type: str, output_path: Path | str | None = None, **prompt_kwargs: Any
    ) -> str:
        """Generate a trading agent using LLM.

        Args:
            prompt_type: Type of prompt (initial, evolution, etc.)
            output_path: Optional path to save generated code
            **prompt_kwargs: Variables for prompt formatting

        Returns:
            Generated Python code as string
        """
        if not self.llm_client:
            raise RuntimeError("No LLM client configured")

        # Get the prompt
        user_prompt = get_prompt(prompt_type, **prompt_kwargs)

        # Call LLM
        code = self._call_llm(user_prompt)

        # Extract code from markdown if needed
        code = self._extract_code(code)

        # Validate code
        self._validate_code(code)

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(code, encoding="utf-8")
            logger.info(f"Generated agent saved to {output_path}")

        return code

    def _call_llm(self, user_prompt: str) -> str:
        """Call LLM with prompt, with automatic fallback across providers.

        Tries the primary client first. On quota/auth errors, falls back
        to other available providers (OpenAI → Anthropic → Gemini).

        Args:
            user_prompt: User prompt text

        Returns:
            LLM response
        """
        import os

        # Build ordered list of (call_fn, available) based on primary client
        providers: list[tuple[str, callable]] = []

        client_type = type(self.llm_client).__name__ if self.llm_client else ""
        is_gemini = hasattr(self, "_is_gemini") and self._is_gemini
        is_openrouter = hasattr(self, "_is_openrouter") and self._is_openrouter

        # Primary provider first
        if is_openrouter:
            providers.append(("openrouter", self._call_openrouter))
        elif is_gemini:
            providers.append(("gemini", self._call_gemini))
        elif "OpenAI" in client_type:
            providers.append(("openai", self._call_openai))
        elif "Anthropic" in client_type:
            providers.append(("anthropic", self._call_anthropic))

        # Add remaining providers as fallbacks (cheapest first)
        if os.getenv("OPENROUTER_API_KEY") and not any(p[0] == "openrouter" for p in providers):
            providers.append(("openrouter", self._call_openrouter))
        if os.getenv("GEMINI_API_KEY") and not any(p[0] == "gemini" for p in providers):
            providers.append(("gemini", self._call_gemini))
        if os.getenv("OPENAI_API_KEY") and not any(p[0] == "openai" for p in providers):
            providers.append(("openai", self._call_openai))
        if os.getenv("ANTHROPIC_API_KEY") and not any(p[0] == "anthropic" for p in providers):
            providers.append(("anthropic", self._call_anthropic))

        last_error = None
        for provider_name, call_fn in providers:
            try:
                result = call_fn(user_prompt)
                return result
            except Exception as e:
                error_str = str(e).lower()
                # Retryable errors: quota, rate limit, auth, leaked key
                if any(kw in error_str for kw in ["quota", "429", "402", "rate_limit", "401", "403", "leaked", "insufficient", "credits"]):
                    logger.warning(f"AgentGenerator: {provider_name} failed ({e}), trying next provider...")
                    last_error = e
                    continue
                # Non-retryable error — re-raise
                raise

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def _call_openrouter(self, user_prompt: str) -> str:
        """Call OpenRouter API (Qwen3-Max-Thinking — best price/quality)."""
        import os

        client = self.llm_client
        is_openrouter = hasattr(self, "_is_openrouter") and self._is_openrouter
        if not is_openrouter:
            import openai as _openai  # type: ignore
            client = _openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )

        response = client.chat.completions.create(
            model="qwen/qwen3-max",
            max_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            extra_headers={
                "HTTP-Referer": "https://hean.trading",
                "X-Title": "HEAN Agent Generator",
            },
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenRouter returned empty response")

        # Strip thinking tags if present (Qwen3 thinking model)
        if "<think>" in content:
            import re
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        logger.info("Successfully used OpenRouter: qwen/qwen3-max")
        return content

    def _call_openai(self, user_prompt: str) -> str:
        """Call OpenAI API."""
        import os

        # Use existing client or create one for fallback
        client = self.llm_client
        client_type = type(client).__name__ if client else ""
        if "OpenAI" not in client_type:
            import openai  # type: ignore
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Try different models in order of preference
        models = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]

        for model in models:
            try:
                response = client.chat.completions.create(  # type: ignore
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                )
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("OpenAI returned empty response")
                logger.info(f"Successfully used model: {model}")
                return content
            except Exception as e:
                if "model_not_found" in str(e) or "404" in str(e):
                    logger.debug(f"Model {model} not available, trying next...")
                    continue
                raise

        raise RuntimeError(f"None of the models {models} are available")

    def _call_anthropic(self, user_prompt: str) -> str:
        """Call Anthropic API."""
        import os

        # Use existing client or create one for fallback
        client = self.llm_client
        client_type = type(client).__name__ if client else ""
        if "Anthropic" not in client_type:
            import anthropic  # type: ignore
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(  # type: ignore
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        if not response.content or len(response.content) == 0:
            raise RuntimeError("Anthropic returned empty response")
        text_content = response.content[0].text
        if not text_content:
            raise RuntimeError("Anthropic returned empty text")
        return text_content

    def _call_gemini(self, user_prompt: str) -> str:
        """Call Google Gemini API."""
        import google.generativeai as genai  # type: ignore

        # Combine system prompt and user prompt
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

        # Try different models in order of preference
        models = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]

        for model_name in models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=4096,
                    ),
                )
                if not response.text:
                    raise RuntimeError("Gemini returned empty response")
                logger.info(f"Successfully used Gemini model: {model_name}")
                return response.text
            except Exception as e:
                if "not found" in str(e).lower() or "404" in str(e):
                    logger.debug(f"Gemini model {model_name} not available, trying next...")
                    continue
                raise

        raise RuntimeError(f"None of the Gemini models {models} are available")

    def _extract_code(self, text: str) -> str:
        """Extract Python code from LLM response.

        Args:
            text: LLM response text

        Returns:
            Extracted Python code
        """
        # Try to find code blocks
        code_block_pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(code_block_pattern, text, re.DOTALL)

        if matches:
            # Return the largest code block (likely the main code)
            return max(matches, key=len).strip()

        # If no code blocks, assume entire text is code
        return text.strip()

    def _validate_code(self, code: str) -> None:
        """Validate generated code.

        Args:
            code: Python code to validate

        Raises:
            SyntaxError: If code is invalid
        """
        try:
            ast.parse(code)
            logger.info("Generated code is valid Python")
        except SyntaxError as e:
            logger.error(f"Generated code has syntax errors: {e}")
            raise

    def generate_initial_agents(
        self, count: int = 10, output_dir: Path | str = "generated_agents"
    ) -> list[str]:
        """Generate multiple initial agents.

        Args:
            count: Number of agents to generate
            output_dir: Directory to save agents

        Returns:
            List of generated code strings
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated = []
        for i in range(count):
            logger.info(f"Generating agent {i + 1}/{count}")
            try:
                code = self.generate_agent(
                    prompt_type="initial", output_path=output_dir / f"agent_{i + 1:03d}.py"
                )
                generated.append(code)
            except Exception as e:
                logger.error(f"Failed to generate agent {i + 1}: {e}")
                continue

        logger.info(f"Generated {len(generated)}/{count} agents")
        return generated

    def evolve_agent(
        self,
        best_agents_info: str,
        worst_agents_info: str,
        market_conditions: str,
        performance_metrics: str,
        output_path: Path | str | None = None,
    ) -> str:
        """Generate evolved agent based on best/worst performers.

        Args:
            best_agents_info: Information about best performing agents
            worst_agents_info: Information about worst performing agents
            market_conditions: Current market conditions
            performance_metrics: Performance metrics summary
            output_path: Optional path to save generated code

        Returns:
            Generated Python code
        """
        return self.generate_agent(
            prompt_type="evolution",
            output_path=output_path,
            best_agents_info=best_agents_info,
            worst_agents_info=worst_agents_info,
            market_conditions=market_conditions,
            performance_metrics=performance_metrics,
        )

    def mutate_agent(
        self,
        agent_code: str,
        profit_factor: float,
        total_pnl: float,
        max_drawdown_pct: float,
        win_rate: float,
        issues: str,
        output_path: Path | str | None = None,
    ) -> str:
        """Mutate/improve existing agent.

        Args:
            agent_code: Original agent code
            profit_factor: Current profit factor
            total_pnl: Total PnL
            max_drawdown_pct: Max drawdown percentage
            win_rate: Win rate percentage
            issues: Description of issues
            output_path: Optional path to save generated code

        Returns:
            Improved Python code
        """
        return self.generate_agent(
            prompt_type="mutation",
            output_path=output_path,
            agent_code=agent_code,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            issues=issues,
        )

    def generate_market_specialized_agent(
        self,
        volatility_level: str,
        volatility_value: float,
        trend_direction: str,
        trend_strength: str,
        volume_level: str,
        market_regime: str,
        spread_bps: float,
        historical_summary: str,
        suggested_style: str,
        suggested_timeframe: str,
        suggested_size: str,
        risk_approach: str,
        output_path: Path | str | None = None,
    ) -> str:
        """Generate agent specialized for market conditions.

        Args:
            volatility_level: Volatility level description
            volatility_value: Volatility numeric value
            trend_direction: Trend direction
            trend_strength: Trend strength
            volume_level: Volume level
            market_regime: Market regime
            spread_bps: Spread in basis points
            historical_summary: Historical data summary
            suggested_style: Suggested trading style
            suggested_timeframe: Suggested timeframe
            suggested_size: Suggested position size
            risk_approach: Risk management approach
            output_path: Optional path to save generated code

        Returns:
            Generated Python code
        """
        return self.generate_agent(
            prompt_type="market_conditions",
            output_path=output_path,
            volatility_level=volatility_level,
            volatility_value=volatility_value,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            volume_level=volume_level,
            market_regime=market_regime,
            spread_bps=spread_bps,
            historical_summary=historical_summary,
            suggested_style=suggested_style,
            suggested_timeframe=suggested_timeframe,
            suggested_size=suggested_size,
            risk_approach=risk_approach,
        )

    def generate_hybrid_agent(
        self,
        agent1_code: str,
        pf1: float,
        pnl1: float,
        agent2_code: str,
        pf2: float,
        wr2: float,
        agent3_code: str,
        pf3: float,
        sharpe3: float,
        output_path: Path | str | None = None,
    ) -> str:
        """Generate hybrid agent combining multiple agents.

        Args:
            agent1_code: Code of best PF agent
            pf1: Profit factor of agent 1
            pnl1: PnL of agent 1
            agent2_code: Code of best WR agent
            pf2: Profit factor of agent 2
            wr2: Win rate of agent 2
            agent3_code: Code of best Sharpe agent
            pf3: Profit factor of agent 3
            sharpe3: Sharpe ratio of agent 3
            output_path: Optional path to save generated code

        Returns:
            Generated Python code
        """
        return self.generate_agent(
            prompt_type="hybrid",
            output_path=output_path,
            agent1_code=agent1_code,
            pf1=pf1,
            pnl1=pnl1,
            agent2_code=agent2_code,
            pf2=pf2,
            wr2=wr2,
            agent3_code=agent3_code,
            pf3=pf3,
            sharpe3=sharpe3,
        )

    def generate_problem_focused_agent(
        self,
        problem_description: str,
        current_pf: float,
        problem_areas: str,
        failed_patterns: str,
        focus_area_1: str,
        focus_area_2: str,
        focus_area_3: str,
        output_path: Path | str | None = None,
    ) -> str:
        """Generate agent focused on solving specific problem.

        Args:
            problem_description: Description of the problem
            current_pf: Current profit factor
            problem_areas: Problem areas
            failed_patterns: Failed patterns to avoid
            focus_area_1: First focus area
            focus_area_2: Second focus area
            focus_area_3: Third focus area
            output_path: Optional path to save generated code

        Returns:
            Generated Python code
        """
        return self.generate_agent(
            prompt_type="problem_focused",
            output_path=output_path,
            problem_description=problem_description,
            current_pf=current_pf,
            problem_areas=problem_areas,
            failed_patterns=failed_patterns,
            focus_area_1=focus_area_1,
            focus_area_2=focus_area_2,
            focus_area_3=focus_area_3,
        )

    def evaluate_and_improve(
        self,
        agent_code: str,
        pf: float,
        pnl: float,
        dd: float,
        wr: float,
        sharpe: float,
        trades: int,
        output_path: Path | str | None = None,
    ) -> str:
        """Evaluate agent and generate improved version.

        Args:
            agent_code: Agent code to evaluate
            pf: Profit factor
            pnl: Total PnL
            dd: Max drawdown percentage
            wr: Win rate percentage
            sharpe: Sharpe ratio
            trades: Total number of trades
            output_path: Optional path to save generated code

        Returns:
            Improved Python code
        """
        return self.generate_agent(
            prompt_type="evaluation",
            output_path=output_path,
            agent_code=agent_code,
            pf=pf,
            pnl=pnl,
            dd=dd,
            wr=wr,
            sharpe=sharpe,
            trades=trades,
        )

    def generate_creative_agent(self, output_path: Path | str | None = None) -> str:
        """Generate creative, innovative agent from scratch.

        Args:
            output_path: Optional path to save generated code

        Returns:
            Generated Python code
        """
        return self.generate_agent(
            prompt_type="creative",
            output_path=output_path,
        )
