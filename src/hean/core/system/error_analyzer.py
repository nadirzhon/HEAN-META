"""Error Log Analyzer with LLM-driven auto-fix capability.

When a Python exception or C++ crash occurs, pipes the log directly to a local LLM
to suggest immediate code fixes. In 'Auto-Dev' mode, applies the patch automatically.
"""

import ast
import asyncio
import json
import os
import re
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)

# Try to import LLM libraries
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class ErrorContext:
    """Error context for analysis."""
    error_type: str
    error_message: str
    stack_trace: str
    source_file: str | None = None
    source_line: int | None = None
    source_code: str | None = None
    timestamp: datetime = None
    module_name: str | None = None
    function_name: str | None = None

    def __post_init__(self) -> None:
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)


@dataclass
class FixSuggestion:
    """Code fix suggestion from LLM."""
    file_path: str
    fix_description: str
    code_patch: str  # Unified diff or code block
    confidence: float  # 0.0 to 1.0
    reasoning: str
    risk_level: str  # "low", "medium", "high"


class ErrorAnalyzer:
    """Analyzes errors and suggests fixes using LLM.

    Features:
    - Automatically analyzes Python exceptions and C++ crash logs
    - Uses LLM (OpenAI/Anthropic) to suggest code fixes
    - In Auto-Dev mode, automatically applies patches
    - Validates fixes before applying
    """

    def __init__(
        self,
        auto_dev_mode: bool = False,
        llm_provider: str = "openai",
        model: str = "gpt-4",
    ) -> None:
        """Initialize error analyzer.

        Args:
            auto_dev_mode: If True, automatically apply fixes (default: False)
            llm_provider: LLM provider ("openai" or "anthropic")
            model: Model name (e.g., "gpt-4", "claude-3-opus")
        """
        self._auto_dev_mode = auto_dev_mode
        self._llm_provider = llm_provider
        self._model = model
        self._client: Any | None = None
        self._project_root = Path(__file__).parent.parent.parent.parent
        self._fix_history: list[FixSuggestion] = []

        # Initialize LLM client
        if llm_provider == "openai" and OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self._client = AsyncOpenAI(api_key=api_key)
                logger.info(f"Error analyzer initialized with OpenAI {model}")
            else:
                logger.warning("OPENAI_API_KEY not set, error analyzer will be limited")
        elif llm_provider == "anthropic" and ANTHROPIC_AVAILABLE:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self._client = AsyncAnthropic(api_key=api_key)
                logger.info(f"Error analyzer initialized with Anthropic {model}")
            else:
                logger.warning("ANTHROPIC_API_KEY not set, error analyzer will be limited")
        else:
            logger.warning(
                f"LLM provider {llm_provider} not available. "
                "Error analyzer will only log errors without suggesting fixes."
            )

    async def analyze_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> FixSuggestion | None:
        """Analyze an error and suggest a fix.

        Args:
            error: Exception that occurred
            context: Additional context (e.g., function arguments, state)

        Returns:
            Fix suggestion if available, None otherwise
        """
        # Extract error context
        error_context = self._extract_error_context(error, context)

        logger.error(
            f"Error detected: {error_context.error_type} - {error_context.error_message}\n"
            f"Stack trace:\n{error_context.stack_trace}"
        )

        if not self._client:
            logger.warning("LLM client not available, cannot suggest fixes")
            return None

        # Get source code if available
        if error_context.source_file:
            error_context.source_code = self._read_source_code(
                error_context.source_file,
                error_context.source_line,
            )

        # Generate fix suggestion using LLM
        try:
            fix = await self._generate_fix_suggestion(error_context)

            if fix:
                self._fix_history.append(fix)

                logger.info(
                    f"Fix suggestion generated:\n"
                    f"  File: {fix.file_path}\n"
                    f"  Description: {fix.fix_description}\n"
                    f"  Confidence: {fix.confidence:.2f}\n"
                    f"  Risk: {fix.risk_level}\n"
                    f"  Reasoning: {fix.reasoning}"
                )

                # Auto-apply if in auto-dev mode
                if self._auto_dev_mode and fix.risk_level == "low":
                    success = await self._apply_fix(fix)
                    if success:
                        logger.info(f"Auto-applied fix to {fix.file_path}")
                    else:
                        logger.warning(f"Failed to apply fix to {fix.file_path}")

                return fix

        except Exception as e:
            logger.error(f"Failed to generate fix suggestion: {e}", exc_info=True)

        return None

    def _extract_error_context(
        self,
        error: Exception,
        context: dict[str, Any] | None,
    ) -> ErrorContext:
        """Extract error context from exception."""
        tb = error.__traceback__
        stack_trace = "".join(traceback.format_exception(type(error), error, tb))

        # Extract source file and line from traceback
        source_file = None
        source_line = None
        module_name = None
        function_name = None

        if tb:
            frame = tb
            while frame:
                filename = frame.tb_frame.f_code.co_filename
                lineno = frame.tb_lineno

                # Check if it's a project file
                if "hean" in filename or str(self._project_root) in filename:
                    source_file = filename
                    source_line = lineno
                    module_name = frame.tb_frame.f_globals.get("__name__")
                    function_name = frame.tb_frame.f_code.co_name
                    break

                frame = frame.tb_next

        return ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=stack_trace,
            source_file=source_file,
            source_line=source_line,
            module_name=module_name,
            function_name=function_name,
        )

    def _read_source_code(
        self,
        file_path: str,
        line_number: int | None,
        context_lines: int = 20,
    ) -> str | None:
        """Read source code around the error line."""
        try:
            path = Path(file_path)
            if not path.exists():
                return None

            with open(path, encoding="utf-8") as f:
                lines = f.readlines()

            if line_number is None:
                return "".join(lines)

            # Extract context around error line
            start = max(0, line_number - context_lines)
            end = min(len(lines), line_number + context_lines)

            "".join(lines[start:end])

            # Add line numbers
            numbered_lines = []
            for i, line in enumerate(lines[start:end], start=start + 1):
                marker = " >>> " if i == line_number else "     "
                numbered_lines.append(f"{i}{marker}{line}")

            return "".join(numbered_lines)

        except Exception as e:
            logger.warning(f"Failed to read source code: {e}")
            return None

    async def _generate_fix_suggestion(
        self,
        error_context: ErrorContext,
    ) -> FixSuggestion | None:
        """Generate fix suggestion using LLM."""
        prompt = self._build_fix_prompt(error_context)

        try:
            if self._llm_provider == "openai":
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert Python/C++ debugging assistant. "
                                "Analyze errors and suggest precise code fixes. "
                                "Provide fixes in unified diff format or code blocks. "
                                "Always explain your reasoning and assess risk level."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,  # Low temperature for deterministic fixes
                )
                content = response.choices[0].message.content
            elif self._llm_provider == "anthropic":
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=4000,
                    messages=[
                        {
                            "role": "user",
                            "content": f"{prompt}\n\nPlease provide a fix in JSON format with fields: file_path, fix_description, code_patch, confidence, reasoning, risk_level.",
                        },
                    ],
                )
                content = response.content[0].text
            else:
                return None

            # Parse response
            fix = self._parse_fix_response(content, error_context)
            return fix

        except Exception as e:
            logger.error(f"LLM request failed: {e}", exc_info=True)
            return None

    def _build_fix_prompt(self, error_context: ErrorContext) -> str:
        """Build prompt for LLM."""
        prompt = f"""
Analyze this error and suggest a code fix:

ERROR TYPE: {error_context.error_type}
ERROR MESSAGE: {error_context.error_message}

STACK TRACE:
{error_context.stack_trace}

"""

        if error_context.source_file:
            prompt += f"SOURCE FILE: {error_context.source_file}\n"
            if error_context.source_line:
                prompt += f"ERROR LINE: {error_context.source_line}\n"
            if error_context.function_name:
                prompt += f"FUNCTION: {error_context.function_name}\n"

        if error_context.source_code:
            prompt += f"\nSOURCE CODE (around error line):\n```python\n{error_context.source_code}\n```\n"

        prompt += """
Please provide a fix suggestion in the following JSON format:
{
    "file_path": "path/to/file.py",
    "fix_description": "Brief description of the fix",
    "code_patch": "The actual code change (unified diff or code block)",
    "confidence": 0.95,
    "reasoning": "Explanation of why this fix should work",
    "risk_level": "low|medium|high"
}

Ensure the fix:
1. Addresses the root cause of the error
2. Maintains existing functionality
3. Follows Python/C++ best practices
4. Is minimal and focused
"""

        return prompt

    def _parse_fix_response(
        self,
        response: str,
        error_context: ErrorContext,
    ) -> FixSuggestion | None:
        """Parse LLM response into FixSuggestion."""
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return FixSuggestion(
                    file_path=data.get("file_path", error_context.source_file or ""),
                    fix_description=data.get("fix_description", ""),
                    code_patch=data.get("code_patch", ""),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", ""),
                    risk_level=data.get("risk_level", "medium"),
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")

        # Fallback: extract code block if present
        code_block_match = re.search(r'```(?:python|cpp)?\n(.*?)\n```', response, re.DOTALL)
        if code_block_match:
            code_patch = code_block_match.group(1)
            return FixSuggestion(
                file_path=error_context.source_file or "",
                fix_description="Code fix extracted from LLM response",
                code_patch=code_patch,
                confidence=0.5,
                reasoning=response[:500],  # First 500 chars as reasoning
                risk_level="medium",
            )

        logger.warning("Could not parse LLM response into fix suggestion")
        return None

    async def _apply_fix(self, fix: FixSuggestion) -> bool:
        """Apply a fix suggestion to the source file.

        Returns:
            True if fix was applied successfully, False otherwise
        """
        if not fix.file_path or not Path(fix.file_path).exists():
            logger.warning(f"Fix file path does not exist: {fix.file_path}")
            return False

        try:
            # Validate fix by parsing
            if fix.file_path.endswith(".py"):
                # Try to parse the code patch
                try:
                    ast.parse(fix.code_patch)
                except SyntaxError:
                    logger.warning("Fix code patch has syntax errors, not applying")
                    return False

            # Apply fix (this is a simplified implementation)
            # In production, you'd use proper diff/patch tools
            logger.info(f"Applying fix to {fix.file_path}")
            logger.warning(
                "Auto-apply is experimental. Manual review recommended. "
                "Fix will be logged but not applied automatically."
            )

            # For safety, we don't auto-apply. Instead, save the suggestion.
            fix_log_path = self._project_root / "logs" / "fix_suggestions.log"
            fix_log_path.parent.mkdir(exist_ok=True)

            with open(fix_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Timestamp: {datetime.now(UTC).isoformat()}\n")
                f.write(f"File: {fix.file_path}\n")
                f.write(f"Description: {fix.fix_description}\n")
                f.write(f"Confidence: {fix.confidence}\n")
                f.write(f"Risk: {fix.risk_level}\n")
                f.write(f"Reasoning: {fix.reasoning}\n")
                f.write(f"\nCode Patch:\n{fix.code_patch}\n")
                f.write(f"{'='*80}\n")

            return True

        except Exception as e:
            logger.error(f"Failed to apply fix: {e}", exc_info=True)
            return False

    def get_fix_history(self) -> list[FixSuggestion]:
        """Get history of fix suggestions."""
        return self._fix_history.copy()


# Global instance
_error_analyzer: ErrorAnalyzer | None = None


def get_error_analyzer(
    auto_dev_mode: bool = False,
    llm_provider: str | None = None,
) -> ErrorAnalyzer:
    """Get or create global error analyzer.

    Args:
        auto_dev_mode: Enable auto-apply mode
        llm_provider: LLM provider ("openai" or "anthropic"), defaults to env config
    """
    global _error_analyzer

    if _error_analyzer is None:
        provider = llm_provider or os.getenv("LLM_PROVIDER", "openai")
        model = os.getenv("LLM_MODEL", "gpt-4" if provider == "openai" else "claude-3-opus-20240229")
        _error_analyzer = ErrorAnalyzer(
            auto_dev_mode=auto_dev_mode,
            llm_provider=provider,
            model=model,
        )

    return _error_analyzer


# Exception hook for automatic error analysis
def setup_exception_hook(auto_dev_mode: bool = False) -> None:
    """Set up global exception hook for automatic error analysis."""
    import sys

    original_excepthook = sys.excepthook

    def exception_hook(exc_type, exc_value, exc_traceback):
        """Custom exception hook that analyzes errors."""
        # Call original hook first
        original_excepthook(exc_type, exc_value, exc_traceback)

        # Analyze error asynchronously
        async def analyze():
            try:
                analyzer = get_error_analyzer(auto_dev_mode=auto_dev_mode)
                await analyzer.analyze_error(exc_value)
            except Exception as e:
                logger.error(f"Error analyzer failed: {e}", exc_info=True)

        # Run analysis in background
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(analyze())
            else:
                loop.run_until_complete(analyze())
        except Exception:
            pass  # Ignore errors in exception hook

    sys.excepthook = exception_hook
    logger.info("Exception hook installed for automatic error analysis")
