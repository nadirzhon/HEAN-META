"""Generative Strategy Sandbox: LLM-powered code generation for self-evolving strategies.

Analyzes recent loss events and generates new mathematical logic for the Warden.
Uses local LLM (llama.cpp) for analysis and code generation.
Runs generated code in a restricted Python sandbox with shadow testing.
"""

import ast
import asyncio
import inspect
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger
from hean.portfolio.accounting import PortfolioAccounting

logger = get_logger(__name__)


@dataclass
class LossEvent:
    """Represents a loss event for analysis."""
    timestamp: datetime
    symbol: str
    strategy_id: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    holding_time_seconds: float
    market_conditions: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class GeneratedCode:
    """Represents generated code with metadata."""
    code: str
    description: str
    target_component: str  # e.g., "warden", "position_sizer", "edge_estimator"
    generation_timestamp: datetime
    test_results: Optional[dict[str, Any]] = None
    shadow_test_duration_hours: float = 0.0
    approval_status: str = "pending"  # pending, shadow_testing, approved, rejected


class RestrictedSandbox:
    """Python restricted sandbox for executing generated code safely."""

    ALLOWED_MODULES = {
        'math', 'random', 'statistics', 'collections', 'dataclasses',
        'typing', 'decimal', 'fractions', 'itertools', 'operator'
    }
    
    FORBIDDEN_PATTERNS = [
        r'__import__',
        r'exec\s*\(',
        r'eval\s*\(',
        r'compile\s*\(',
        r'open\s*\(',
        r'file\s*\(',
        r'input\s*\(',
        r'raw_input\s*\(',
        r'subprocess',
        r'os\.',
        r'sys\.',
        r'import\s+os',
        r'import\s+sys',
        r'import\s+subprocess',
    ]

    def __init__(self):
        """Initialize the sandbox."""
        self._globals = {
            '__builtins__': {
                'abs': abs, 'all': all, 'any': any, 'bool': bool,
                'dict': dict, 'enumerate': enumerate, 'float': float,
                'int': int, 'len': len, 'list': list, 'max': max,
                'min': min, 'range': range, 'round': round, 'set': set,
                'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple,
                'type': type, 'zip': zip,
            }
        }
        
        # Import allowed modules
        for module_name in self.ALLOWED_MODULES:
            try:
                module = __import__(module_name)
                self._globals[module_name] = module
            except ImportError:
                pass

    def validate_code(self, code: str) -> tuple[bool, Optional[str]]:
        """Validate code before execution.
        
        Returns:
            (is_valid, error_message)
        """
        # Check for forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Forbidden pattern detected: {pattern}"
        
        # Parse and validate AST
        try:
            tree = ast.parse(code, mode='exec')
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Check AST nodes for dangerous operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] not in self.ALLOWED_MODULES:
                        return False, f"Forbidden import: {alias.name}"
            
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] not in self.ALLOWED_MODULES:
                    return False, f"Forbidden import from: {node.module}"
            
            if isinstance(node, (ast.Call, ast.Attribute)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ('eval', 'exec', 'compile', '__import__'):
                        return False, f"Forbidden function call: {node.func.id}"
        
        return True, None

    def execute(self, code: str, inputs: dict[str, Any]) -> tuple[bool, Any, Optional[str]]:
        """Execute code in sandbox.
        
        Args:
            code: Python code to execute
            inputs: Input variables to provide
            
        Returns:
            (success, result, error_message)
        """
        is_valid, error = self.validate_code(code)
        if not is_valid:
            return False, None, error
        
        # Prepare execution environment
        exec_globals = self._globals.copy()
        exec_globals.update(inputs)
        exec_locals = {}
        
        try:
            # Execute code
            exec(compile(code, '<sandbox>', 'exec'), exec_globals, exec_locals)
            
            # Extract result (look for common output patterns)
            if 'result' in exec_locals:
                return True, exec_locals['result'], None
            elif 'output' in exec_locals:
                return True, exec_locals['output'], None
            else:
                # Return all variables that aren't in globals
                result = {k: v for k, v in exec_locals.items() if k not in self._globals}
                return True, result, None
        except Exception as e:
            return False, None, str(e)


class LLMCodeGenerator:
    """LLM-powered code generator using llama.cpp."""

    def __init__(self, model_path: Optional[str] = None, llama_cpp_path: str = "llama-cli"):
        """Initialize LLM code generator.
        
        Args:
            model_path: Path to LLM model file (GGUF format)
            llama_cpp_path: Path to llama-cli or llama.cpp executable
        """
        self.model_path = Path(model_path) if model_path else None
        self.llama_cpp_path = llama_cpp_path
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if llama.cpp is available."""
        try:
            result = subprocess.run(
                [self.llama_cpp_path, "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("llama.cpp not found. Code generation will be disabled.")
            return False

    def generate_code(
        self,
        loss_events: list[LossEvent],
        target_component: str,
        prompt_template: Optional[str] = None
    ) -> Optional[GeneratedCode]:
        """Generate code based on loss event analysis.
        
        Args:
            loss_events: List of recent loss events
            target_component: Target component to generate code for
            prompt_template: Optional custom prompt template
            
        Returns:
            GeneratedCode object or None if generation fails
        """
        if not self._available or not self.model_path or not self.model_path.exists():
            logger.warning("LLM not available. Using fallback code generation.")
            return self._fallback_generation(loss_events, target_component)
        
        # Build prompt
        prompt = self._build_prompt(loss_events, target_component, prompt_template)
        
        try:
            # Call llama.cpp
            result = subprocess.run(
                [
                    self.llama_cpp_path,
                    "-m", str(self.model_path),
                    "-p", prompt,
                    "-n", "512",  # Max tokens
                    "-t", "4",    # Threads
                    "--temp", "0.7",
                    "--top-p", "0.9",
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"LLM generation failed: {result.stderr}")
                return None
            
            # Parse response
            response = result.stdout.strip()
            code, description = self._parse_llm_response(response)
            
            return GeneratedCode(
                code=code,
                description=description,
                target_component=target_component,
                generation_timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Code generation error: {e}")
            return None

    def _build_prompt(
        self,
        loss_events: list[LossEvent],
        target_component: str,
        custom_template: Optional[str]
    ) -> str:
        """Build prompt for LLM."""
        if custom_template:
            return custom_template.format(
                loss_events=loss_events,
                target_component=target_component
            )
        
        # Default prompt template
        loss_summary = self._summarize_loss_events(loss_events)
        
        return f"""You are a quantitative trading system analyst. Analyze the following loss events and generate improved mathematical logic.

Loss Events Summary:
{loss_summary}

Target Component: {target_component}

Generate Python code that:
1. Improves the {target_component} logic to avoid similar losses
2. Uses only safe mathematical operations (math, statistics, etc.)
3. Returns a function or formula that can be integrated into the system
4. Includes a brief description of the improvement

Format your response as JSON:
{{
    "description": "Brief description of the improvement",
    "code": "Python code here (function or formula)",
    "parameters": {{"param1": "description"}}
}}

Code:"""

    def _summarize_loss_events(self, events: list[LossEvent]) -> str:
        """Summarize loss events for prompt."""
        if not events:
            return "No loss events"
        
        total_loss = sum(e.pnl for e in events if e.pnl < 0)
        avg_loss_pct = sum(e.pnl_pct for e in events if e.pnl_pct < 0) / len([e for e in events if e.pnl_pct < 0])
        
        strategies = {}
        for event in events:
            strategies[event.strategy_id] = strategies.get(event.strategy_id, 0) + 1
        
        return f"""
Total Loss Events: {len(events)}
Total Loss: ${total_loss:.2f}
Average Loss %: {avg_loss_pct:.2f}%
Strategies Affected: {', '.join(strategies.keys())}
Most Recent: {events[-1].symbol} @ {events[-1].timestamp}
"""

    def _parse_llm_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response to extract code and description."""
        # Try to extract JSON
        json_match = re.search(r'\{[^{}]*"code"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get('code', ''), data.get('description', '')
            except json.JSONDecodeError:
                pass
        
        # Fallback: extract code blocks
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            description = response.split('```')[0].strip()
            return code, description
        
        # Last resort: return entire response
        return response, "Generated code"

    def _fallback_generation(
        self,
        loss_events: list[LossEvent],
        target_component: str
    ) -> GeneratedCode:
        """Fallback code generation when LLM is unavailable."""
        # Simple heuristic-based generation
        if target_component == "position_sizer":
            avg_loss = abs(sum(e.pnl_pct for e in loss_events if e.pnl_pct < 0) / max(len(loss_events), 1))
            code = f"""
def calculate_position_size(capital, risk_pct):
    # Reduced risk based on recent losses
    adjusted_risk = risk_pct * 0.8  # 20% reduction
    return capital * adjusted_risk / 100.0
"""
            description = f"Reduced position sizing by 20% based on {len(loss_events)} loss events (avg loss: {avg_loss:.2f}%)"
        else:
            code = "# No improvement generated"
            description = "Fallback: No LLM available"
        
        return GeneratedCode(
            code=code,
            description=description,
            target_component=target_component,
            generation_timestamp=datetime.now()
        )


class CodegenEngine:
    """Main code generation engine with shadow testing."""

    def __init__(
        self,
        bus: EventBus,
        accounting: PortfolioAccounting,
        model_path: Optional[str] = None,
        shadow_test_duration_hours: float = 1.0
    ):
        """Initialize code generation engine.
        
        Args:
            bus: Event bus for publishing events
            accounting: Portfolio accounting for accessing loss data
            model_path: Path to LLM model
            shadow_test_duration_hours: Duration of shadow testing before approval
        """
        self._bus = bus
        self._accounting = accounting
        self._sandbox = RestrictedSandbox()
        self._generator = LLMCodeGenerator(model_path)
        self._shadow_test_duration = shadow_test_duration_hours
        
        # Track loss events
        self._loss_events: list[LossEvent] = []
        self._max_loss_events = 100
        
        # Track generated code
        self._generated_code: list[GeneratedCode] = []
        self._shadow_testing_code: dict[str, GeneratedCode] = {}
        self._approved_code: dict[str, Callable] = {}

    async def start(self) -> None:
        """Start the code generation engine."""
        self._bus.subscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        logger.info("Code generation engine started")

    async def stop(self) -> None:
        """Stop the code generation engine."""
        self._bus.unsubscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        logger.info("Code generation engine stopped")

    async def _handle_position_closed(self, event: Event) -> None:
        """Handle position closed events to track losses."""
        position = event.data.get("position")
        if not position:
            return
        
        # Record loss event if position was a loss
        if position.realized_pnl < 0:
            loss_event = LossEvent(
                timestamp=datetime.now(),
                symbol=position.symbol,
                strategy_id=position.strategy_id or "unknown",
                entry_price=position.entry_price,
                exit_price=position.exit_price or position.entry_price,
                pnl=position.realized_pnl,
                pnl_pct=(position.realized_pnl / position.entry_price * position.size) * 100,
                holding_time_seconds=0.0,  # TODO: Calculate from timestamps
                market_conditions={},  # TODO: Extract from regime/volatility
                metadata={}
            )
            
            self._loss_events.append(loss_event)
            if len(self._loss_events) > self._max_loss_events:
                self._loss_events.pop(0)
            
            logger.info(f"Recorded loss event: {loss_event.symbol} - ${loss_event.pnl:.2f}")
            
            # Trigger code generation if threshold reached
            if len(self._loss_events) >= 10:
                await self._analyze_and_generate()

    async def _analyze_and_generate(self) -> None:
        """Analyze loss events and generate new code."""
        if len(self._loss_events) < 10:
            return
        
        logger.info(f"Analyzing {len(self._loss_events)} loss events for code generation...")
        
        # Generate code for different components
        components = ["position_sizer", "warden", "edge_estimator"]
        
        for component in components:
            generated = self._generator.generate_code(
                loss_events=self._loss_events[-20:],  # Last 20 events
                target_component=component
            )
            
            if generated:
                # Validate and test in sandbox
                is_valid, result, error = self._sandbox.execute(
                    generated.code,
                    inputs={"capital": 1000.0, "risk_pct": 2.0}
                )
                
                if is_valid:
                    generated.test_results = {"sandbox_test": "passed", "result": str(result)}
                    self._generated_code.append(generated)
                    
                    # Start shadow testing
                    await self._start_shadow_testing(generated)
                    logger.info(f"Generated code for {component}: {generated.description}")
                else:
                    logger.warning(f"Generated code failed sandbox validation: {error}")

    async def _start_shadow_testing(self, generated_code: GeneratedCode) -> None:
        """Start shadow testing for generated code."""
        generated_code.approval_status = "shadow_testing"
        generated_code.shadow_test_duration_hours = 0.0
        
        test_id = f"{generated_code.target_component}_{int(time.time())}"
        self._shadow_testing_code[test_id] = generated_code
        
        logger.info(f"Started shadow testing for {test_id}: {generated_code.description}")
        
        # Schedule approval check
        await asyncio.sleep(self._shadow_test_duration_hours * 3600)
        
        # After shadow testing period, evaluate performance
        await self._evaluate_shadow_test(test_id)

    async def _evaluate_shadow_test(self, test_id: str) -> None:
        """Evaluate shadow test results and approve/reject code."""
        if test_id not in self._shadow_testing_code:
            return
        
        generated_code = self._shadow_testing_code[test_id]
        
        # TODO: Evaluate performance metrics from shadow testing
        # For now, approve if no errors occurred
        if generated_code.test_results and generated_code.test_results.get("sandbox_test") == "passed":
            generated_code.approval_status = "approved"
            logger.info(f"Approved code: {test_id}")
            
            # Integrate into dynamic logic layer
            await self._integrate_code(generated_code)
        else:
            generated_code.approval_status = "rejected"
            logger.warning(f"Rejected code: {test_id}")
        
        del self._shadow_testing_code[test_id]

    async def _integrate_code(self, generated_code: GeneratedCode) -> None:
        """Integrate approved code into the dynamic logic layer."""
        # This would integrate the code into the appropriate component
        # For now, just store it
        logger.info(f"Integrating approved code into {generated_code.target_component}")
        # TODO: Actual integration logic would go here
