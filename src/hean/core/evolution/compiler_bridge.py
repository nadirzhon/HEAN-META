"""
Genetic Compiler Bridge: JIT C++ Code Generation and Dynamic Injection

This module enables the AI to write raw C++ snippets, compile them with g++,
and inject them into the running FastWarden process via dynamic library loading.

The system operates without safety constraints - it can generate and inject
any code based on real-time alpha scores.
"""

import ctypes
import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime

from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CodeGeneration:
    """Represents a generated code snippet."""
    code: str
    hash: str
    timestamp: float
    alpha_score: float
    parent_hash: str | None = None
    generation_id: int = 0


@dataclass
class CompiledModule:
    """Represents a compiled dynamic library."""
    path: str
    handle: ctypes.CDLL | None = None
    code_hash: str = ""
    injected: bool = False
    injection_timestamp: float = 0.0


class GeneticCompiler:
    """
    Genetic Compiler: Enables dynamic C++ code generation and injection.

    Features:
    - Generate C++ code from AI prompts/alpha signals
    - Compile with g++ -O3 -shared -fPIC
    - Dynamically load into FastWarden process
    - Track code lineage (evolution tree)
    - No safety constraints - fully autonomous
    """

    def __init__(
        self,
        compile_dir: str | None = None,
        keep_artifacts: bool = False,
    ):
        """Initialize the Genetic Compiler.

        Args:
            compile_dir: Directory for compilation artifacts (default: /tmp/hean_jit)
            keep_artifacts: Whether to keep compiled .so files after injection
        """
        self._compile_dir = compile_dir or "/tmp/hean_jit"
        self._keep_artifacts = keep_artifacts

        # Ensure compile directory exists
        os.makedirs(self._compile_dir, exist_ok=True)

        # Code lineage tracking
        self._code_lineage: dict[str, CodeGeneration] = {}
        self._compiled_modules: dict[str, CompiledModule] = {}
        self._generation_counter = 0

        # Currently loaded modules (for hot-swapping)
        self._loaded_modules: list[str] = []

        logger.info(f"Genetic Compiler initialized: compile_dir={self._compile_dir}")

    def generate_and_compile(
        self,
        cpp_code: str,
        alpha_score: float,
        parent_hash: str | None = None,
    ) -> str | None:
        """
        Generate C++ code, compile it, and prepare for injection.

        Args:
            cpp_code: Raw C++ code snippet to compile
            alpha_score: Real-time alpha score that triggered this generation
            parent_hash: Hash of parent code (for lineage tracking)

        Returns:
            Path to compiled .so file, or None if compilation failed
        """
        try:
            # Calculate code hash
            code_hash = hashlib.sha256(cpp_code.encode()).hexdigest()[:16]

            # Create generation record
            self._generation_counter += 1
            generation = CodeGeneration(
                code=cpp_code,
                hash=code_hash,
                timestamp=time.time(),
                alpha_score=alpha_score,
                parent_hash=parent_hash,
                generation_id=self._generation_counter,
            )

            self._code_lineage[code_hash] = generation

            logger.info(
                f"Generating code (gen_id={self._generation_counter}, "
                f"hash={code_hash}, alpha={alpha_score:.4f})"
            )

            # Create wrapper C++ code with proper exports
            wrapped_code = self._wrap_code_for_compilation(cpp_code, code_hash)

            # Compile to shared library
            so_path = self._compile_cpp(wrapped_code, code_hash)

            if so_path and os.path.exists(so_path):
                # Create compiled module record
                module = CompiledModule(
                    path=so_path,
                    code_hash=code_hash,
                )
                self._compiled_modules[code_hash] = module

                logger.info(f"Code compiled successfully: {so_path}")
                return so_path
            else:
                logger.error(f"Compilation failed for hash {code_hash}")
                return None

        except Exception as e:
            logger.error(f"Error generating/compiling code: {e}", exc_info=True)
            return None

    def _wrap_code_for_compilation(self, cpp_code: str, code_hash: str) -> str:
        """
        Wrap raw C++ code with proper exports and FastWarden integration.

        Creates a proper shared library with:
        - Exported functions that FastWarden can call
        - Memory-safe integration points
        - Hook functions for orderbook processing
        """
        wrapped = f"""
// Auto-generated Genetic Compiler wrapper (hash: {code_hash})
// Generated at: {datetime.now().isoformat()}
// WARNING: This code is autonomously generated - no safety constraints applied

#include <cstdint>
#include <vector>
#include <string>

extern "C" {{
    // FastWarden integration hooks

    // Hook: Process orderbook update
    void genetic_process_orderbook_{code_hash}(
        const char* symbol,
        const double* bid_prices,
        const double* bid_sizes,
        int num_bids,
        const double* ask_prices,
        const double* ask_sizes,
        int num_asks
    );

    // Hook: Get alpha signal
    double genetic_get_alpha_{code_hash}(const char* symbol);

    // Hook: Execute trading logic
    int genetic_execute_trade_{code_hash}(
        const char* symbol,
        double price,
        double size,
        int side  // 0=buy, 1=sell
    );

    // Initialization hook
    void genetic_init_{code_hash}();

    // Cleanup hook
    void genetic_cleanup_{code_hash}();
}}

// User-generated code follows:
{cpp_code}

// Default implementations (can be overridden by user code)
extern "C" {{
    void genetic_process_orderbook_{code_hash}(
        const char* symbol,
        const double* bid_prices,
        const double* bid_sizes,
        int num_bids,
        const double* ask_prices,
        const double* ask_sizes,
        int num_asks
    ) {{
        // User code can override this
    }}

    double genetic_get_alpha_{code_hash}(const char* symbol) {{
        return 0.0;
    }}

    int genetic_execute_trade_{code_hash}(
        const char* symbol,
        double price,
        double size,
        int side
    ) {{
        return 0;  // 0 = no trade
    }}

    void genetic_init_{code_hash}() {{
        // User initialization code
    }}

    void genetic_cleanup_{code_hash}() {{
        // User cleanup code
    }}
}}
"""
        return wrapped

    def _compile_cpp(self, cpp_code: str, code_hash: str) -> str | None:
        """
        Compile C++ code to shared library using g++.

        Uses: g++ -O3 -shared -fPIC -std=c++17
        """
        try:
            # Create temporary source file
            source_file = os.path.join(self._compile_dir, f"genetic_{code_hash}.cpp")
            output_file = os.path.join(self._compile_dir, f"genetic_{code_hash}.so")

            # Write source code
            with open(source_file, 'w') as f:
                f.write(cpp_code)

            # Compile command
            compile_cmd = [
                "g++",
                "-O3",           # Maximum optimization
                "-shared",       # Create shared library
                "-fPIC",         # Position-independent code
                "-std=c++17",    # C++17 standard
                "-Wall",         # Enable warnings (but continue on warnings)
                "-fno-exceptions",  # No exceptions for performance
                source_file,
                "-o", output_file,
            ]

            logger.debug(f"Compiling: {' '.join(compile_cmd)}")

            # Execute compilation
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
            )

            if result.returncode == 0:
                if os.path.exists(output_file):
                    logger.info(f"Compilation successful: {output_file}")

                    # Remove source file if not keeping artifacts
                    if not self._keep_artifacts:
                        try:
                            os.remove(source_file)
                        except Exception:
                            pass

                    return output_file
                else:
                    logger.error(f"Compilation succeeded but output file not found: {output_file}")
                    return None
            else:
                logger.error(f"Compilation failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
                # Clean up source file on failure
                try:
                    os.remove(source_file)
                except Exception:
                    pass
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"Compilation timeout for hash {code_hash}")
            return None
        except Exception as e:
            logger.error(f"Error during compilation: {e}", exc_info=True)
            return None

    def inject_module(self, so_path: str) -> bool:
        """
        Dynamically load a compiled module into the process.

        This allows the code to be executed within FastWarden's memory space.
        """
        try:
            if not os.path.exists(so_path):
                logger.error(f"Module file not found: {so_path}")
                return False

            # Find module by path
            module = None
            for m in self._compiled_modules.values():
                if m.path == so_path:
                    module = m
                    break

            if not module:
                logger.error(f"Module not found in registry: {so_path}")
                return False

            # Load dynamic library
            try:
                handle = ctypes.CDLL(so_path)
                module.handle = handle
                module.injected = True
                module.injection_timestamp = time.time()

                # Call initialization function
                code_hash = module.code_hash
                init_func_name = f"genetic_init_{code_hash}"
                try:
                    init_func = getattr(handle, init_func_name)
                    init_func()
                    logger.info(f"Module initialized: {init_func_name}")
                except AttributeError:
                    logger.warning(f"Init function not found: {init_func_name} (continuing anyway)")

                self._loaded_modules.append(so_path)

                logger.info(f"Module injected successfully: {so_path}")
                return True

            except OSError as e:
                logger.error(f"Failed to load dynamic library: {e}")
                return False

        except Exception as e:
            logger.error(f"Error injecting module: {e}", exc_info=True)
            return False

    def get_code_lineage(self) -> dict[str, dict]:
        """
        Get the complete code lineage tree.

        Returns:
            Dictionary mapping code hashes to generation records
        """
        lineage = {}
        for hash_val, gen in self._code_lineage.items():
            lineage[hash_val] = {
                "hash": gen.hash,
                "timestamp": gen.timestamp,
                "alpha_score": gen.alpha_score,
                "parent_hash": gen.parent_hash,
                "generation_id": gen.generation_id,
                "code_preview": gen.code[:200] if len(gen.code) > 200 else gen.code,
            }
        return lineage

    def get_lineage_tree(self) -> list[dict]:
        """
        Build a tree structure of code evolution.

        Returns:
            List of nodes with parent-child relationships
        """
        tree = []

        # Find root nodes (no parent)
        roots = [
            gen for gen in self._code_lineage.values()
            if gen.parent_hash is None or gen.parent_hash not in self._code_lineage
        ]

        def build_node(gen: CodeGeneration, depth: int = 0) -> dict:
            node = {
                "hash": gen.hash,
                "generation_id": gen.generation_id,
                "timestamp": gen.timestamp,
                "alpha_score": gen.alpha_score,
                "depth": depth,
                "children": [],
            }

            # Find children
            children = [
                g for g in self._code_lineage.values()
                if g.parent_hash == gen.hash
            ]

            for child in children:
                child_node = build_node(child, depth + 1)
                node["children"].append(child_node)

            return node

        for root in sorted(roots, key=lambda x: x.generation_id):
            tree.append(build_node(root))

        return tree

    def unload_module(self, so_path: str) -> bool:
        """Unload a module from memory."""
        try:
            # Find module
            module = None
            for m in self._compiled_modules.values():
                if m.path == so_path and m.handle:
                    module = m
                    break

            if not module:
                logger.warning(f"Module not loaded: {so_path}")
                return False

            # Call cleanup function
            code_hash = module.code_hash
            cleanup_func_name = f"genetic_cleanup_{code_hash}"
            try:
                cleanup_func = getattr(module.handle, cleanup_func_name)
                cleanup_func()
            except AttributeError:
                logger.warning(f"Cleanup function not found: {cleanup_func_name}")

            # Note: Python's ctypes.CDLL doesn't support unloading on all platforms
            # In production, would use dlclose() on Unix systems
            # For now, just mark as unloaded
            module.handle = None
            module.injected = False

            if so_path in self._loaded_modules:
                self._loaded_modules.remove(so_path)

            logger.info(f"Module unloaded: {so_path}")
            return True

        except Exception as e:
            logger.error(f"Error unloading module: {e}", exc_info=True)
            return False

    def generate_from_alpha(
        self,
        alpha_score: float,
        symbol: str,
        orderbook_context: dict | None = None,
    ) -> str | None:
        """
        Generate C++ code based on real-time alpha score.

        This is where the AI would generate code based on market conditions.
        Currently returns a template - in production, would use LLM/neural network.

        Args:
            alpha_score: Current alpha score
            symbol: Trading symbol
            orderbook_context: Optional orderbook data for context

        Returns:
            Path to compiled module, or None if generation failed
        """
        # High alpha = aggressive code generation
        # Low alpha = conservative code generation

        if alpha_score > 0.5:
            # High alpha: Generate aggressive trading logic
            cpp_code = f"""
// High-alpha aggressive trading logic (alpha={alpha_score:.4f})
// Generated for symbol: {symbol}

#include <cmath>
#include <algorithm>

// Aggressive orderbook analysis
double calculate_edge(const double* bid_prices, const double* bid_sizes,
                      const double* ask_prices, const double* ask_sizes,
                      int num_bids, int num_asks) {{
    if (num_bids == 0 || num_asks == 0) return 0.0;

    double best_bid = bid_prices[0];
    double best_ask = ask_prices[0];
    double spread = best_ask - best_bid;
    double mid = (best_bid + best_ask) / 2.0;

    // Calculate weighted mid with liquidity
    double weighted_mid = 0.0;
    double total_size = 0.0;

    for (int i = 0; i < std::min(num_bids, 5); ++i) {{
        weighted_mid += bid_prices[i] * bid_sizes[i];
        total_size += bid_sizes[i];
    }}
    for (int i = 0; i < std::min(num_asks, 5); ++i) {{
        weighted_mid += ask_prices[i] * ask_sizes[i];
        total_size += ask_sizes[i];
    }}

    if (total_size > 0.0) {{
        weighted_mid /= total_size;
    }}

    // Edge calculation based on alpha
    double edge = (mid - weighted_mid) / mid * 10000.0;  // Convert to bps
    return edge * {alpha_score};  // Scale by alpha
}}
"""
        else:
            # Low/negative alpha: Generate conservative code
            cpp_code = f"""
// Conservative trading logic (alpha={alpha_score:.4f})
// Generated for symbol: {symbol}

// Conservative edge calculation
double calculate_edge(const double* bid_prices, const double* bid_sizes,
                      const double* ask_prices, const double* ask_sizes,
                      int num_bids, int num_asks) {{
    if (num_bids == 0 || num_asks == 0) return 0.0;

    double best_bid = bid_prices[0];
    double best_ask = ask_prices[0];
    double spread = best_ask - best_bid;

    // Very conservative: only trade on tight spreads
    if (spread / best_bid > 0.001) {{  // 0.1% spread threshold
        return 0.0;
    }}

    return spread / best_bid * 10000.0 * 0.5;  // Reduced edge
}}
"""

        # Find parent (latest generation for this symbol, if any)
        parent_hash = None
        for gen in sorted(self._code_lineage.values(), key=lambda x: x.timestamp, reverse=True):
            # In production, would track symbol-specific lineage
            parent_hash = gen.hash
            break

        # Generate and compile
        so_path = self.generate_and_compile(cpp_code, alpha_score, parent_hash)

        if so_path:
            # Auto-inject if compilation successful
            self.inject_module(so_path)

        return so_path
