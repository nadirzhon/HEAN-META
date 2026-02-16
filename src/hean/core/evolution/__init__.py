"""Evolution module — dynamic C++ code generation, compilation, and injection."""

from .compiler_bridge import CodeGeneration, CompiledModule, GeneticCompiler

__all__ = ["CodeGeneration", "CompiledModule", "GeneticCompiler"]
