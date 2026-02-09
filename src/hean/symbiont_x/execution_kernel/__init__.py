"""
Execution Microkernel - Ультра-быстрое исполнение

Python wrapper для будущего Rust ядра
"""

from .executor import ExecutionKernel, OrderRequest, OrderResult

__all__ = [
    'ExecutionKernel',
    'OrderRequest',
    'OrderResult',
]
