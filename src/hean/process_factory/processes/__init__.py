"""Built-in process definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hean.process_factory.schemas import ProcessDefinition

_BUILTIN_MODULES = {
    "p1_capital_parking": ".p1_capital_parking",
    "p2_funding_monitor": ".p2_funding_monitor",
    "p3_fee_monitor": ".p3_fee_monitor",
    "p4_campaign_checklist": ".p4_campaign_checklist",
    "p5_execution_optimizer": ".p5_execution_optimizer",
    "p6_opportunity_scanner": ".p6_opportunity_scanner",
}


def list_builtin_processes() -> list[str]:
    """Return IDs of all built-in process definitions."""
    return list(_BUILTIN_MODULES.keys())


def get_process_definition(process_id: str) -> ProcessDefinition:
    """Load and return a built-in ProcessDefinition by ID.

    Args:
        process_id: One of the IDs from :func:`list_builtin_processes`.

    Raises:
        KeyError: If *process_id* is not a known built-in process.
    """
    if process_id not in _BUILTIN_MODULES:
        raise KeyError(
            f"Unknown built-in process {process_id!r}. "
            f"Available: {list_builtin_processes()}"
        )
    import importlib

    module = importlib.import_module(_BUILTIN_MODULES[process_id], package=__name__)
    return module.get_process_definition()


__all__ = [
    "get_process_definition",
    "list_builtin_processes",
]
