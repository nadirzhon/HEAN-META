"""Directive types for ARCHON → component communication."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DirectiveType(str, Enum):
    """Types of directives ARCHON can issue."""

    ACTIVATE_STRATEGY = "activate_strategy"
    DEACTIVATE_STRATEGY = "deactivate_strategy"
    UPDATE_STRATEGY_PARAMS = "update_strategy_params"
    SET_RISK_MODE = "set_risk_mode"
    QUARANTINE_SYMBOL = "quarantine_symbol"
    INITIATE_RECONCILIATION = "initiate_reconciliation"
    TRIGGER_EVOLUTION_CYCLE = "trigger_evolution_cycle"
    PAUSE_TRADING = "pause_trading"
    RESUME_TRADING = "resume_trading"


@dataclass
class Directive:
    """Command from ARCHON to a component."""

    directive_type: DirectiveType
    target_component: str
    params: dict[str, Any] = field(default_factory=dict)
    directive_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    issued_at: datetime = field(default_factory=datetime.utcnow)
    requires_ack: bool = True


@dataclass
class DirectiveAck:
    """Acknowledgment of a directive."""

    directive_id: str
    component_id: str
    success: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    executed_at: datetime = field(default_factory=datetime.utcnow)
