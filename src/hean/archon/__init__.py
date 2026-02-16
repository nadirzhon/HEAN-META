"""ARCHON — Autonomous Runtime Controller & Holistic Orchestration Nexus.

Central brain-orchestrator for HEAN trading system.
"""

try:
    from hean.archon.archon import Archon
except ImportError:
    Archon = None  # type: ignore[assignment,misc]

try:
    from hean.archon.chronicle import Chronicle
except ImportError:
    Chronicle = None  # type: ignore[assignment,misc]

try:
    from hean.archon.cortex import Cortex
except ImportError:
    Cortex = None  # type: ignore[assignment,misc]

try:
    from hean.archon.directives import Directive, DirectiveAck, DirectiveType
except ImportError:
    Directive = DirectiveAck = DirectiveType = None  # type: ignore[assignment,misc]

try:
    from hean.archon.genome_director import GenomeDirector
except ImportError:
    GenomeDirector = None  # type: ignore[assignment,misc]

try:
    from hean.archon.health_matrix import HealthMatrix
except ImportError:
    HealthMatrix = None  # type: ignore[assignment,misc]

try:
    from hean.archon.heartbeat import HeartbeatRegistry
except ImportError:
    HeartbeatRegistry = None  # type: ignore[assignment,misc]

try:
    from hean.archon.protocols import ArchonComponent, ComponentState
except ImportError:
    ArchonComponent = ComponentState = None  # type: ignore[assignment,misc]

try:
    from hean.archon.reconciler import ArchonReconciler
except ImportError:
    ArchonReconciler = None  # type: ignore[assignment,misc]

try:
    from hean.archon.signal_pipeline import SignalStage, SignalTrace, StageRecord
except ImportError:
    SignalStage = SignalTrace = StageRecord = None  # type: ignore[assignment,misc]

try:
    from hean.archon.signal_pipeline_manager import DeadLetterQueue, SignalPipelineManager
except ImportError:
    DeadLetterQueue = SignalPipelineManager = None  # type: ignore[assignment,misc]

__all__ = [
    "Archon",
    "ArchonComponent",
    "ArchonReconciler",
    "Chronicle",
    "ComponentState",
    "Cortex",
    "DeadLetterQueue",
    "Directive",
    "DirectiveAck",
    "DirectiveType",
    "GenomeDirector",
    "HealthMatrix",
    "HeartbeatRegistry",
    "SignalPipelineManager",
    "SignalStage",
    "SignalTrace",
    "StageRecord",
]
