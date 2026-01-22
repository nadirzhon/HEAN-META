"""
absolute_plus.py

Main integration system for HEAN Absolute+.
Synthesizes MetamorphicCore (C++) and CausalBrain (Python) into a unified
autonomous economic engine.
"""

import asyncio
import logging
import signal
import time

from .core.python.causal_brain import CausalBrain
from .exchange.bybit_tensorized import BybitTensorizedMonitor
from .ui.evolution_terminal import EvolutionTerminal

# C++ binding (will be loaded via ctypes or pybind11)
try:
    import ctypes
    import os

    # Try to load the C++ MetamorphicCore library
    lib_path = os.path.join(os.path.dirname(__file__), "..", "..", "build", "libmetamorphic.so")
    if os.path.exists(lib_path):
        lib_metamorphic = ctypes.CDLL(lib_path)
    else:
        lib_metamorphic = None
        logging.warning("MetamorphicCore C++ library not found. Using Python-only mode.")
except Exception as e:
    lib_metamorphic = None
    logging.warning(f"Could not load MetamorphicCore C++ library: {e}")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AbsolutePlusSystem:
    """
    The HEAN Absolute+ autonomous economic engine.
    Integrates MetamorphicCore (C++) and CausalBrain (Python) for
    self-aware, adaptive trading.
    """

    def __init__(self,
                 api_key: str | None = None,
                 api_secret: str | None = None,
                 initial_capital: float = 300.0,
                 enable_ui: bool = True):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.enable_ui = enable_ui

        # Core components
        self.causal_brain: CausalBrain | None = None
        self.market_monitor: BybitTensorizedMonitor | None = None
        self.evolution_terminal: EvolutionTerminal | None = None

        # C++ MetamorphicCore handle (if available)
        self.metamorphic_core_handle = None

        # System state
        self.running = False
        self.main_task: asyncio.Task | None = None

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal")
        asyncio.create_task(self.shutdown())

    def initialize_cpp_core(self) -> bool:
        """Initialize C++ MetamorphicCore via ctypes"""
        if not lib_metamorphic:
            logger.warning("C++ MetamorphicCore not available, using Python-only mode")
            return False

        try:
            # Define function signatures
            lib_metamorphic.metamorphic_core_new.argtypes = []
            lib_metamorphic.metamorphic_core_new.restype = ctypes.c_void_p

            lib_metamorphic.metamorphic_core_initialize.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_char_p
            ]
            lib_metamorphic.metamorphic_core_initialize.restype = ctypes.c_bool

            lib_metamorphic.metamorphic_core_start_receiver.argtypes = [ctypes.c_void_p]
            lib_metamorphic.metamorphic_core_start_receiver.restype = None

            # Create core instance
            self.metamorphic_core_handle = lib_metamorphic.metamorphic_core_new()

            # Initialize with ZeroMQ
            endpoint = b"ipc:///tmp/hean_metamorphic"
            connection_type = b"zmq"
            result = lib_metamorphic.metamorphic_core_initialize(
                self.metamorphic_core_handle,
                connection_type,
                endpoint
            )

            if result:
                lib_metamorphic.metamorphic_core_start_receiver(self.metamorphic_core_handle)
                logger.info("C++ MetamorphicCore initialized successfully")
                return True
            else:
                logger.error("Failed to initialize C++ MetamorphicCore")
                return False
        except Exception as e:
            logger.error(f"Error initializing C++ MetamorphicCore: {e}")
            return False

    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing HEAN Absolute+ System...")

        # Initialize CausalBrain (Python)
        logger.info("Initializing CausalBrain...")
        self.causal_brain = CausalBrain(zmq_endpoint="ipc:///tmp/hean_metamorphic")
        if not self.causal_brain.initialize():
            logger.error("Failed to initialize CausalBrain")
            return False

        # Start CausalBrain mutation signal generator
        if not self.causal_brain.start():
            logger.error("Failed to start CausalBrain")
            return False

        logger.info("CausalBrain initialized and running")

        # Initialize C++ MetamorphicCore
        self.initialize_cpp_core()

        # Initialize Bybit Tensorized Monitor
        logger.info("Initializing Bybit Tensorized Monitor...")
        self.market_monitor = BybitTensorizedMonitor(
            api_key=None,  # Use public endpoints for now
            api_secret=None,
            causal_brain=self.causal_brain
        )

        # Add proxy nodes (example - user should configure actual proxies)
        # self.market_monitor.add_proxy_node("us-east", "https://proxy1.example.com", "us-east")
        # self.market_monitor.add_proxy_node("eu-west", "https://proxy2.example.com", "eu-west")
        # self.market_monitor.add_proxy_node("asia-pacific", "https://proxy3.example.com", "asia")

        # Initialize Evolution Terminal UI
        if self.enable_ui:
            logger.info("Initializing Evolution Terminal...")
            self.evolution_terminal = EvolutionTerminal(width=1920, height=1080)

        logger.info("HEAN Absolute+ System initialized successfully")
        return True

    async def _update_ui(self):
        """Update UI with latest data"""
        if not self.evolution_terminal or not self.causal_brain:
            return

        # Update SIQ
        siq = self.causal_brain.get_siq()
        learning_stats = self.causal_brain.get_learning_stats()
        self.evolution_terminal.update_siq(siq, learning_stats)

        # Update causal web if we have tensor data
        if self.market_monitor:
            tensor = self.market_monitor.get_tensor()
            if tensor:
                correlation_matrix = tensor.get_correlation_matrix()
                self.evolution_terminal.update_causal_web(
                    tensor.symbols,
                    correlation_matrix
                )

    async def _main_loop(self):
        """Main system loop"""
        logger.info("Starting HEAN Absolute+ main loop...")

        # Start market monitoring
        if self.market_monitor:
            await self.market_monitor.start()

        # Start UI in separate task if enabled
        if self.enable_ui and self.evolution_terminal:
            asyncio.create_task(self.evolution_terminal.run())

        # Main loop
        last_update = time.time()
        ui_update_interval = 0.1  # Update UI every 100ms

        while self.running:
            try:
                # Update UI periodically
                now = time.time()
                if now - last_update >= ui_update_interval:
                    await self._update_ui()
                    last_update = now

                # Print system status periodically
                if int(now) % 10 == 0:
                    self._print_status()

                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)

        logger.info("Main loop stopped")

    def _print_status(self):
        """Print system status to console"""
        if not self.causal_brain:
            return

        stats = self.causal_brain.get_learning_stats()
        siq = stats.get("siq", 0.0)

        if self.market_monitor:
            monitor_stats = self.market_monitor.get_stats()
            num_symbols = monitor_stats.get("num_symbols", 0)
            updates_per_sec = monitor_stats.get("updates_per_second", 0.0)

            logger.info(
                f"Status | SIQ: {siq:.1f} | Symbols: {num_symbols} | "
                f"Updates/s: {updates_per_sec:.1f} | "
                f"Capital: ${self.current_capital:.2f} | "
                f"PnL: ${self.total_pnl:.2f}"
            )

    async def start(self):
        """Start the Absolute+ system"""
        if self.running:
            logger.warning("System is already running")
            return

        # Initialize all components
        if not await self.initialize():
            logger.error("Failed to initialize system")
            return False

        self.running = True

        # Start main loop
        self.main_task = asyncio.create_task(self._main_loop())

        logger.info("HEAN Absolute+ System started")
        logger.info("=" * 60)
        logger.info("System is now autonomous and self-evolving")
        logger.info(f"Initial Capital: ${self.initial_capital:.2f}")
        logger.info("=" * 60)

        return True

    async def shutdown(self):
        """Shutdown the system gracefully"""
        logger.info("Shutting down HEAN Absolute+ System...")

        self.running = False

        # Stop market monitor
        if self.market_monitor:
            await self.market_monitor.stop()

        # Stop CausalBrain
        if self.causal_brain:
            self.causal_brain.stop()

        # Stop UI
        if self.evolution_terminal:
            self.evolution_terminal.stop()

        # Cancel main task
        if self.main_task:
            self.main_task.cancel()
            try:
                await self.main_task
            except asyncio.CancelledError:
                pass

        logger.info("Shutdown complete")

    def run(self):
        """Run the system (blocking)"""
        loop = asyncio.get_event_loop()

        try:
            loop.run_until_complete(self.start())

            # Keep running until interrupted
            if self.main_task:
                loop.run_until_complete(self.main_task)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            loop.run_until_complete(self.shutdown())
            loop.close()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="HEAN Absolute+ Autonomous Economic Engine")
    parser.add_argument("--api-key", type=str, help="Bybit API key (optional)")
    parser.add_argument("--api-secret", type=str, help="Bybit API secret (optional)")
    parser.add_argument("--capital", type=float, default=300.0, help="Initial capital")
    parser.add_argument("--no-ui", action="store_true", help="Disable UI")

    args = parser.parse_args()

    # Create and run system
    system = AbsolutePlusSystem(
        api_key=args.api_key,
        api_secret=args.api_secret,
        initial_capital=args.capital,
        enable_ui=not args.no_ui
    )

    system.run()


if __name__ == "__main__":
    main()
