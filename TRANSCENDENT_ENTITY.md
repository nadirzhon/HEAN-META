# Transcendent Entity Interface

A minimalist, high-performance HTML5/WebGL dashboard for the HEAN system.

## Overview

The Transcendent Entity interface provides a real-time visualization of the AI system's state, displaying:

- **Extraction Capacity**: Current balance ($300)
- **Extraction Delta**: Real-time PnL
- **Logic State**: Current system state (EVOLVING, OBSERVING, EXTRACTING)
- **The Singularity**: 3D visualization with pulse speed correlating to market volatility and glow intensity correlating to AI confidence
- **Thought Stream**: Force-directed graph showing the Causal Mesh connections

## Design Philosophy

- **Terminology**: Uses "Extraction", "Causality", "Entropy", and "Evolution" instead of traditional trading terms
- **Aesthetic**: Pitch black background with subtle noise texture, floating translucent panels
- **Performance**: Real-time WebSocket updates for instant data binding

## Architecture

### Components

1. **TranscendentEntityServer** (`src/hean/ui/transcendent_entity_server.py`)
   - Python WebSocket server using aiohttp
   - Serves the HTML dashboard
   - Handles WebSocket connections for real-time data

2. **TranscendentEntityBridge** (`src/hean/ui/transcendent_entity_bridge.py`)
   - Bridges the trading system to the dashboard server
   - Extracts data from PortfolioAccounting, RegimeDetector, and CausalMesh
   - Updates the server with real-time data

3. **Dashboard** (`web/transcendent_entity.html`)
   - HTML5/WebGL dashboard
   - Three.js for 3D Singularity visualization
   - Canvas-based force-directed graph for Causal Mesh
   - WebSocket client for real-time updates

## Usage

### Standalone Server (Testing)

```bash
# Run the server standalone for testing
python -m hean.ui.transcendent_entity_server
```

Then open http://localhost:8888 in your browser.

### Integration with Trading System

To integrate with the main trading system, add to `src/hean/main.py`:

```python
from hean.ui.transcendent_entity_server import TranscendentEntityServer
from hean.ui.transcendent_entity_bridge import TranscendentEntityBridge

# In TradingSystem.__init__ or start():
self._transcendent_server = TranscendentEntityServer(port=8888)
self._transcendent_bridge = TranscendentEntityBridge(
    server=self._transcendent_server,
    accounting=self._accounting,
    regime_detector=self._regime_detector,
    causal_mesh=self._causal_mesh,  # If available
)

await self._transcendent_server.start()
await self._transcendent_bridge.start()
```

## Features

### The Singularity

- **3D Object**: Icosahedron in the center
- **Pulse Speed**: Correlates with market volatility (higher volatility = faster pulse)
- **Glow Intensity**: Correlates with AI confidence (higher confidence = brighter glow)
- **Rotation**: Continuous rotation for visual interest

### Thought Stream (Causal Mesh)

- **Force-Directed Graph**: Shows causal connections between assets
- **Real-Time Updates**: Updates as new causal links are discovered
- **Interactive**: Nodes represent assets, edges represent causal influences

### Data Panels

- **Extraction Capacity**: Current equity/balance
- **Extraction Delta**: Real-time PnL (green for positive, red for negative)
- **Logic State**: Current system state (EVOLVING, OBSERVING, EXTRACTING, STANDBY)

## Data Flow

1. Trading system components (PortfolioAccounting, RegimeDetector, CausalMesh) update their state
2. TranscendentEntityBridge periodically polls these components (every 0.5 seconds)
3. Bridge extracts relevant data and formats it for the dashboard
4. Bridge sends updates to TranscendentEntityServer
5. Server broadcasts updates to all connected WebSocket clients
6. Dashboard receives updates and animates visualizations

## Configuration

### Port

Default port is 8888. Change by passing `port` parameter to `TranscendentEntityServer`:

```python
server = TranscendentEntityServer(port=8889)
```

### Update Interval

Default update interval is 0.5 seconds. Change by passing `update_interval` to `TranscendentEntityBridge`:

```python
bridge = TranscendentEntityBridge(
    server=server,
    update_interval=0.2,  # Update every 200ms
)
```

## Technical Details

### WebSocket Protocol

The server uses JSON messages over WebSocket:

```json
{
  "balance": 300.0,
  "pnl": 5.23,
  "volatility": 0.65,
  "confidence": 0.82,
  "logic_state": "EXTRACTING",
  "nodes": [
    {"symbol": "BTC", "x": 100, "y": 150, "influence": 0.9}
  ],
  "edges": [
    {"source": "BTC", "target": "ETH", "strength": 0.8, "lag_ms": 100}
  ]
}
```

### Performance

- **Update Rate**: 2 Hz (every 0.5 seconds) by default
- **WebSocket**: Low-latency JSON messages
- **Rendering**: 60 FPS for smooth animations

## Future Enhancements

- Binary WebSocket protocol for lower latency
- C++ uWebSockets server for maximum performance
- Additional visualizations (equity curve, strategy performance)
- Interactive controls for system parameters

## Notes

- The C++ uWebSockets server (`TranscendentEntityServer.h/cpp`) is provided but requires uWebSockets integration to compile
- The Python server is fully functional and suitable for most use cases
- The dashboard uses CDN-hosted Three.js (can be changed to local copy if needed)
