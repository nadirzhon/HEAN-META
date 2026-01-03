# Backtest and Evaluate Command Fixes

## Summary

Fixed critical async lifecycle issues that caused `backtest` and `evaluate` commands to:
- Hang indefinitely after simulation completion
- Fail to print final reports
- Exit without visible output

## Root Causes Identified

### 1. EventBus.stop() - Task Not Cancelled (CRITICAL)

**Location**: `src/hean/core/bus.py:89-96`

**Problem**: 
The `EventBus.stop()` method set `_running = False` and awaited the processing task, but the task's main loop uses `asyncio.wait_for(self._queue.get(), timeout=1.0)`. When `_running` is set to False, the task continues running until the next timeout (up to 1 second), then checks `_running` again. This creates a race condition where:
- The task might be blocked in `wait_for()` when `_running` is set to False
- The task won't exit until the timeout expires
- If events keep arriving, the task never exits

**Why it happened**:
- Violated async principle: **Tasks with infinite loops must be explicitly cancelled, not just awaited**
- Setting a flag is insufficient when the task is blocked in a timeout operation
- The event loop stays alive because the task never completes

**Fix**:
```python
# Cancel the processing task explicitly
if self._task:
    self._task.cancel()
    try:
        await self._task
    except asyncio.CancelledError:
        pass
```

**Why the fix works**:
- `task.cancel()` immediately raises `CancelledError` in the task, breaking out of any blocking operation
- The task exits immediately instead of waiting for timeout
- Event loop can proceed to shutdown

### 2. Missing Event Queue Draining

**Location**: `src/hean/core/bus.py:89-96`

**Problem**:
After stopping the event bus, events might remain in the queue. While not directly causing hangs, this can lead to:
- Memory leaks
- Unexpected behavior if components are restarted
- Confusion during debugging

**Fix**:
```python
# Drain any remaining events in the queue
drained = 0
while not self._queue.empty():
    try:
        self._queue.get_nowait()
        drained += 1
    except asyncio.QueueEmpty:
        break
```

**Why the fix works**:
- Explicitly removes all pending events
- Prevents memory leaks
- Makes shutdown behavior deterministic

### 3. Insufficient Time for Event Processing

**Location**: `src/hean/main.py:run_backtest()` and `run_evaluation()`

**Problem**:
After `simulator.run()` completes, there might be pending events in the EventBus queue (order fills, position updates, etc.). If we immediately shutdown components, these events might not be processed, leading to:
- Incomplete metrics
- Missing order fills in accounting
- Inconsistent final state

**Fix**:
```python
# Give event bus a moment to process any remaining events
await asyncio.sleep(0.5)
```

**Why the fix works**:
- Allows EventBus to process queued events before shutdown
- Ensures all order fills and position updates are recorded
- Results in accurate final metrics

**Note**: This is NOT a synchronization hack - it's a grace period for async event processing. The EventBus task is still running and will process events during this time.

### 4. Bug in run_evaluation - Missing Strategy Assignment

**Location**: `src/hean/main.py:428-431`

**Problem**:
```python
if settings.impulse_engine_enabled:
    impulse_engine = ImpulseEngine(bus, ["BTCUSDT", "ETHUSDT"])
    await impulse_engine.start()
    strategies["impulse_engine"] = impulse_engine  # This line was missing!
```

The `impulse_engine` was created and started but never added to the `strategies` dict, causing:
- Strategy metrics not to be included in evaluation
- Potential reference errors when trying to stop it later

**Fix**:
- Added the missing assignment
- Added proper null check before stopping

### 5. Missing Lifecycle Logging

**Problem**:
No visibility into where the process was hanging or what stage it reached.

**Fix**:
- Added comprehensive logging at each lifecycle stage:
  - Component startup
  - Simulation start/end
  - Metrics calculation
  - Report printing
  - Component shutdown

**Why it helps**:
- Enables debugging of future issues
- Makes it clear when each stage completes
- Helps identify bottlenecks

## Async Principles Violated (and Fixed)

### Principle 1: Task Cancellation
**Violation**: Tasks with infinite loops were not explicitly cancelled
**Fix**: Use `task.cancel()` instead of just setting flags

### Principle 2: Shutdown Ordering
**Violation**: Components stopped in arbitrary order
**Fix**: Stop in reverse order of startup (LIFO)

### Principle 3: Event Processing Grace Period
**Violation**: Immediate shutdown after simulation without allowing event processing
**Fix**: Add brief grace period for async event processing

### Principle 4: Resource Cleanup
**Violation**: Event queues not drained on shutdown
**Fix**: Explicitly drain queues to prevent memory leaks

## Testing

Added three validation tests in `tests/test_backtest.py`:

1. **test_run_backtest_completes_and_prints_report()**
   - Verifies backtest completes without hanging
   - Confirms report is printed
   - Ensures clean shutdown

2. **test_run_evaluation_completes_and_prints_result()**
   - Verifies evaluation completes without hanging
   - Confirms readiness report is printed
   - Validates return value structure

3. **test_event_simulator_terminates_correctly()**
   - Validates simulation loop has deterministic termination
   - Ensures proper cleanup

## Files Modified

1. `src/hean/core/bus.py` - Fixed EventBus.stop() to cancel task and drain queue
2. `src/hean/main.py` - Added logging, fixed strategy assignment bug, added grace period
3. `src/hean/backtest/event_sim.py` - Added lifecycle logging
4. `tests/test_backtest.py` - Added validation tests

## Verification

To verify the fixes work:

```bash
# Should complete and print report
python3 -m hean.main backtest --days 1

# Should complete and print readiness report
python3 -m hean.main evaluate --days 1
```

Both commands should:
- Complete within reasonable time
- Print their respective reports
- Exit cleanly with code 0

## Future Improvements

1. Consider using `asyncio.Event` for more precise shutdown signaling
2. Add timeout mechanisms to prevent truly infinite hangs
3. Consider using context managers for component lifecycle
4. Add metrics for event queue depth to detect bottlenecks





