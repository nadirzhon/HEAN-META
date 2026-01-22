# HEAN Assumptions and Design Decisions

This document captures key assumptions and design decisions made during the production-ready implementation.

## Architecture Assumptions

### 1. Engine Facade Pattern

**Assumption**: `EngineFacade` provides unified orchestration for both `TradingSystem` and `ProcessFactory`.

**Rationale**: 
- Maintains backward compatibility with existing CLI
- Provides single point of entry for API
- Allows future extension without breaking changes

**Trade-offs**:
- Adds abstraction layer (slight overhead)
- Requires careful state management

### 2. Process Factory as Extension

**Assumption**: `ProcessFactory` remains experimental and optional, `TradingSystem` is the primary engine.

**Rationale**:
- ProcessFactory is marked as experimental in codebase
- TradingSystem has more mature implementation
- Allows gradual migration if ProcessFactory becomes primary

**Trade-offs**:
- May need refactoring if ProcessFactory becomes primary
- Some duplication between systems

### 3. Reconcile Service

**Assumption**: Reconcile runs on-demand via API, not as background task.

**Rationale**:
- Simpler implementation
- Avoids race conditions
- User controls when reconcile happens

**Trade-offs**:
- Requires explicit API calls
- May miss discrepancies if not called regularly

**Future Enhancement**: Add periodic background reconcile task.

## API Design Assumptions

### 1. No Authentication (Current)

**Assumption**: API is unauthenticated in current implementation.

**Rationale**:
- Faster development
- Can be added later via middleware
- Suitable for local/development use

**Trade-offs**:
- Not production-ready for exposed APIs
- Requires authentication middleware for production

**Future Enhancement**: Add JWT or API key authentication.

### 2. CORS Enabled for All Origins

**Assumption**: CORS allows all origins (`*`).

**Rationale**:
- Simplifies development
- Dashboard needs API access
- Can be restricted in production

**Trade-offs**:
- Security risk if exposed publicly
- Should restrict in production

**Future Enhancement**: Configurable CORS origins.

### 3. Synchronous Engine Control

**Assumption**: Engine start/stop is synchronous.

**Rationale**:
- Simpler API design
- Immediate feedback
- Sufficient for current use case

**Trade-offs**:
- May block on slow initialization
- Could timeout on very slow starts

**Future Enhancement**: Async engine control with status polling.

## Frontend Assumptions

### 1. Simple Polling

**Assumption**: Dashboard uses polling (5s interval) instead of WebSocket/SSE.

**Rationale**:
- Simpler implementation
- Works with any HTTP client
- Sufficient for current needs

**Trade-offs**:
- Higher latency
- More server load
- Not real-time

**Current**: WebSocket used for real-time updates; polling remains for dashboard metrics.

### 2. Next.js Proxy/Rewrites

**Assumption**: Frontend connects directly to API/WS via configured public URLs, with optional Next.js rewrites.

**Rationale**:
- Avoids CORS issues
- Single origin for frontend
- Standard pattern

**Trade-offs**:
- Requires correct public URLs in env
- Rebuild needed if public endpoints change

## Observability Assumptions

### 1. Request ID in Context

**Assumption**: Request ID stored in Python `contextvars`, not thread-local.

**Rationale**:
- Works with async code
- Proper context propagation
- Standard Python pattern

**Trade-offs**:
- Requires context setup in all async paths
- May not propagate to threads

### 2. Prometheus Metrics Format

**Assumption**: Metrics exported in Prometheus text format.

**Rationale**:
- Standard format
- Works with existing Prometheus setup
- Easy to scrape

**Trade-offs**:
- Text format is verbose
- No structured metadata

## Safety Assumptions

### 1. Paper Trading Default

**Assumption**: System defaults to paper trading, requires explicit flags for live.

**Rationale**:
- Prevents accidental live trading
- Multiple gates (LIVE_CONFIRM, confirm_phrase, DRY_RUN)
- Industry best practice

**Trade-offs**:
- May be inconvenient for frequent live trading
- Requires multiple steps to enable

### 2. Reconcile in Live Mode Only

**Assumption**: Reconcile only runs in live mode.

**Rationale**:
- Paper mode doesn't need reconcile
- Saves API calls
- Simpler logic

**Trade-offs**:
- Can't verify paper mode accuracy
- May hide bugs in paper mode

## Testing Assumptions

### 1. TestClient for API Tests

**Assumption**: Use FastAPI `TestClient` for API tests.

**Rationale**:
- Fast execution
- No network overhead
- Standard FastAPI pattern

**Trade-offs**:
- Doesn't test network layer
- May miss async issues

**Future Enhancement**: Add integration tests with real HTTP server.

### 2. E2E Smoke Test

**Assumption**: E2E test verifies health → status → positions flow.

**Rationale**:
- Validates critical path
- Fast execution
- Catches major issues

**Trade-offs**:
- Doesn't test full engine lifecycle
- May miss edge cases

## Deployment Assumptions

### 1. Docker Compose for Development

**Assumption**: `make dev` uses Docker Compose.

**Rationale**:
- Consistent environment
- Easy to start/stop
- Matches production setup

**Trade-offs**:
- Requires Docker
- Slower than local execution

### 2. Single API Instance

**Assumption**: Single API instance, no load balancing.

**Rationale**:
- Sufficient for current scale
- Simpler deployment
- Can scale later

**Trade-offs**:
- Single point of failure
- Limited scalability

**Future Enhancement**: Multiple API instances with load balancer.

## Data Assumptions

### 1. Position Reconciliation

**Assumption**: Reconcile compares sizes with 0.0001 tolerance.

**Rationale**:
- Accounts for floating point precision
- Reasonable for crypto trading
- Prevents false positives

**Trade-offs**:
- May miss small discrepancies
- Tolerance may need adjustment per symbol

### 2. Order Status Mapping

**Assumption**: Order statuses map directly between internal and exchange.

**Rationale**:
- Simpler implementation
- Standard statuses (PENDING, PLACED, FILLED, etc.)

**Trade-offs**:
- May need mapping for different exchanges
- Exchange-specific statuses not handled

## Performance Assumptions

### 1. Polling Interval

**Assumption**: Dashboard polls every 5 seconds.

**Rationale**:
- Balance between freshness and load
- Responsive enough for trading
- Not too frequent

**Trade-offs**:
- Not real-time
- May miss rapid changes

### 2. Metrics Cache

**Assumption**: Metrics cached for 5 seconds in PortfolioAccounting.

**Rationale**:
- Reduces computation
- Acceptable staleness
- Improves performance

**Trade-offs**:
- Slight delay in metrics
- May need adjustment

## Future Considerations

These assumptions may need revision as the system evolves:

1. **Scale**: If system handles high volume, may need:
   - WebSocket/SSE for real-time updates
   - Distributed event bus
   - Multiple API instances

2. **Security**: For production deployment:
   - Add authentication/authorization
   - Restrict CORS origins
   - Add rate limiting

3. **Reliability**: For production:
   - Add background reconcile task
   - Implement retry logic
   - Add circuit breakers

4. **Observability**: For production:
   - Add distributed tracing
   - More detailed metrics
   - Alerting integration
