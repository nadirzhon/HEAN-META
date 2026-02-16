# council -- Multi-Agent AI Council

Periodic multi-model AI review of the trading system's performance, architecture, and risk posture. Four AI council members with different specializations provide independent assessments and recommendations.

## Architecture

The `AICouncil` orchestrates periodic review cycles (default every 6 hours). Each cycle: (1) the `Introspector` collects a comprehensive system snapshot including portfolio metrics, strategy performance, risk state, recent events, and error logs; (2) each `CouncilMember` receives the snapshot along with their role-specific system prompt; (3) responses are parsed into structured `CouncilReview` objects with recommendations; (4) safe, auto-applicable recommendations are sent to the `CouncilExecutor` immediately; (5) code-level recommendations are queued for human approval; (6) `COUNCIL_REVIEW` and `COUNCIL_RECOMMENDATION` events are published. All API calls go through OpenRouter, allowing different model vendors per council role.

## Key Classes

- `AICouncil` (`council.py`) -- Main orchestrator. Manages review cycles, member coordination, session history (last 50 sessions), and recommendation tracking (last 500). Publishes events to EventBus.
- `CouncilMember` (`members.py`) -- Definition of an AI council member: role, display name, model ID (OpenRouter format), system prompt, max tokens, temperature.
- `Introspector` (`introspector.py`) -- Collects comprehensive system state for council review. Subscribes to SELF_ANALYTICS, RISK_ALERT, ORDER_FILLED, ORDER_REJECTED, PNL_UPDATE, ERROR, and KILLSWITCH_TRIGGERED events.
- `CouncilExecutor` (`executor.py`) -- Applies safe, auto-applicable recommendations. Code-level changes require human approval.
- `CouncilReview` / `CouncilSession` / `Recommendation` (`review.py`) -- Data models for review results with categories, severity levels, and approval status.

### Default Council Members

| Role | Model | Specialization |
|------|-------|---------------|
| System Architect | Qwen3-Max (qwen/qwen3-max) | Architecture, design patterns, system coherence |
| Code Reviewer | Claude Sonnet (anthropic/claude-sonnet-4-5) | Code quality, bugs, security |
| Quant Analyst | GPT-4o (openai/gpt-4o) | Quantitative analysis, strategy performance |
| Performance Optimizer | DeepSeek-R1 (deepseek/deepseek-r1) | Latency, throughput, resource optimization |

## Events

| Event | Direction | Description |
|-------|-----------|-------------|
| COUNCIL_REVIEW | Publishes | Complete review from a council member |
| COUNCIL_RECOMMENDATION | Publishes | Individual recommendation for action |
| SELF_ANALYTICS | Subscribes (Introspector) | System self-insight telemetry |
| RISK_ALERT | Subscribes (Introspector) | Risk events for context |
| ERROR | Subscribes (Introspector) | Error events for context |
| KILLSWITCH_TRIGGERED | Subscribes (Introspector) | Killswitch events for context |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| COUNCIL_ENABLED | false | Enable AI council reviews |
| COUNCIL_REVIEW_INTERVAL | 21600 | Seconds between review cycles (6 hours) |
| COUNCIL_AUTO_APPLY_SAFE | true | Auto-apply safe recommendations |
| OPENROUTER_API_KEY | "" | OpenRouter API key for council models |
