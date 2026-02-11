"""AI Council orchestrator — periodic multi-model system review."""

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from datetime import datetime
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.council.executor import CouncilExecutor
from hean.council.introspector import Introspector
from hean.council.members import DEFAULT_MEMBERS, CouncilMember
from hean.council.review import (
    ApprovalStatus,
    Category,
    CouncilReview,
    CouncilSession,
    Recommendation,
    Severity,
)

logger = logging.getLogger(__name__)


class AICouncil:
    """Orchestrates periodic system reviews by multiple AI models via OpenRouter.

    Each review cycle:
    1. Introspector collects system snapshot
    2. Each CouncilMember receives snapshot + their role prompt
    3. Responses are parsed into CouncilReview objects
    4. Auto-applicable recommendations go to Executor immediately
    5. Code-level recommendations are queued for human approval
    6. COUNCIL_REVIEW and COUNCIL_RECOMMENDATION events published
    """

    def __init__(
        self,
        bus: EventBus,
        accounting: Any | None = None,
        strategies: dict[str, Any] | None = None,
        killswitch: Any | None = None,
        ai_factory: Any | None = None,
        improvement_catalyst: Any | None = None,
        review_interval: int = 21600,
        auto_apply_safe: bool = True,
    ) -> None:
        self._bus = bus
        self._review_interval = review_interval
        self._auto_apply_safe = auto_apply_safe

        self._introspector = Introspector(bus, accounting, strategies, killswitch)
        self._executor = CouncilExecutor(bus, ai_factory, improvement_catalyst)
        self._members = list(DEFAULT_MEMBERS)

        self._client: Any | None = None
        self._setup_client()

        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._sessions: deque[CouncilSession] = deque(maxlen=50)
        self._all_recommendations: deque[Recommendation] = deque(maxlen=500)

    def _setup_client(self) -> None:
        """Initialize AsyncOpenAI client for OpenRouter."""
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            try:
                from hean.config import settings
                api_key = settings.openrouter_api_key
            except Exception:
                pass

        if api_key:
            try:
                import openai
                self._client = openai.AsyncOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
                logger.info("AI Council: OpenRouter client initialized")
            except ImportError:
                logger.warning("AI Council: openai package not installed")
        else:
            logger.warning("AI Council: No OPENROUTER_API_KEY, council will be inactive")

    async def start(self) -> None:
        """Start the council."""
        if self._running:
            return
        if not self._client:
            logger.warning("AI Council: No client configured, not starting")
            return

        self._running = True
        await self._introspector.start()
        self._task = asyncio.create_task(self._review_loop())
        logger.info(
            f"AI Council started ({len(self._members)} members, "
            f"interval={self._review_interval}s)"
        )

    async def stop(self) -> None:
        """Stop the council."""
        self._running = False
        await self._introspector.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("AI Council stopped")

    async def _review_loop(self) -> None:
        """Main review loop — runs periodically."""
        # Warmup: let the system accumulate meaningful data
        await asyncio.sleep(300)

        while self._running:
            try:
                session = await self._run_review_session()
                self._sessions.append(session)
                logger.info(
                    f"Council session {session.session_id}: "
                    f"{session.total_recommendations} recommendations, "
                    f"{session.auto_applied_count} auto-applied, "
                    f"{session.pending_approval_count} pending"
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Council review error: {e}", exc_info=True)

            await asyncio.sleep(self._review_interval)

    async def _run_review_session(self) -> CouncilSession:
        """Execute one complete review session."""
        session = CouncilSession()

        # 1. Collect system snapshot
        snapshot = self._introspector.collect_snapshot()

        # 2. Format for LLMs
        context = self._format_snapshot_for_llm(snapshot)

        # 3. Fan out to all council members concurrently
        review_tasks = [
            self._get_member_review(member, context)
            for member in self._members
        ]
        reviews = await asyncio.gather(*review_tasks, return_exceptions=True)

        # 4. Process results
        for result in reviews:
            if isinstance(result, Exception):
                logger.warning(f"Council member review failed: {result}")
                continue
            if not isinstance(result, CouncilReview):
                continue

            session.reviews.append(result)
            for rec in result.recommendations:
                self._all_recommendations.append(rec)
                session.total_recommendations += 1

                # 5. Auto-apply safe recommendations
                if rec.auto_applicable and self._auto_apply_safe:
                    apply_result = await self._executor.apply_recommendation(rec)
                    if apply_result.get("status") == "applied":
                        rec.approval_status = ApprovalStatus.AUTO_APPLIED
                        rec.applied_at = datetime.utcnow().isoformat()
                        rec.apply_result = apply_result
                        session.auto_applied_count += 1
                    else:
                        session.pending_approval_count += 1
                else:
                    session.pending_approval_count += 1

        session.completed_at = datetime.utcnow().isoformat()

        # 6. Publish events
        await self._publish_session_events(session)

        return session

    async def _get_member_review(
        self, member: CouncilMember, context: str
    ) -> CouncilReview:
        """Get review from a single council member via OpenRouter."""
        if not self._client:
            raise RuntimeError("OpenRouter client not configured")

        start_time = time.time()

        user_prompt = (
            "Review the following HEAN trading system state and provide "
            "your analysis from your role perspective.\n\n"
            f"## System State\n{context}\n\n"
            "Respond ONLY with valid JSON matching the expected format."
        )

        response = await self._client.chat.completions.create(
            model=member.model_id,
            max_tokens=member.max_tokens,
            temperature=member.temperature,
            messages=[
                {"role": "system", "content": member.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            extra_headers={
                "HTTP-Referer": "https://hean.trading",
                "X-Title": "HEAN AI Council",
            },
        )

        elapsed_ms = (time.time() - start_time) * 1000
        raw_text = response.choices[0].message.content or ""

        # Strip thinking tags (Qwen/DeepSeek reasoning models)
        if "<think>" in raw_text:
            raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

        review = self._parse_member_response(member, raw_text)
        review.processing_time_ms = elapsed_ms

        if response.usage:
            review.token_usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
            }

        logger.info(
            f"Council {member.role}: {len(review.recommendations)} recommendations "
            f"({elapsed_ms:.0f}ms)"
        )
        return review

    def _parse_member_response(
        self, member: CouncilMember, raw_text: str
    ) -> CouncilReview:
        """Parse LLM JSON response into CouncilReview."""
        review = CouncilReview(
            member_role=member.role,
            model_id=member.model_id,
            raw_response=raw_text[:5000],
        )

        try:
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            if start < 0 or end <= start:
                review.summary = f"No JSON found in response ({len(raw_text)} chars)"
                return review

            data = json.loads(raw_text[start:end])
            review.summary = data.get("summary", "")

            for rec_data in data.get("recommendations", []):
                try:
                    severity_str = rec_data.get("severity", "medium").lower()
                    category_str = rec_data.get("category", "code_quality").lower()

                    rec = Recommendation(
                        member_role=member.role,
                        model_id=member.model_id,
                        severity=Severity(severity_str) if severity_str in Severity.__members__.values() or severity_str in [s.value for s in Severity] else Severity.MEDIUM,
                        category=Category(category_str) if category_str in [c.value for c in Category] else Category.CODE_QUALITY,
                        title=str(rec_data.get("title", ""))[:100],
                        description=str(rec_data.get("description", "")),
                        action=str(rec_data.get("action", "")),
                        auto_applicable=bool(rec_data.get("auto_applicable", False)),
                        target_strategy=rec_data.get("target_strategy"),
                        param_changes=rec_data.get("param_changes"),
                        code_diff=rec_data.get("code_diff"),
                        target_file=rec_data.get("target_file"),
                    )
                    review.recommendations.append(rec)
                except Exception as e:
                    logger.warning(f"Failed to parse recommendation: {e}")

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse {member.role} review JSON: {e}")
            review.summary = f"Parse error: {raw_text[:200]}"

        return review

    def _format_snapshot_for_llm(self, snapshot: dict[str, Any]) -> str:
        """Format system snapshot into LLM-readable context string."""
        parts: list[str] = [f"Timestamp: {snapshot.get('timestamp', 'N/A')}"]

        # Trading metrics
        tm = snapshot.get("trading_metrics", {})
        if tm.get("status") != "no_accounting":
            parts.append(
                f"\n### Trading Metrics\n"
                f"- Equity: ${tm.get('equity', 0):.2f} (Initial: ${tm.get('initial_capital', 0):.2f})\n"
                f"- Daily PnL: ${tm.get('daily_pnl', 0):.2f}\n"
                f"- Realized PnL: ${tm.get('realized_pnl', 0):.2f}\n"
                f"- Unrealized PnL: ${tm.get('unrealized_pnl', 0):.2f}\n"
                f"- Drawdown: {tm.get('drawdown_pct', 0):.1f}%\n"
                f"- Open Positions: {tm.get('open_positions', 0)}\n"
                f"- Total Fees: ${tm.get('total_fees', 0):.2f}"
            )

        # Strategy performance
        sp = snapshot.get("strategy_performance", {})
        if sp and not sp.get("error"):
            parts.append("\n### Strategy Performance")
            for sid, metrics in sp.items():
                if isinstance(metrics, dict):
                    parts.append(
                        f"- {sid}: PnL=${metrics.get('pnl', 0):.2f}, "
                        f"WR={metrics.get('win_rate_pct', 0):.1f}%, "
                        f"PF={metrics.get('profit_factor', 0):.2f}, "
                        f"Trades={int(metrics.get('trades', 0))}, "
                        f"MaxDD={metrics.get('max_drawdown_pct', 0):.1f}%"
                    )

        # Risk state
        rs = snapshot.get("risk_state", {})
        ks_status = "TRIGGERED" if rs.get("killswitch_triggered") else "OK"
        parts.append(f"\n### Risk State\n- KillSwitch: {ks_status}")
        if rs.get("killswitch_reason"):
            parts.append(f"- Reason: {rs['killswitch_reason']}")
        alerts = rs.get("recent_risk_alerts", [])
        if alerts:
            parts.append(f"- Recent risk alerts: {len(alerts)}")
            for a in alerts[-3:]:
                parts.append(f"  - {a.get('type', '?')}: {a.get('reason', a.get('message', '?'))}")

        # System health
        sh = snapshot.get("system_health", {})
        if not sh.get("error"):
            parts.append(
                f"\n### System Health (EventBus)\n"
                f"- Events published: {sh.get('events_published', 0)}\n"
                f"- Events dropped: {sh.get('events_dropped', 0)}\n"
                f"- Queue utilization: {sh.get('total_queue_utilization_pct', 0):.1f}%\n"
                f"- Circuit breaker: {'OPEN' if sh.get('circuit_breaker_open') else 'closed'}\n"
                f"- Handler errors: {sh.get('handler_errors', 0)}\n"
                f"- Events/sec: {sh.get('events_per_second', 0):.1f}\n"
                f"- Drop rate: {sh.get('drop_rate_pct', 0):.2f}%"
            )

        # Errors
        es = snapshot.get("error_summary", {})
        errors = es.get("recent_errors", [])
        if errors:
            parts.append(f"\n### Recent Errors ({len(errors)})")
            for err in errors[:5]:
                parts.append(f"- {err.get('error', err.get('message', 'unknown'))}")

        # Code structure (strategy excerpts)
        cs = snapshot.get("code_structure", {})
        strategies_code = cs.get("strategies", {})
        if strategies_code:
            parts.append("\n### Active Strategies (source excerpts)")
            for sid, info in strategies_code.items():
                excerpt = info.get("source_excerpt", "")[:800]
                if excerpt:
                    parts.append(
                        f"\n#### {sid} ({info.get('class_name', '?')})\n"
                        f"```python\n{excerpt}\n```"
                    )

        return "\n".join(parts)

    # -- Public API --

    def get_latest_session(self) -> dict[str, Any] | None:
        if self._sessions:
            return self._sessions[-1].model_dump()
        return None

    def get_all_recommendations(self, limit: int = 50) -> list[dict[str, Any]]:
        recs = list(self._all_recommendations)[-limit:]
        return [r.model_dump() for r in recs]

    def get_pending_recommendations(self) -> list[dict[str, Any]]:
        return [
            r.model_dump() for r in self._all_recommendations
            if r.approval_status == ApprovalStatus.PENDING
        ]

    def approve_recommendation(self, rec_id: str) -> Recommendation | None:
        for rec in self._all_recommendations:
            if rec.id == rec_id and rec.approval_status == ApprovalStatus.PENDING:
                rec.approval_status = ApprovalStatus.APPROVED
                return rec
        return None

    def get_status(self) -> dict[str, Any]:
        last_session = self._sessions[-1] if self._sessions else None
        pending = sum(
            1 for r in self._all_recommendations
            if r.approval_status == ApprovalStatus.PENDING
        )
        return {
            "enabled": self._running,
            "client_configured": self._client is not None,
            "review_interval_hours": self._review_interval / 3600,
            "auto_apply_safe": self._auto_apply_safe,
            "total_sessions": len(self._sessions),
            "total_recommendations": len(self._all_recommendations),
            "pending_recommendations": pending,
            "last_review_at": last_session.completed_at if last_session else None,
            "members": [
                {"role": m.role, "model": m.model_id, "name": m.display_name}
                for m in self._members
            ],
        }

    async def _publish_session_events(self, session: CouncilSession) -> None:
        """Publish council events to EventBus."""
        try:
            await self._bus.publish(Event(
                event_type=EventType.COUNCIL_REVIEW,
                data={
                    "session_id": session.session_id,
                    "total_recommendations": session.total_recommendations,
                    "auto_applied": session.auto_applied_count,
                    "pending": session.pending_approval_count,
                    "timestamp": session.completed_at,
                },
            ))

            for review in session.reviews:
                for rec in review.recommendations:
                    if rec.severity in (Severity.CRITICAL, Severity.HIGH):
                        await self._bus.publish(Event(
                            event_type=EventType.COUNCIL_RECOMMENDATION,
                            data={
                                "id": rec.id,
                                "severity": rec.severity.value,
                                "category": rec.category.value,
                                "title": rec.title,
                                "member_role": rec.member_role,
                                "auto_applicable": rec.auto_applicable,
                            },
                        ))
        except Exception as e:
            logger.warning(f"Failed to publish council events: {e}")
