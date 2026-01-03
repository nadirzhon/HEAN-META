"""Daily report generation for Process Factory."""

import json
from datetime import datetime
from pathlib import Path

from hean.process_factory.schemas import (
    DailyCapitalPlan,
    ProcessPortfolioEntry,
    ProcessPortfolioState,
    ProcessRun,
)


class ProcessReportGenerator:
    """Generates daily reports for Process Factory."""

    def __init__(self, output_dir: str | Path = ".") -> None:
        """Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_daily_report(
        self,
        portfolio: list[ProcessPortfolioEntry],
        capital_plan: DailyCapitalPlan | None = None,
        recent_runs: list[ProcessRun] | None = None,
        suggested_processes: list[str] | None = None,
    ) -> tuple[Path, Path] | None:
        """Generate daily report (markdown and JSON).
        
        Idempotent: returns existing paths if report already exists for today.

        Args:
            portfolio: Process portfolio
            capital_plan: Daily capital plan (optional)
            recent_runs: Recent process runs (optional)
            suggested_processes: List of suggested new process IDs (optional)

        Returns:
            Tuple of (markdown_path, json_path) or None if skipped (idempotent)
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        md_path = self.output_dir / f"process_factory_report_{date_str}.md"
        json_path = self.output_dir / f"process_factory_report_{date_str}.json"

        # Idempotency check: skip if both files already exist
        if md_path.exists() and json_path.exists():
            return None  # Signal idempotent skip

        # Generate JSON report
        json_data = self._generate_json_report(
            portfolio, capital_plan, recent_runs, suggested_processes
        )
        json_path.write_text(json.dumps(json_data, indent=2, default=str))

        # Generate Markdown report
        md_content = self._generate_markdown_report(
            portfolio, capital_plan, recent_runs, suggested_processes
        )
        md_path.write_text(md_content)

        return md_path, json_path

    def _generate_json_report(
        self,
        portfolio: list[ProcessPortfolioEntry],
        capital_plan: DailyCapitalPlan | None,
        recent_runs: list[ProcessRun] | None,
        suggested_processes: list[str] | None,
    ) -> dict:
        """Generate JSON report data."""
        return {
            "date": datetime.now().isoformat(),
            "portfolio": [entry.model_dump(mode="json") for entry in portfolio],
            "capital_plan": capital_plan.model_dump(mode="json") if capital_plan else None,
            "recent_runs": [
                run.model_dump(mode="json") for run in (recent_runs or [])[:50]
            ],  # Limit to 50 most recent
            "suggested_processes": suggested_processes or [],
            "summary": {
                "total_processes": len(portfolio),
                "core_processes": len([p for p in portfolio if p.state == ProcessPortfolioState.CORE]),
                "testing_processes": len(
                    [p for p in portfolio if p.state == ProcessPortfolioState.TESTING]
                ),
                "killed_processes": len(
                    [p for p in portfolio if p.state == ProcessPortfolioState.KILLED]
                ),
                "total_pnl": sum(p.pnl_sum for p in portfolio),
                "total_runs": sum(p.runs_count for p in portfolio),
            },
        }

    def _generate_markdown_report(
        self,
        portfolio: list[ProcessPortfolioEntry],
        capital_plan: DailyCapitalPlan | None,
        recent_runs: list[ProcessRun] | None,
        suggested_processes: list[str] | None,
    ) -> str:
        """Generate Markdown report content."""
        lines = []
        lines.append("# Process Factory Daily Report")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        core_count = len([p for p in portfolio if p.state == ProcessPortfolioState.CORE])
        testing_count = len([p for p in portfolio if p.state == ProcessPortfolioState.TESTING])
        killed_count = len([p for p in portfolio if p.state == ProcessPortfolioState.KILLED])
        total_pnl = sum(p.pnl_sum for p in portfolio)
        total_runs = sum(p.runs_count for p in portfolio)

        lines.append(f"- **Total Processes:** {len(portfolio)}")
        lines.append(f"- **Core Processes:** {core_count}")
        lines.append(f"- **Testing Processes:** {testing_count}")
        lines.append(f"- **Killed Processes:** {killed_count}")
        lines.append(f"- **Total PnL:** ${total_pnl:.2f}")
        lines.append(f"- **Total Runs:** {total_runs}")
        lines.append("")

        # Top Processes by Contribution
        lines.append("## Top Processes by Contribution")
        lines.append("")
        sorted_portfolio = sorted(
            portfolio, key=lambda p: p.pnl_sum, reverse=True
        )[:10]
        lines.append("| Process ID | State | PnL | Runs | ROI | Fail Rate |")
        lines.append("|------------|-------|-----|------|-----|-----------|")
        for entry in sorted_portfolio:
            lines.append(
                f"| {entry.process_id} | {entry.state.value} | ${entry.pnl_sum:.2f} | "
                f"{entry.runs_count} | {entry.avg_roi:.2%} | {entry.fail_rate:.2%} |"
            )
        lines.append("")

        # Killed Processes
        killed = [p for p in portfolio if p.state == ProcessPortfolioState.KILLED]
        if killed:
            lines.append("## Killed Processes")
            lines.append("")
            lines.append("| Process ID | PnL | Runs | Fail Rate | Max DD |")
            lines.append("|------------|-----|------|-----------|--------|")
            for entry in killed:
                lines.append(
                    f"| {entry.process_id} | ${entry.pnl_sum:.2f} | {entry.runs_count} | "
                    f"{entry.fail_rate:.2%} | {entry.max_dd:.2%} |"
                )
            lines.append("")

        # Capital Plan
        if capital_plan:
            lines.append("## Daily Capital Plan")
            lines.append("")
            lines.append(f"- **Reserve:** ${capital_plan.reserve_usd:.2f}")
            lines.append(f"- **Active:** ${capital_plan.active_usd:.2f}")
            lines.append(f"- **Experimental:** ${capital_plan.experimental_usd:.2f}")
            lines.append(f"- **Total:** ${capital_plan.total_capital_usd:.2f}")
            lines.append("")
            if capital_plan.allocations:
                lines.append("### Allocations")
                lines.append("")
                lines.append("| Process ID | Allocation |")
                lines.append("|------------|------------|")
                for process_id, allocation in sorted(
                    capital_plan.allocations.items(), key=lambda x: x[1], reverse=True
                ):
                    lines.append(f"| {process_id} | ${allocation:.2f} |")
                lines.append("")

        # Suggested New Processes
        if suggested_processes:
            lines.append("## Suggested New Processes")
            lines.append("")
            for process_id in suggested_processes:
                lines.append(f"- `{process_id}`")
            lines.append("")

        return "\n".join(lines)

