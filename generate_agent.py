#!/usr/bin/env python3
"""CLI tool for generating trading agents using LLM prompts."""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hean.agent_generation import AgentGenerator
from hean.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate trading agents using LLM prompts"
    )
    
    parser.add_argument(
        "prompt_type",
        choices=[
            "initial",
            "evolution",
            "analytical",
            "mutation",
            "market_conditions",
            "hybrid",
            "problem_focused",
            "evaluation",
            "creative",
        ],
        help="Type of prompt to use",
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (default: stdout)",
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of agents to generate (for initial type)",
    )
    
    # Evolution arguments
    parser.add_argument("--best-agents", type=str, help="Best agents info")
    parser.add_argument("--worst-agents", type=str, help="Worst agents info")
    parser.add_argument("--market-conditions", type=str, help="Market conditions")
    parser.add_argument("--performance-metrics", type=str, help="Performance metrics")
    
    # Mutation arguments
    parser.add_argument("--agent-code", type=str, help="Agent code to mutate")
    parser.add_argument("--profit-factor", type=float, help="Profit factor")
    parser.add_argument("--total-pnl", type=float, help="Total PnL")
    parser.add_argument("--max-drawdown", type=float, help="Max drawdown %")
    parser.add_argument("--win-rate", type=float, help="Win rate %")
    parser.add_argument("--issues", type=str, help="Issues description")
    
    # Market conditions arguments
    parser.add_argument("--volatility-level", type=str, help="Volatility level")
    parser.add_argument("--volatility-value", type=float, help="Volatility value")
    parser.add_argument("--trend-direction", type=str, help="Trend direction")
    parser.add_argument("--trend-strength", type=str, help="Trend strength")
    parser.add_argument("--volume-level", type=str, help="Volume level")
    parser.add_argument("--market-regime", type=str, help="Market regime")
    parser.add_argument("--spread-bps", type=float, help="Spread in bps")
    parser.add_argument("--historical-summary", type=str, help="Historical summary")
    parser.add_argument("--suggested-style", type=str, help="Suggested style")
    parser.add_argument("--suggested-timeframe", type=str, help="Suggested timeframe")
    parser.add_argument("--suggested-size", type=str, help="Suggested size")
    parser.add_argument("--risk-approach", type=str, help="Risk approach")
    
    # Hybrid arguments
    parser.add_argument("--agent1-code", type=str, help="Agent 1 code")
    parser.add_argument("--pf1", type=float, help="Agent 1 PF")
    parser.add_argument("--pnl1", type=float, help="Agent 1 PnL")
    parser.add_argument("--agent2-code", type=str, help="Agent 2 code")
    parser.add_argument("--pf2", type=float, help="Agent 2 PF")
    parser.add_argument("--wr2", type=float, help="Agent 2 WR")
    parser.add_argument("--agent3-code", type=str, help="Agent 3 code")
    parser.add_argument("--pf3", type=float, help="Agent 3 PF")
    parser.add_argument("--sharpe3", type=float, help="Agent 3 Sharpe")
    
    # Problem-focused arguments
    parser.add_argument("--problem", type=str, help="Problem description")
    parser.add_argument("--current-pf", type=float, help="Current PF")
    parser.add_argument("--problem-areas", type=str, help="Problem areas")
    parser.add_argument("--failed-patterns", type=str, help="Failed patterns")
    parser.add_argument("--focus1", type=str, help="Focus area 1")
    parser.add_argument("--focus2", type=str, help="Focus area 2")
    parser.add_argument("--focus3", type=str, help="Focus area 3")
    
    # Evaluation arguments
    parser.add_argument("--pf", type=float, help="Profit factor")
    parser.add_argument("--pnl", type=float, help="Total PnL")
    parser.add_argument("--dd", type=float, help="Max drawdown %")
    parser.add_argument("--wr", type=float, help="Win rate %")
    parser.add_argument("--sharpe", type=float, help="Sharpe ratio")
    parser.add_argument("--trades", type=int, help="Total trades")
    
    args = parser.parse_args()
    
    # Initialize generator
    try:
        generator = AgentGenerator()
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        logger.error("Make sure OPENAI_API_KEY or ANTHROPIC_API_KEY is set")
        sys.exit(1)
    
    # Generate based on prompt type
    try:
        if args.prompt_type == "initial":
            if args.count > 1:
                output_dir = args.output or "generated_agents"
                codes = generator.generate_initial_agents(
                    count=args.count,
                    output_dir=output_dir
                )
                for i, code in enumerate(codes, 1):
                    if args.output:
                        output_path = Path(output_dir) / f"agent_{i:03d}.py"
                        output_path.write_text(code, encoding="utf-8")
                    else:
                        print(f"# Agent {i}\n{code}\n")
            else:
                code = generator.generate_agent(
                    prompt_type="initial",
                    output_path=args.output
                )
                if not args.output:
                    print(code)
        
        elif args.prompt_type == "evolution":
            if not all([args.best_agents, args.worst_agents, args.market_conditions, args.performance_metrics]):
                parser.error("Evolution requires --best-agents, --worst-agents, --market-conditions, --performance-metrics")
            code = generator.evolve_agent(
                best_agents_info=args.best_agents,
                worst_agents_info=args.worst_agents,
                market_conditions=args.market_conditions,
                performance_metrics=args.performance_metrics,
                output_path=args.output
            )
            if not args.output:
                print(code)
        
        elif args.prompt_type == "mutation":
            if not all([args.agent_code, args.profit_factor, args.total_pnl, args.max_drawdown, args.win_rate, args.issues]):
                parser.error("Mutation requires --agent-code, --profit-factor, --total-pnl, --max-drawdown, --win-rate, --issues")
            code = generator.mutate_agent(
                agent_code=args.agent_code,
                profit_factor=args.profit_factor,
                total_pnl=args.total_pnl,
                max_drawdown_pct=args.max_drawdown,
                win_rate=args.win_rate,
                issues=args.issues,
                output_path=args.output
            )
            if not args.output:
                print(code)
        
        elif args.prompt_type == "market_conditions":
            required = [
                args.volatility_level, args.volatility_value, args.trend_direction,
                args.trend_strength, args.volume_level, args.market_regime,
                args.spread_bps, args.historical_summary, args.suggested_style,
                args.suggested_timeframe, args.suggested_size, args.risk_approach
            ]
            if not all(required):
                parser.error("Market conditions requires all market condition arguments")
            code = generator.generate_market_specialized_agent(
                volatility_level=args.volatility_level,
                volatility_value=args.volatility_value,
                trend_direction=args.trend_direction,
                trend_strength=args.trend_strength,
                volume_level=args.volume_level,
                market_regime=args.market_regime,
                spread_bps=args.spread_bps,
                historical_summary=args.historical_summary,
                suggested_style=args.suggested_style,
                suggested_timeframe=args.suggested_timeframe,
                suggested_size=args.suggested_size,
                risk_approach=args.risk_approach,
                output_path=args.output
            )
            if not args.output:
                print(code)
        
        elif args.prompt_type == "hybrid":
            required = [
                args.agent1_code, args.pf1, args.pnl1,
                args.agent2_code, args.pf2, args.wr2,
                args.agent3_code, args.pf3, args.sharpe3
            ]
            if not all(required):
                parser.error("Hybrid requires all agent codes and metrics")
            code = generator.generate_hybrid_agent(
                agent1_code=args.agent1_code,
                pf1=args.pf1,
                pnl1=args.pnl1,
                agent2_code=args.agent2_code,
                pf2=args.pf2,
                wr2=args.wr2,
                agent3_code=args.agent3_code,
                pf3=args.pf3,
                sharpe3=args.sharpe3,
                output_path=args.output
            )
            if not args.output:
                print(code)
        
        elif args.prompt_type == "problem_focused":
            required = [
                args.problem, args.current_pf, args.problem_areas,
                args.failed_patterns, args.focus1, args.focus2, args.focus3
            ]
            if not all(required):
                parser.error("Problem focused requires all problem arguments")
            code = generator.generate_problem_focused_agent(
                problem_description=args.problem,
                current_pf=args.current_pf,
                problem_areas=args.problem_areas,
                failed_patterns=args.failed_patterns,
                focus_area_1=args.focus1,
                focus_area_2=args.focus2,
                focus_area_3=args.focus3,
                output_path=args.output
            )
            if not args.output:
                print(code)
        
        elif args.prompt_type == "evaluation":
            if not all([args.agent_code, args.pf, args.pnl, args.dd, args.wr, args.sharpe, args.trades]):
                parser.error("Evaluation requires --agent-code and all metrics")
            code = generator.evaluate_and_improve(
                agent_code=args.agent_code,
                pf=args.pf,
                pnl=args.pnl,
                dd=args.dd,
                wr=args.wr,
                sharpe=args.sharpe,
                trades=args.trades,
                output_path=args.output
            )
            if not args.output:
                print(code)
        
        elif args.prompt_type == "creative":
            code = generator.generate_creative_agent(output_path=args.output)
            if not args.output:
                print(code)
        
        elif args.prompt_type == "analytical":
            code = generator.generate_agent(
                prompt_type="analytical",
                output_path=args.output
            )
            if not args.output:
                print(code)
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

