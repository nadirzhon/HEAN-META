"""Optuna-based parameter optimization for trading strategies."""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.study import Study

from hean.backtesting.vectorbt_engine import BacktestConfig, BacktestResult, VectorBTEngine
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for Optuna optimization."""

    n_trials: int = 100  # Number of trials to run
    n_jobs: int = -1  # Parallel jobs (-1 = all CPUs)
    timeout: Optional[int] = None  # Timeout in seconds

    # Pruning
    enable_pruning: bool = True  # Early stopping for bad trials
    pruner_warmup_steps: int = 5  # Steps before pruning starts
    pruner_interval_steps: int = 1  # Steps between pruning checks

    # Multi-objective
    directions: list[str] = None  # ['maximize', 'minimize'] for multi-objective

    # Study
    study_name: Optional[str] = None
    storage: Optional[str] = None  # Database URL for persistence
    load_if_exists: bool = True

    # Logging
    show_progress_bar: bool = True
    verbosity: int = 1  # 0=silent, 1=info, 2=debug

    def __post_init__(self) -> None:
        """Set defaults."""
        if self.directions is None:
            self.directions = ['maximize']  # Single-objective by default


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""

    study: Study
    best_params: dict[str, Any]
    best_value: float | list[float]  # Single or multi-objective
    best_trial: optuna.trial.FrozenTrial
    n_trials: int
    optimization_time: float  # seconds

    # Analysis
    param_importances: Optional[dict[str, float]] = None
    trials_df: Optional[pd.DataFrame] = None

    def __repr__(self) -> str:
        """String representation."""
        if isinstance(self.best_value, list):
            values_str = ', '.join([f"{v:.4f}" for v in self.best_value])
            return (
                f"OptimizationResult(\n"
                f"  Best Values: [{values_str}]\n"
                f"  Best Params: {self.best_params}\n"
                f"  Trials: {self.n_trials}\n"
                f"  Time: {self.optimization_time:.2f}s\n"
                f")"
            )
        else:
            return (
                f"OptimizationResult(\n"
                f"  Best Value: {self.best_value:.4f}\n"
                f"  Best Params: {self.best_params}\n"
                f"  Trials: {self.n_trials}\n"
                f"  Time: {self.optimization_time:.2f}s\n"
                f")"
            )


class OptunaOptimizer:
    """
    Optuna-based optimizer for trading strategy parameters.

    Supports:
    - Single and multi-objective optimization
    - Parallel execution
    - Early pruning of bad trials
    - Parameter importance analysis

    Example:
        >>> optimizer = OptunaOptimizer(engine, data, config)
        >>> result = optimizer.optimize_strategy(objective_func, param_space)
        >>> print(result.best_params)
    """

    def __init__(
        self,
        engine: VectorBTEngine,
        data: pd.DataFrame,
        config: Optional[OptimizationConfig] = None,
    ):
        """Initialize optimizer."""
        self.engine = engine
        self.data = data
        self.config = config or OptimizationConfig()

        # Configure Optuna logging
        if self.config.verbosity == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        elif self.config.verbosity == 1:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.DEBUG)

        logger.info(f"OptunaOptimizer initialized with {self.config.n_trials} trials")

    def optimize_strategy(
        self,
        strategy_func: Callable,
        param_space: dict[str, tuple],
        objective_metric: str | list[str] = 'sharpe_ratio',
        **backtest_kwargs: Any,
    ) -> OptimizationResult:
        """
        Optimize strategy parameters.

        Args:
            strategy_func: Function(data, **params) -> (entries, exits)
            param_space: Dict of parameter ranges:
                - ('int', low, high) for integers
                - ('float', low, high) for floats
                - ('categorical', [choices]) for categorical
            objective_metric: Metric to optimize ('sharpe_ratio', 'sortino_ratio', etc.)
                             or list of metrics for multi-objective
            **backtest_kwargs: Additional args for backtest

        Returns:
            OptimizationResult

        Example:
            >>> param_space = {
            ...     'fast_period': ('int', 5, 30),
            ...     'slow_period': ('int', 20, 100),
            ...     'rsi_oversold': ('float', 20, 40),
            ... }
            >>> result = optimizer.optimize_strategy(
            ...     my_strategy,
            ...     param_space,
            ...     objective_metric=['sharpe_ratio', 'max_drawdown'],
            ... )
        """
        import time
        start_time = time.time()

        # Multi-objective?
        is_multi_objective = isinstance(objective_metric, list)

        if is_multi_objective:
            directions = []
            for metric in objective_metric:
                # Metrics to minimize (lower is better)
                if any(
                    x in metric.lower()
                    for x in ['drawdown', 'loss', 'volatility', 'var', 'cvar']
                ):
                    directions.append('minimize')
                else:
                    directions.append('maximize')
            self.config.directions = directions

        # Create study
        sampler = TPESampler(seed=42, multivariate=True)
        pruner = (
            MedianPruner(
                n_startup_trials=self.config.pruner_warmup_steps,
                n_warmup_steps=self.config.pruner_warmup_steps,
                interval_steps=self.config.pruner_interval_steps,
            )
            if self.config.enable_pruning
            else optuna.pruners.NopPruner()
        )

        study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.storage,
            load_if_exists=self.config.load_if_exists,
            directions=self.config.directions,
            sampler=sampler,
            pruner=pruner,
        )

        # Define objective
        def objective(trial: optuna.Trial) -> float | list[float]:
            """Objective function for Optuna."""
            # Sample parameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config[0] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
                elif param_config[0] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2])
                elif param_config[0] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config[1])
                else:
                    raise ValueError(f"Unknown param type: {param_config[0]}")

            # Run backtest
            try:
                entries, exits = strategy_func(self.data, **params)
                result = self.engine.backtest(
                    self.data, entries, exits, **backtest_kwargs
                )

                # Extract objective value(s)
                if is_multi_objective:
                    values = []
                    for metric in objective_metric:
                        value = getattr(result, metric, None)
                        if value is None:
                            raise ValueError(f"Metric {metric} not found in BacktestResult")
                        values.append(value)
                    return values
                else:
                    value = getattr(result, objective_metric, None)
                    if value is None:
                        raise ValueError(f"Metric {objective_metric} not found in BacktestResult")
                    return value

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                # Return worst possible value
                if is_multi_objective:
                    return [float('-inf') if d == 'maximize' else float('inf') for d in directions]
                else:
                    return float('-inf') if self.config.directions[0] == 'maximize' else float('inf')

        # Run optimization
        logger.info(f"Starting optimization with {self.config.n_trials} trials...")
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout,
            show_progress_bar=self.config.show_progress_bar,
        )

        optimization_time = time.time() - start_time

        # Get best results
        if is_multi_objective:
            # For multi-objective, use first Pareto-optimal trial
            best_trial = study.best_trials[0]
            best_value = best_trial.values
        else:
            best_trial = study.best_trial
            best_value = study.best_value

        best_params = best_trial.params

        # Analyze parameter importance
        try:
            if is_multi_objective:
                # Use first objective for importance
                param_importances = optuna.importance.get_param_importances(
                    study, target=lambda t: t.values[0]
                )
            else:
                param_importances = optuna.importance.get_param_importances(study)
        except Exception as e:
            logger.warning(f"Could not calculate param importances: {e}")
            param_importances = None

        # Create trials DataFrame
        trials_df = study.trials_dataframe()

        logger.info(
            f"Optimization completed in {optimization_time:.2f}s\n"
            f"Best params: {best_params}\n"
            f"Best value: {best_value}"
        )

        return OptimizationResult(
            study=study,
            best_params=best_params,
            best_value=best_value,
            best_trial=best_trial,
            n_trials=len(study.trials),
            optimization_time=optimization_time,
            param_importances=param_importances,
            trials_df=trials_df,
        )

    def optimize_multi_objective(
        self,
        strategy_func: Callable,
        param_space: dict[str, tuple],
        objectives: list[str] = None,
        **backtest_kwargs: Any,
    ) -> OptimizationResult:
        """
        Multi-objective optimization (e.g., maximize profit, minimize drawdown).

        Args:
            strategy_func: Strategy function
            param_space: Parameter space
            objectives: List of objectives (default: ['total_return', 'max_drawdown'])
            **backtest_kwargs: Additional backtest args

        Returns:
            OptimizationResult with Pareto frontier
        """
        if objectives is None:
            objectives = ['total_return', 'max_drawdown']

        return self.optimize_strategy(
            strategy_func,
            param_space,
            objective_metric=objectives,
            **backtest_kwargs,
        )

    def grid_search(
        self,
        strategy_func: Callable,
        param_grid: dict[str, list],
        objective_metric: str = 'sharpe_ratio',
        **backtest_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Grid search over parameter space.

        Args:
            strategy_func: Strategy function
            param_grid: Dict of parameter values to test
            objective_metric: Metric to optimize
            **backtest_kwargs: Additional backtest args

        Returns:
            DataFrame with all combinations and results
        """
        from itertools import product

        logger.info("Running grid search...")

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        results = []
        for combo in combinations:
            params = dict(zip(param_names, combo))

            try:
                entries, exits = strategy_func(self.data, **params)
                result = self.engine.backtest(
                    self.data, entries, exits, **backtest_kwargs
                )

                metric_value = getattr(result, objective_metric)

                results.append({
                    **params,
                    objective_metric: metric_value,
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades,
                })
            except Exception as e:
                logger.warning(f"Failed for params {params}: {e}")

        df = pd.DataFrame(results)
        df = df.sort_values(objective_metric, ascending=False)

        logger.info(f"Grid search completed. Tested {len(combinations)} combinations.")

        return df


def optimize_ma_crossover(
    data: pd.DataFrame,
    engine: VectorBTEngine,
    config: Optional[OptimizationConfig] = None,
) -> OptimizationResult:
    """
    Optimize moving average crossover strategy.

    Example helper function showing how to use the optimizer.
    """
    from hean.backtesting.vectorbt_engine import create_simple_ma_crossover_signals

    optimizer = OptunaOptimizer(engine, data, config)

    param_space = {
        'fast_period': ('int', 5, 50),
        'slow_period': ('int', 20, 200),
    }

    result = optimizer.optimize_strategy(
        create_simple_ma_crossover_signals,
        param_space,
        objective_metric='sharpe_ratio',
    )

    return result


def optimize_rsi_strategy(
    data: pd.DataFrame,
    engine: VectorBTEngine,
    config: Optional[OptimizationConfig] = None,
) -> OptimizationResult:
    """
    Optimize RSI mean reversion strategy.

    Example helper function.
    """
    from hean.backtesting.vectorbt_engine import create_rsi_mean_reversion_signals

    optimizer = OptunaOptimizer(engine, data, config)

    param_space = {
        'rsi_period': ('int', 7, 21),
        'oversold': ('float', 20, 40),
        'overbought': ('float', 60, 80),
    }

    result = optimizer.optimize_multi_objective(
        create_rsi_mean_reversion_signals,
        param_space,
        objectives=['sharpe_ratio', 'max_drawdown'],
    )

    return result
