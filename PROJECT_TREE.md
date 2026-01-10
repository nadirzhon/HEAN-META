# 🌳 Структура проекта HEAN

```

📁 HEAN/
    ├── 📁 control-center/
    │   ├── 📁 lib/
    │   │   ├── 📘 api.ts
    │   │   ├── 📘 event-stream.ts
    │   │   ├── 📘 store.ts
    │   │   ├── 📄 toast.tsx
    │   │   └── 📘 utils.ts
    │   └── 📘 next-env.d.ts
    ├── 📁 docs/
    │   ├── 📝 API.md
    │   ├── 📝 ARCHITECTURE.md
    │   ├── 📝 ASSUMPTIONS.md
    │   └── 📝 UI.md
    ├── 📁 examples/
    │   └── 🐍 generate_agent_example.py
    ├── 📁 EXPORT_BUNDLE/
    │   ├── 📁 logs/
    │   │   ├── 📄 backtest_1day_quick.log
    │   │   ├── 📄 backtest_30days.log
    │   │   └── 📄 backtest_30days_output.log
    │   ├── 📁 manifests/
    │   │   ├── 📄 excluded_paths.txt
    │   │   ├── 📋 export_meta.json
    │   │   └── 📄 sha256_manifest.txt
    │   ├── 📁 project_snapshot/
    │   │   ├── 📁 docs/
    │   │   │   ├── 📝 API.md
    │   │   │   ├── 📝 ARCHITECTURE.md
    │   │   │   ├── 📝 ASSUMPTIONS.md
    │   │   │   └── 📝 UI.md
    │   │   ├── 📁 examples/
    │   │   │   └── 🐍 generate_agent_example.py
    │   │   ├── 📁 monitoring/
    │   │   │   ├── 📁 dashboards/
    │   │   │   ├── 📋 dashboard.json
    │   │   │   ├── ⚙️ grafana-datasources.yml
    │   │   │   └── ⚙️ prometheus.yml
    │   │   ├── 📁 src/
    │   │   │   ├── 📁 hean/
    │   │   │   └── 📁 hean.egg-info/
    │   │   ├── 📁 templates/
    │   │   │   └── 📄 openai_process_factory_prompt.txt
    │   │   ├── 📁 tests/
    │   │   │   ├── 🐍 __init__.py
    │   │   │   ├── 🐍 test_adaptive_allocator.py
    │   │   │   ├── 🐍 test_adaptive_maker_router.py
    │   │   │   ├── 🐍 test_api.py
    │   │   │   ├── 🐍 test_api_e2e.py
    │   │   │   ├── 🐍 test_api_routers.py
    │   │   │   ├── 🐍 test_backtest.py
    │   │   │   ├── 🐍 test_backtest_execution_metrics.py
    │   │   │   ├── 🐍 test_capital_pressure.py
    │   │   │   ├── 🐍 test_config.py
    │   │   │   ├── 🐍 test_contracts.py
    │   │   │   ├── 🐍 test_decision_memory.py
    │   │   │   ├── 🐍 test_density.py
    │   │   │   ├── 🐍 test_dynamic_risk.py
    │   │   │   ├── 🐍 test_edge_confirmation.py
    │   │   │   ├── 🐍 test_edge_estimator.py
    │   │   │   ├── 🐍 test_evaluation.py
    │   │   │   ├── 🐍 test_execution_diagnostics.py
    │   │   │   ├── 🐍 test_execution_retry_queue.py
    │   │   │   ├── 🐍 test_execution_volatility_gating.py
    │   │   │   ├── 🐍 test_idempotency_resilience.py
    │   │   │   ├── 🐍 test_impulse_filters.py
    │   │   │   ├── 🐍 test_impulse_improvements.py
    │   │   │   ├── 🐍 test_impulse_precision.py
    │   │   │   ├── 🐍 test_maker_execution.py
    │   │   │   ├── 🐍 test_maker_retry_queue.py
    │   │   │   ├── 🐍 test_no_trade_fix.py
    │   │   │   ├── 🐍 test_no_trade_report.py
    │   │   │   ├── 🐍 test_no_trade_report_counters.py
    │   │   │   ├── 🐍 test_openai_factory_hardening.py
    │   │   │   ├── 🐍 test_paper_broker.py
    │   │   │   ├── 🐍 test_paper_broker_maker_fill_model.py
    │   │   │   ├── 🐍 test_paper_trade_assist.py
    │   │   │   ├── 🐍 test_process_factory_schemas.py
    │   │   │   ├── 🐍 test_process_factory_scorer.py
    │   │   │   ├── 🐍 test_process_factory_selector.py
    │   │   │   ├── 🐍 test_process_factory_storage.py
    │   │   │   ├── 🐍 test_regime.py
    │   │   │   ├── 🐍 test_risk.py
    │   │   │   ├── 🐍 test_selector_anti_overfitting.py
    │   │   │   ├── 🐍 test_smoke_test.py
    │   │   │   ├── 🐍 test_strategies.py
    │   │   │   ├── 🐍 test_strategy_accounting.py
    │   │   │   ├── 🐍 test_strategy_memory.py
    │   │   │   ├── 🐍 test_streams_smoke.py
    │   │   │   ├── 🐍 test_timeframes.py
    │   │   │   ├── 🐍 test_trade_density.py
    │   │   │   ├── 🐍 test_trade_diagnostics.py
    │   │   │   ├── 🐍 test_truth_layer.py
    │   │   │   └── 🐍 test_truth_layer_invariants.py
    │   │   ├── 📁 web/
    │   │   │   ├── 📄 .dockerignore
    │   │   │   ├── 📜 api-client.js
    │   │   │   ├── 🎨 command-center.css
    │   │   │   ├── 🌐 command-center.html
    │   │   │   ├── 📜 command-center.js
    │   │   │   ├── 🎨 dashboard.css
    │   │   │   ├── 🌐 dashboard.html
    │   │   │   ├── 📜 dashboard.js
    │   │   │   ├── ⚙️ docker-compose.yml
    │   │   │   ├── 📄 Dockerfile
    │   │   │   ├── 🌐 index.html
    │   │   │   ├── 📄 nginx.conf
    │   │   │   ├── 📝 QUICK_START.md
    │   │   │   ├── 📝 README.md
    │   │   │   ├── 📜 script.js
    │   │   │   ├── 🔧 start.sh
    │   │   │   └── 🎨 styles.css
    │   │   ├── 📄 .dockerignore
    │   │   ├── 📝 AGENT_GENERATION.md
    │   │   ├── 📝 AGENT_GENERATION_QUICKSTART.md
    │   │   ├── 📝 AUTO_IMPROVEMENT_SYSTEM.md
    │   │   ├── 📄 backtest_1day_quick.log
    │   │   ├── 📄 backtest_30days.log
    │   │   ├── 📄 backtest_30days_output.log
    │   │   ├── 📝 BACKTEST_30DAYS_RESULTS.md
    │   │   ├── 📝 BACKTEST_30DAYS_STATUS.md
    │   │   ├── 📝 BACKTEST_FIXES.md
    │   │   ├── 📝 BACKTEST_PROGRESS.md
    │   │   ├── 📝 BYBIT_API_SETUP.md
    │   │   ├── 📝 BYBIT_CONNECTION_FIXED.md
    │   │   ├── 📝 BYBIT_INTEGRATION_COMPLETE.md
    │   │   ├── 📝 BYBIT_KEYS_UPDATED.md
    │   │   ├── 📝 BYBIT_SETUP_GUIDE.md
    │   │   ├── 📝 BYBIT_TESTNET_RESULTS.md
    │   │   ├── 🔧 check_backtest_results.sh
    │   │   ├── 🐍 check_balance.py
    │   │   ├── 🐍 check_trading_status.py
    │   │   ├── 📝 COMMAND_CENTER_IMPLEMENTATION.md
    │   │   ├── 🐍 create_forensic_export.py
    │   │   ├── 📝 DEBUG_ORDER_FILL_DETECTION.md
    │   │   ├── 📝 DEBUG_ORDER_FILL_PROMPT.md
    │   │   ├── 📝 DIAGNOSTIC_REPORT.md
    │   │   ├── ⚙️ docker-compose.monitoring.yml
    │   │   ├── 📄 docker-compose.override.yml.example
    │   │   ├── ⚙️ docker-compose.yml
    │   │   ├── 📝 DOCKER_GUIDE.md
    │   │   ├── 📄 Dockerfile
    │   │   ├── 🔧 extract_backtest_results.sh
    │   │   ├── 🐍 extract_final_results.py
    │   │   ├── 📝 FORCE_SIGNAL_CALCULATION.md
    │   │   ├── 🐍 generate_agent.py
    │   │   ├── 🔧 get_backtest_stats.sh
    │   │   ├── 🐍 get_bybit_results.py
    │   │   ├── 🐍 get_local_trading_results.py
    │   │   ├── 🐍 get_order_results.py
    │   │   ├── 🐍 get_today_profit.py
    │   │   ├── 🐍 get_trading_summary.py
    │   │   ├── 📄 HEAN-project.zip
    │   │   ├── 📝 IMPLEMENTATION_SUMMARY.md
    │   │   ├── 📝 IMPROVEMENTS_ANALYSIS.md
    │   │   ├── 📝 IMPROVEMENTS_COMPLETED.md
    │   │   ├── 📝 IMPROVEMENTS_SUMMARY.md
    │   │   ├── 📄 Makefile
    │   │   ├── 📝 OPTIMIZATION_REPORT.md
    │   │   ├── 📝 PAPER_TRADE_ASSIST_IMPLEMENTATION.md
    │   │   ├── 📝 PERFORMANCE_IMPROVEMENTS.md
    │   │   ├── 📝 PRODUCTION_COMPLETE_PR.md
    │   │   ├── 📝 PRODUCTION_COMPLETE_SUMMARY.md
    │   │   ├── 📝 PRODUCTION_READY_SUMMARY.md
    │   │   ├── 📝 PROJECT_ANALYSIS_30DAYS.md
    │   │   ├── ⚙️ pyproject.toml
    │   │   ├── 📝 QUICK_START_DOCKER.md
    │   │   ├── 📝 QUICK_START_IMPROVEMENTS.md
    │   │   ├── 📝 README.md
    │   │   ├── 📝 SETUP_CHECKLIST.md
    │   │   ├── 🔧 show_results.sh
    │   │   ├── 📝 SMART_AGGRESSIVE_SYSTEM.md
    │   │   ├── 🔧 start_real_trading.sh
    │   │   ├── 🔧 start_trading.sh
    │   │   ├── 📝 STARVATION_FIX_SUMMARY.md
    │   │   ├── 🐍 test_500_orders.py
    │   │   ├── 🐍 test_500_orders_backtest.py
    │   │   ├── 🐍 test_bybit_connection.py
    │   │   ├── 🔧 wait_and_show_results.sh
    │   │   ├── 📝 WEB_DOCKER_SETUP.md
    │   │   ├── 📝 БЫСТРЫЙ_СТАРТ_РЕАЛЬНОЙ_ТОРГОВЛИ.md
    │   │   ├── 📝 ГОТОВНОСТЬ_К_ЗАПУСКУ.md
    │   │   ├── 📝 ГОТОВО.md
    │   │   ├── 📝 ДОБАВЛЕНИЕ_МОНЕТ.md
    │   │   ├── 📝 ЗАПУСК_САЙТА.md
    │   │   ├── 📝 ИСПРАВЛЕНИЕ_ОРДЕРОВ.md
    │   │   ├── 📝 РЕАЛЬНАЯ_ТОРГОВЛЯ.md
    │   │   ├── 📝 РЕАЛЬНАЯ_ТОРГОВЛЯ_АКТИВНА.md
    │   │   ├── 📝 СТАТУС_СИСТЕМЫ.md
    │   │   ├── 📝 ТОРГОВЛЯ_ЗАПУЩЕНА.md
    │   │   ├── 📝 ФИНАЛЬНАЯ_ПРОВЕРКА.md
    │   │   └── 📝 ЧТО_НУЖНО_ДЛЯ_ЗАПУСКА.md
    │   ├── 📁 reports/
    │   │   ├── 📄 docker_info.txt
    │   │   ├── 📄 file_inventory.csv
    │   │   ├── 📄 git_diff.patch
    │   │   ├── 📄 git_log.txt
    │   │   ├── 📄 git_status.txt
    │   │   ├── 📄 lint_last_run.txt
    │   │   ├── 📄 make_targets.txt
    │   │   ├── 📄 node_env.txt
    │   │   ├── 📄 python_env.txt
    │   │   ├── 📄 repo_tree.txt
    │   │   ├── 📄 runtime_smoke.txt
    │   │   └── 📄 tests_last_run.txt
    │   └── 📁 system/
    │       └── 📄 system_info.txt
    ├── 📁 logs/
    ├── 📁 monitoring/
    │   ├── 📁 dashboards/
    │   │   └── ⚙️ dashboard.yml
    │   ├── 📋 dashboard.json
    │   ├── ⚙️ grafana-datasources.yml
    │   └── ⚙️ prometheus.yml
    ├── 📁 src/
    │   ├── 📁 hean/
    │   │   ├── 📁 afo_core/
    │   │   ├── 📁 agent_generation/
    │   │   │   ├── 🐍 __init__.py
    │   │   │   ├── 🐍 capital_optimizer.py
    │   │   │   ├── 🐍 catalyst.py
    │   │   │   ├── 🐍 generator.py
    │   │   │   ├── 🐍 parameter_optimizer.py
    │   │   │   ├── 🐍 prompts.py
    │   │   │   └── 🐍 report_generator.py
    │   │   ├── 📁 api/
    │   │   │   ├── 📁 routers/
    │   │   │   ├── 📁 services/
    │   │   │   ├── 🐍 __init__.py
    │   │   │   ├── 🐍 app.py
    │   │   │   ├── 🐍 engine_facade.py
    │   │   │   ├── 🐍 reconcile.py
    │   │   │   ├── 🐍 schemas.py
    │   │   │   └── 🐍 server.py
    │   │   ├── 📁 backtest/
    │   │   │   ├── 🐍 __init__.py
    │   │   │   ├── 🐍 event_sim.py
    │   │   │   └── 🐍 metrics.py
    │   │   ├── 📁 core/
    │   │   │   ├── 📁 intelligence/
    │   │   │   ├── 📁 speed_engine/
    │   │   │   ├── 🐍 __init__.py
    │   │   │   ├── 🐍 bus.py
    │   │   │   ├── 🐍 clock.py
    │   │   │   ├── 🐍 context.py
    │   │   │   ├── 🐍 contracts.py
    │   │   │   ├── 🐍 density.py
    │   │   │   ├── 🐍 regime.py
    │   │   │   ├── 🐍 timeframes.py
    │   │   │   ├── 🐍 trade_density.py
    │   │   │   └── 🐍 types.py
    │   │   ├── 📁 evaluation/
    │   │   │   ├── 🐍 __init__.py
    │   │   │   ├── 🐍 readiness.py
    │   │   │   └── 🐍 truth_layer.py
    │   │   ├── 📁 exchange/
    │   │   │   ├── 📁 bybit/
    │   │   │   ├── 🐍 __init__.py
    │   │   │   ├── 🐍 models.py
    │   │   │   └── 🐍 synthetic_feed.py
    │   │   ├── 📁 execution/
    │   │   │   ├── 🐍 __init__.py
    │   │   │   ├── 🐍 edge_estimator.py
    │   │   │   ├── 🐍 execution_diagnostics.py
    │   │   │   ├── 🐍 maker_retry_queue.py
    │   │   │   ├── 🐍 order_manager.py
    │   │   │   ├── 🐍 paper_broker.py
    │   │   │   └── 🐍 router.py
    │   │   ├── 📁 hft/
    │   │   │   ├── 🐍 __init__.py
    │   │   │   ├── 🐍 circuit_breaker.py
    │   │   │   └── 🐍 shared_memory.py
    │   │   ├── 📁 income/
    │   │   │   └── 🐍 streams.py
    │   │   ├── 📁 observability/
    │   │   │   ├── 📁 monitoring/
    │   │   │   ├── 🐍 __init__.py
    │   │   │   ├── 🐍 health.py
    │   │   │   ├── 🐍 metrics.py
    │   │   │   ├── 🐍 metrics_exporter.py
    │   │   │   ├── 🐍 no_trade_report.py
    │   │   │   └── 🐍 prometheus_server.py
    │   │   ├── 📁 portfolio/
    │   │   │   ├── 🐍 __init__.py
    │   │   │   ├── 🐍 accounting.py
    │   │   │   ├── 🐍 allocator.py
    │   │   │   ├── 🐍 capital_pressure.py
    │   │   │   ├── 🐍 decision_memory.py
    │   │   │   ├── 🐍 profit_target_tracker.py
    │   │   │   ├── 🐍 rebalancer.py
    │   │   │   ├── 🐍 smart_reinvestor.py
    │   │   │   └── 🐍 strategy_memory.py
    │   │   ├── 📁 process_factory/
    │   │   │   ├── 📁 integrations/
    │   │   │   ├── 📁 processes/
    │   │   │   ├── 🐍 __init__.py
    │   │   │   ├── 🐍 engine.py
    │   │   │   ├── 🐍 evaluation.py
    │   │   │   ├── 🐍 leverage_engine.py
    │   │   │   ├── 🐍 process_quality.py
    │   │   │   ├── 🐍 registry.py
    │   │   │   ├── 🐍 report.py
    │   │   │   ├── 🐍 router.py
    │   │   │   ├── 🐍 sandbox.py
    │   │   │   ├── 🐍 schemas.py
    │   │   │   ├── 🐍 scorer.py
    │   │   │   ├── 🐍 selector.py
    │   │   │   ├── 🐍 storage.py
    │   │   │   └── 🐍 truth_layer.py
    │   │   ├── 📁 risk/
    │   │   │   ├── 🐍 __init__.py
    │   │   │   ├── 🐍 capital_preservation.py
    │   │   │   ├── 🐍 deposit_protector.py
    │   │   │   ├── 🐍 dynamic_risk.py
    │   │   │   ├── 🐍 kelly_criterion.py
    │   │   │   ├── 🐍 killswitch.py
    │   │   │   ├── 🐍 limits.py
    │   │   │   ├── 🐍 multi_level_protection.py
    │   │   │   ├── 🐍 position_sizer.py
    │   │   │   ├── 🐍 smart_leverage.py
    │   │   │   └── 🐍 tail_risk.py
    │   │   ├── 📁 strategies/
    │   │   │   ├── 📁 evolved/
    │   │   │   ├── 🐍 __init__.py
    │   │   │   ├── 🐍 base.py
    │   │   │   ├── 🐍 basis_arbitrage.py
    │   │   │   ├── 🐍 edge_confirmation.py
    │   │   │   ├── 🐍 enhanced_grid.py
    │   │   │   ├── 🐍 funding_harvester.py
    │   │   │   ├── 🐍 hf_scalping.py
    │   │   │   ├── 🐍 impulse_engine.py
    │   │   │   ├── 🐍 impulse_filters.py
    │   │   │   └── 🐍 momentum_trader.py
    │   │   ├── 🐍 __init__.py
    │   │   ├── 🐍 config.py
    │   │   ├── 🐍 logging.py
    │   │   ├── 🐍 main.py
    │   │   └── 🐍 paper_trade_assist.py
    │   └── 📁 hean.egg-info/
    │       ├── 📄 dependency_links.txt
    │       ├── 📄 PKG-INFO
    │       ├── 📄 requires.txt
    │       ├── 📄 SOURCES.txt
    │       └── 📄 top_level.txt
    ├── 📁 templates/
    │   └── 📄 openai_process_factory_prompt.txt
    ├── 📁 tests/
    │   ├── 🐍 __init__.py
    │   ├── 🐍 test_adaptive_allocator.py
    │   ├── 🐍 test_adaptive_maker_router.py
    │   ├── 🐍 test_api.py
    │   ├── 🐍 test_api_e2e.py
    │   ├── 🐍 test_api_routers.py
    │   ├── 🐍 test_backtest.py
    │   ├── 🐍 test_backtest_execution_metrics.py
    │   ├── 🐍 test_capital_pressure.py
    │   ├── 🐍 test_config.py
    │   ├── 🐍 test_contracts.py
    │   ├── 🐍 test_decision_memory.py
    │   ├── 🐍 test_density.py
    │   ├── 🐍 test_dynamic_risk.py
    │   ├── 🐍 test_edge_confirmation.py
    │   ├── 🐍 test_edge_estimator.py
    │   ├── 🐍 test_evaluation.py
    │   ├── 🐍 test_execution_diagnostics.py
    │   ├── 🐍 test_execution_retry_queue.py
    │   ├── 🐍 test_execution_volatility_gating.py
    │   ├── 🐍 test_idempotency_resilience.py
    │   ├── 🐍 test_impulse_filters.py
    │   ├── 🐍 test_impulse_improvements.py
    │   ├── 🐍 test_impulse_precision.py
    │   ├── 🐍 test_maker_execution.py
    │   ├── 🐍 test_maker_retry_queue.py
    │   ├── 🐍 test_no_trade_fix.py
    │   ├── 🐍 test_no_trade_report.py
    │   ├── 🐍 test_no_trade_report_counters.py
    │   ├── 🐍 test_openai_factory_hardening.py
    │   ├── 🐍 test_paper_broker.py
    │   ├── 🐍 test_paper_broker_maker_fill_model.py
    │   ├── 🐍 test_paper_trade_assist.py
    │   ├── 🐍 test_process_factory_schemas.py
    │   ├── 🐍 test_process_factory_scorer.py
    │   ├── 🐍 test_process_factory_selector.py
    │   ├── 🐍 test_process_factory_storage.py
    │   ├── 🐍 test_regime.py
    │   ├── 🐍 test_risk.py
    │   ├── 🐍 test_selector_anti_overfitting.py
    │   ├── 🐍 test_smoke_test.py
    │   ├── 🐍 test_strategies.py
    │   ├── 🐍 test_strategy_accounting.py
    │   ├── 🐍 test_strategy_memory.py
    │   ├── 🐍 test_streams_smoke.py
    │   ├── 🐍 test_timeframes.py
    │   ├── 🐍 test_trade_density.py
    │   ├── 🐍 test_trade_diagnostics.py
    │   ├── 🐍 test_truth_layer.py
    │   └── 🐍 test_truth_layer_invariants.py
    ├── 📁 web/
    │   ├── 📁 eureka_interface/
    │   │   └── 📁 dist/
    │   │       ├── 📁 assets/
    │   │       ├── 🌐 index.html
    │   │       └── 📄 vite.svg
    │   ├── 📄 .dockerignore
    │   ├── 📜 api-client.js
    │   ├── 🎨 command-center.css
    │   ├── 🌐 command-center.html
    │   ├── 📜 command-center.js
    │   ├── 🎨 dashboard.css
    │   ├── 🌐 dashboard.html
    │   ├── 📜 dashboard.js
    │   ├── ⚙️ docker-compose.yml
    │   ├── 📄 Dockerfile
    │   ├── 🌐 index.html
    │   ├── 📄 nginx.conf
    │   ├── 📝 QUICK_START.md
    │   ├── 📝 README.md
    │   ├── 📜 script.js
    │   ├── 🔧 start.sh
    │   └── 🎨 styles.css
    ├── 📄 .dockerignore
    ├── 📝 AGENT_GENERATION.md
    ├── 📝 AGENT_GENERATION_QUICKSTART.md
    ├── 📝 AGGRESSIVE_MODE_FIXES.md
    ├── 📝 AUTO_IMPROVEMENT_SYSTEM.md
    ├── 📄 backtest_1day_quick.log
    ├── 📄 backtest_30days.log
    ├── 📄 backtest_30days_output.log
    ├── 📝 BACKTEST_30DAYS_RESULTS.md
    ├── 📝 BACKTEST_30DAYS_STATUS.md
    ├── 📝 BACKTEST_FIXES.md
    ├── 📝 BACKTEST_PROGRESS.md
    ├── 📝 BYBIT_API_SETUP.md
    ├── 📝 BYBIT_CONNECTION_FIXED.md
    ├── 📝 BYBIT_INTEGRATION_COMPLETE.md
    ├── 📝 BYBIT_KEYS_UPDATED.md
    ├── 📝 BYBIT_SETUP_GUIDE.md
    ├── 📝 BYBIT_TESTNET_RESULTS.md
    ├── 🔧 check_backtest_results.sh
    ├── 🐍 check_balance.py
    ├── 🐍 check_trading_status.py
    ├── 📝 COMMAND_CENTER_IMPLEMENTATION.md
    ├── 🐍 create_forensic_export.py
    ├── 📝 CRITICAL_FIXES_REPORT.md
    ├── 📝 DEBUG_ORDER_FILL_DETECTION.md
    ├── 📝 DEBUG_ORDER_FILL_PROMPT.md
    ├── 🐍 debug_status.py
    ├── 🐍 diagnose_trading_issue.py
    ├── 📝 DIAGNOSTIC_REPORT.md
    ├── 📝 DIAGNOSTIC_SUMMARY.md
    ├── 🔧 docker-build-and-run.sh
    ├── ⚙️ docker-compose.monitoring.yml
    ├── 📄 docker-compose.override.yml.example
    ├── ⚙️ docker-compose.yml
    ├── 📝 DOCKER_GUIDE.md
    ├── 📄 Dockerfile
    ├── 📝 EXPORT_INSTRUCTIONS.md
    ├── 🔧 extract_backtest_results.sh
    ├── 🐍 extract_final_results.py
    ├── 🔧 fix_low_trading_activity.sh
    ├── 📝 FORCE_SIGNAL_CALCULATION.md
    ├── 🐍 generate_agent.py
    ├── 🐍 generate_tree.py
    ├── 🔧 get_backtest_stats.sh
    ├── 🐍 get_bybit_results.py
    ├── 🐍 get_real_profit.py
    ├── 🐍 get_trading_report.py
    ├── 📄 HEAN-project.zip
    ├── 📄 HEAN_FULL_EXPORT_20260103_044327.zip
    ├── 📝 IMPLEMENTATION_SUMMARY.md
    ├── 📝 IMPROVEMENTS_ANALYSIS.md
    ├── 📝 IMPROVEMENTS_COMPLETED.md
    ├── 📝 IMPROVEMENTS_SUMMARY.md
    ├── 📄 Makefile
    ├── 📝 OPTIMIZATION_REPORT.md
    ├── 📝 PAPER_TRADE_ASSIST_IMPLEMENTATION.md
    ├── 📝 PERFORMANCE_IMPROVEMENTS.md
    ├── 📝 PRODUCTION_COMPLETE_PR.md
    ├── 📝 PRODUCTION_COMPLETE_SUMMARY.md
    ├── 📝 PRODUCTION_READY_SUMMARY.md
    ├── 📝 PROJECT_ANALYSIS_30DAYS.md
    ├── 📄 PROJECT_STRUCTURE_TREE.txt
    ├── 📄 PROJECT_TREE.txt
    ├── ⚙️ pyproject.toml
    ├── 📝 QUICK_START_DOCKER.md
    ├── 📝 QUICK_START_IMPROVEMENTS.md
    ├── 📝 README.md
    ├── 📝 SETUP_CHECKLIST.md
    ├── 🔧 show_results.sh
    ├── 📝 SMART_AGGRESSIVE_SYSTEM.md
    ├── 🔧 start_real_trading.sh
    ├── 🔧 start_trading.sh
    ├── 📝 STARVATION_FIX_SUMMARY.md
    ├── 🐍 test_500_orders.py
    ├── 🐍 test_500_orders_backtest.py
    ├── 🐍 test_bybit_connection.py
    ├── 📄 trading.log
    ├── 🔧 wait_and_show_results.sh
    ├── 📝 WEB_DOCKER_SETUP.md
    ├── 📝 БЫСТРЫЙ_СТАРТ_РЕАЛЬНОЙ_ТОРГОВЛИ.md
    ├── 📝 ГОТОВНОСТЬ_К_ЗАПУСКУ.md
    ├── 📝 ГОТОВО.md
    ├── 📝 ДОБАВЛЕНИЕ_МОНЕТ.md
    ├── 📝 ЗАПУСК_САЙТА.md
    ├── 📝 ИСПРАВЛЕНИЕ_ОРДЕРОВ.md
    ├── 📝 ПРОБЛЕМА_МАЛО_ОРДЕРОВ.md
    ├── 📝 РЕАЛЬНАЯ_ТОРГОВЛЯ.md
    ├── 📝 РЕАЛЬНАЯ_ТОРГОВЛЯ_АКТИВНА.md
    ├── 📝 СТАТУС_СИСТЕМЫ.md
    ├── 📝 ТОРГОВЛЯ_ЗАПУЩЕНА.md
    ├── 📝 ФИНАЛЬНАЯ_ПРОВЕРКА.md
    └── 📝 ЧТО_НУЖНО_ДЛЯ_ЗАПУСКА.md
```
