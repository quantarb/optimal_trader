# Scaling Risks

| Path | Line | Pattern | Severity | Function | Message |
| --- | --- | --- | --- | --- | --- |
| domain/trades/panel.py | 127 | nested_loop | 1.00 | labels_panel_to_trades_df | Nested loops over trading dimensions can grow poorly with universe size. |
| pipeline/cohort_runner.py | 283 | nested_loop | 1.00 | _aggregate_walk_forward_rows | Nested loops over trading dimensions can grow poorly with universe size. |
| pipeline/test_support.py | 175 | nested_loop | 1.00 | seed_scalability_universe | Nested loops over trading dimensions can grow poorly with universe size. |
| analysis/alpha_flavors.py | 210 | loop_fit | 0.98 | _auto_select_k | `fit` is running inside a loop and likely scales poorly. |
| ml/autoencoder/vector_db.py | 242 | loop_fit | 0.98 | select_natural_k_fast_elbow | `fit` is running inside a loop and likely scales poorly. |
| analysis/historical_outcomes.py | 62 | nested_loop | 0.85 | enrich_similarity_matches_with_outcomes | Nested loops over trading dimensions can grow poorly with universe size. |
| analysis/llm_prompt_builder.py | 12 | nested_loop | 0.85 | _feature_sections_text | Nested loops over trading dimensions can grow poorly with universe size. |
| analysis/market_insight_schema.py | 289 | nested_loop | 0.85 | _row_canonical_features | Nested loops over trading dimensions can grow poorly with universe size. |
| analysis/oracle_reports.py | 279 | nested_loop | 0.85 | build_oracle_trade_report | Nested loops over trading dimensions can grow poorly with universe size. |
| analysis/oracle_reports.py | 308 | nested_loop | 0.85 | build_oracle_trade_report | Nested loops over trading dimensions can grow poorly with universe size. |
| analysis/oracle_reports.py | 414 | nested_loop | 0.85 | build_oracle_trade_report | Nested loops over trading dimensions can grow poorly with universe size. |
| analysis/research.py | 367 | nested_loop | 0.85 | build_feature_table_context | Nested loops over trading dimensions can grow poorly with universe size. |
| domain/trades/panel.py | 91 | nested_loop | 0.85 | labels_panel_to_trades_df | Nested loops over trading dimensions can grow poorly with universe size. |
| features/macro.py | 49 | nested_loop | 0.85 | _resolve_requested_series_codes | Nested loops over trading dimensions can grow poorly with universe size. |
| fmp/tests.py | 50 | nested_loop | 0.85 | test_period_endpoints_use_lowest_granularity_policy | Nested loops over trading dimensions can grow poorly with universe size. |
| fmp/views.py | 173 | nested_loop | 0.85 | _collect_columns | Nested loops over trading dimensions can grow poorly with universe size. |
| fmp/views.py | 759 | nested_loop | 0.85 | _filter_records_for_symbol | Nested loops over trading dimensions can grow poorly with universe size. |
| labels/views.py | 98 | nested_loop | 0.85 | _download_and_store_adjusted_prices | Nested loops over trading dimensions can grow poorly with universe size. |
| pipeline/cohort_runner.py | 269 | nested_loop | 0.85 | _aggregate_walk_forward_rows | Nested loops over trading dimensions can grow poorly with universe size. |
| pipeline/cohort_runner.py | 871 | nested_loop | 0.85 | run_walk_forward_model_cohort_backtests | Nested loops over trading dimensions can grow poorly with universe size. |
| pipeline/feature_presentation.py | 357 | nested_loop | 0.85 | serialize_features_for_embedding | Nested loops over trading dimensions can grow poorly with universe size. |
| pipeline/research_suite.py | 422 | nested_loop | 0.85 | _build_report_summary | Nested loops over trading dimensions can grow poorly with universe size. |
| pipeline/research_suite.py | 572 | nested_loop | 0.85 | run_optimal_trade_research_suite | Nested loops over trading dimensions can grow poorly with universe size. |
| pipeline/test_support.py | 172 | nested_loop | 0.85 | seed_scalability_universe | Nested loops over trading dimensions can grow poorly with universe size. |
| pipeline/tests.py | 482 | nested_loop | 0.85 | _build_insight_strategy_artifact | Nested loops over trading dimensions can grow poorly with universe size. |
| pipeline/views_insights.py | 355 | nested_loop | 0.85 | symbol_research_view | Nested loops over trading dimensions can grow poorly with universe size. |
| tools/product_quality_analysis/integrations/data_quality_runner.py | 107 | nested_loop | 0.85 | resolve_candidate_symbols | Nested loops over trading dimensions can grow poorly with universe size. |
| workflows/labels.py | 171 | nested_loop | 0.85 | _download_and_store_adjusted_prices | Nested loops over trading dimensions can grow poorly with universe size. |
| analysis/cluster_explanations.py | 56 | loop_groupby | 0.80 | build_cluster_feature_explanations | `groupby` is running inside a loop and likely scales poorly. |
| analysis/diagnostics.py | 46 | loop_groupby | 0.80 | _quantile_bucket_report | `groupby` is running inside a loop and likely scales poorly. |
| analysis/historical_outcomes.py | 14 | loop_groupby | 0.80 | _price_path_lookup | `groupby` is running inside a loop and likely scales poorly. |
| analysis/historical_outcomes.py | 17 | loop_sort_values | 0.80 | _price_path_lookup | `sort_values` is running inside a loop and likely scales poorly. |
| analysis/oracle_reports.py | 298 | loop_merge | 0.80 | build_oracle_trade_report | `merge` is running inside a loop and likely scales poorly. |
| analysis/oracle_reports.py | 359 | loop_sort_values | 0.80 | build_oracle_trade_report | `sort_values` is running inside a loop and likely scales poorly. |
| analysis/oracle_reports.py | 389 | loop_sort_values | 0.80 | build_oracle_trade_report | `sort_values` is running inside a loop and likely scales poorly. |
| analysis/situation_clustering.py | 119 | loop_groupby | 0.80 | fit_market_situation_clusters | `groupby` is running inside a loop and likely scales poorly. |
| analysis/situation_clustering.py | 184 | loop_sort_values | 0.80 | fit_market_situation_clusters | `sort_values` is running inside a loop and likely scales poorly. |
| backtest/strategies/stateful.py | 202 | loop_sort_values | 0.80 | compute_weights | `sort_values` is running inside a loop and likely scales poorly. |
| backtest/strategies/stateful.py | 203 | loop_sort_values | 0.80 | compute_weights | `sort_values` is running inside a loop and likely scales poorly. |
| domain/trades/optimal.py | 338 | loop_groupby | 0.80 | solve_joint_trades_by_frequency | `groupby` is running inside a loop and likely scales poorly. |
| domain/trades/optimal.py | 394 | loop_groupby | 0.80 | solve_trades_by_frequency | `groupby` is running inside a loop and likely scales poorly. |
| domain/trades/panel.py | 73 | loop_groupby | 0.80 | labels_panel_to_trades_df | `groupby` is running inside a loop and likely scales poorly. |
| domain/trades/panel.py | 74 | loop_sort_values | 0.80 | labels_panel_to_trades_df | `sort_values` is running inside a loop and likely scales poorly. |
| domain/trades/panel.py | 106 | loop_groupby | 0.80 | labels_panel_to_trades_df | `groupby` is running inside a loop and likely scales poorly. |
| domain/trades/panel.py | 107 | loop_sort_values | 0.80 | labels_panel_to_trades_df | `sort_values` is running inside a loop and likely scales poorly. |
| features/fundamentals.py | 48 | loop_merge | 0.80 | fetch_fundamentals_data | `merge` is running inside a loop and likely scales poorly. |
| fmp/views.py | 271 | loop_sort_values | 0.80 | _fetch_economic_indicators_from_api | `sort_values` is running inside a loop and likely scales poorly. |
| infra/fmp/client.py | 188 | loop_groupby | 0.80 | fundamentals_to_daily_panel | `groupby` is running inside a loop and likely scales poorly. |
| ml/artifact_datasets.py | 78 | loop_load_artifact_csv_frame | 0.80 | _join_feature_panels | `load_artifact_csv_frame` is running inside a loop and likely scales poorly. |
| ml/artifact_datasets.py | 83 | loop_merge | 0.80 | _join_feature_panels | `merge` is running inside a loop and likely scales poorly. |
