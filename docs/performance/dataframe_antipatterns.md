# DataFrame Anti-Patterns

| Path | Line | Pattern | Severity | Function | Message |
| --- | --- | --- | --- | --- | --- |
| analysis/alpha_flavors.py | 639 | iterrows | 0.95 | cluster_alpha_flavors | Row-wise iteration is usually a vectorization candidate. |
| analysis/alpha_flavors.py | 1029 | iterrows | 0.95 | add_cluster_explanations | Row-wise iteration is usually a vectorization candidate. |
| analysis/oracle_reports.py | 279 | iterrows | 0.95 | build_oracle_trade_report | Row-wise iteration is usually a vectorization candidate. |
| analysis/oracle_reports.py | 308 | iterrows | 0.95 | build_oracle_trade_report | Row-wise iteration is usually a vectorization candidate. |
| analysis/oracle_reports.py | 359 | iterrows | 0.95 | build_oracle_trade_report | Row-wise iteration is usually a vectorization candidate. |
| analysis/oracle_reports.py | 389 | iterrows | 0.95 | build_oracle_trade_report | Row-wise iteration is usually a vectorization candidate. |
| labels/views.py | 325 | iterrows | 0.95 | _build_trade_aggregates | Row-wise iteration is usually a vectorization candidate. |
| labels/views.py | 356 | iterrows | 0.95 | _build_trade_aggregates | Row-wise iteration is usually a vectorization candidate. |
| ml/frameworks/transformers/seq2seq.py | 467 | iterrows | 0.95 | prepare_entry2exit_dataset | Row-wise iteration is usually a vectorization candidate. |
| utils/llm_prompts.py | 26 | iterrows | 0.95 | build_llm_guardrail_prompt_from_results | Row-wise iteration is usually a vectorization candidate. |
| analysis/cluster_explanations.py | 56 | groupby | 0.90 | build_cluster_feature_explanations | `groupby` inside a loop is a scaling risk. |
| analysis/diagnostics.py | 46 | groupby | 0.90 | _quantile_bucket_report | `groupby` inside a loop is a scaling risk. |
| analysis/historical_outcomes.py | 14 | groupby | 0.90 | _price_path_lookup | `groupby` inside a loop is a scaling risk. |
| analysis/historical_outcomes.py | 17 | sort_values | 0.90 | _price_path_lookup | `sort_values` inside a loop is a scaling risk. |
| analysis/oracle_reports.py | 298 | merge | 0.90 | build_oracle_trade_report | `merge` inside a loop is a scaling risk. |
| analysis/oracle_reports.py | 359 | sort_values | 0.90 | build_oracle_trade_report | `sort_values` inside a loop is a scaling risk. |
| analysis/oracle_reports.py | 389 | sort_values | 0.90 | build_oracle_trade_report | `sort_values` inside a loop is a scaling risk. |
| analysis/situation_clustering.py | 119 | groupby | 0.90 | fit_market_situation_clusters | `groupby` inside a loop is a scaling risk. |
| analysis/situation_clustering.py | 184 | sort_values | 0.90 | fit_market_situation_clusters | `sort_values` inside a loop is a scaling risk. |
| backtest/backtest.py | 88 | sort_index | 0.90 | build_panel_from_daily_by_symbol | `sort_index` inside a loop is a scaling risk. |
| backtest/latest.py | 495 | sort_index | 0.90 | run_panel_prediction_custom | `sort_index` inside a loop is a scaling risk. |
| backtest/strategies/stateful.py | 202 | sort_values | 0.90 | compute_weights | `sort_values` inside a loop is a scaling risk. |
| backtest/strategies/stateful.py | 203 | sort_values | 0.90 | compute_weights | `sort_values` inside a loop is a scaling risk. |
| domain/trades/optimal.py | 338 | groupby | 0.90 | solve_joint_trades_by_frequency | `groupby` inside a loop is a scaling risk. |
| domain/trades/optimal.py | 394 | groupby | 0.90 | solve_trades_by_frequency | `groupby` inside a loop is a scaling risk. |
| domain/trades/panel.py | 73 | groupby | 0.90 | labels_panel_to_trades_df | `groupby` inside a loop is a scaling risk. |
| domain/trades/panel.py | 74 | sort_values | 0.90 | labels_panel_to_trades_df | `sort_values` inside a loop is a scaling risk. |
| domain/trades/panel.py | 106 | groupby | 0.90 | labels_panel_to_trades_df | `groupby` inside a loop is a scaling risk. |
| domain/trades/panel.py | 107 | sort_values | 0.90 | labels_panel_to_trades_df | `sort_values` inside a loop is a scaling risk. |
| features/fundamentals.py | 48 | merge | 0.90 | fetch_fundamentals_data | `merge` inside a loop is a scaling risk. |
| fmp/views.py | 271 | sort_values | 0.90 | _fetch_economic_indicators_from_api | `sort_values` inside a loop is a scaling risk. |
| infra/fmp/client.py | 188 | groupby | 0.90 | fundamentals_to_daily_panel | `groupby` inside a loop is a scaling risk. |
| infra/fmp/client.py | 189 | sort_index | 0.90 | fundamentals_to_daily_panel | `sort_index` inside a loop is a scaling risk. |
| ml/artifact_datasets.py | 83 | merge | 0.90 | _join_feature_panels | `merge` inside a loop is a scaling risk. |
| ml/autoencoder/diagnostics.py | 183 | sort_values | 0.90 | summarize_event_windows | `sort_values` inside a loop is a scaling risk. |
| ml/autoencoder/diagnostics.py | 212 | sort_values | 0.90 | analyze_event_feature_breaks | `sort_values` inside a loop is a scaling risk. |
| ml/autoencoder/diagnostics.py | 223 | sort_values | 0.90 | analyze_event_feature_breaks | `sort_values` inside a loop is a scaling risk. |
| ml/views.py | 35 | groupby | 0.90 | _prediction_symbol_rows | `groupby` inside a loop is a scaling risk. |
| pipeline/strategy_definitions.py | 213 | groupby | 0.90 | apply_strategy_definition | `groupby` inside a loop is a scaling risk. |
| pipeline/strategy_definitions.py | 233 | sort_values | 0.90 | apply_strategy_definition | `sort_values` inside a loop is a scaling risk. |
| pipeline/strategy_definitions.py | 249 | sort_values | 0.90 | apply_strategy_definition | `sort_values` inside a loop is a scaling risk. |
| analysis/situation_clustering.py | 121 | apply | 0.85 | fit_market_situation_clusters | DataFrame.apply often hides Python-level loops. |
| analysis/diagnostics.py | 293 | materialization | 0.75 | build_diagnostic_report | Converting frames to Python objects materializes data and can inflate memory. |
| analysis/feature_reasoning.py | 104 | materialization | 0.75 | summarize_feature_changes | Converting frames to Python objects materializes data and can inflate memory. |
| analysis/feature_reasoning.py | 105 | materialization | 0.75 | summarize_feature_changes | Converting frames to Python objects materializes data and can inflate memory. |
| analysis/historical_situation_search.py | 207 | materialization | 0.75 | search_numeric_neighbors | Converting frames to Python objects materializes data and can inflate memory. |
| analysis/historical_situation_search.py | 241 | materialization | 0.75 | search_embedding_neighbors | Converting frames to Python objects materializes data and can inflate memory. |
| analysis/historical_situation_search.py | 393 | materialization | 0.75 | search_hybrid_neighbors | Converting frames to Python objects materializes data and can inflate memory. |
| analysis/insights.py | 277 | materialization | 0.75 | build_opportunity_dashboard | Converting frames to Python objects materializes data and can inflate memory. |
| analysis/similarity_engine.py | 81 | materialization | 0.75 | find_similar_market_states | Converting frames to Python objects materializes data and can inflate memory. |
