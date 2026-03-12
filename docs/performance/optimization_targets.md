# Optimization Targets

| Path | Runtime | Memory | Complexity | Centrality | Scaling | Total | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pipeline/services.py | 0.99 | 0.00 | 0.12 | 0.27 | 0.00 | 0.40 | runtime hotspot, high complexity, central dependency |
| pipeline/scalability.py | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.35 | runtime hotspot |
| analysis/oracle_reports.py | 0.00 | 0.00 | 0.71 | 0.00 | 1.00 | 0.26 | high complexity, scaling risk |
| analysis/alpha_flavors.py | 0.00 | 0.00 | 1.00 | 0.00 | 0.20 | 0.18 | high complexity, scaling risk |
| pipeline/models.py | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.15 | central dependency |
| fmp/models.py | 0.00 | 0.00 | 0.00 | 0.85 | 0.00 | 0.13 | central dependency |
| workflows/feature_runtime.py | 0.36 | 0.00 | 0.00 | 0.00 | 0.00 | 0.13 | runtime hotspot |
| pipeline/artifact_support.py | 0.00 | 0.00 | 0.74 | 0.00 | 0.00 | 0.11 | high complexity |
| domain/trades/panel.py | 0.00 | 0.00 | 0.00 | 0.00 | 0.74 | 0.11 | scaling risk |
| fmp/endpoints/base.py | 0.00 | 0.00 | 0.00 | 0.66 | 0.00 | 0.10 | central dependency |
| pipeline/research_suite.py | 0.00 | 0.00 | 0.50 | 0.04 | 0.09 | 0.09 | high complexity, central dependency, scaling risk |
| pipeline/cohort_runner.py | 0.00 | 0.00 | 0.42 | 0.02 | 0.18 | 0.09 | high complexity, central dependency, scaling risk |
| pipeline/experiments.py | 0.00 | 0.00 | 0.56 | 0.00 | 0.00 | 0.08 | high complexity |
| fmp/views.py | 0.00 | 0.00 | 0.27 | 0.00 | 0.25 | 0.08 | high complexity, scaling risk |
| pipeline/service_jobs_data.py | 0.14 | 0.00 | 0.13 | 0.00 | 0.00 | 0.07 | runtime hotspot, high complexity |
| tools/product_quality_analysis/models.py | 0.00 | 0.00 | 0.00 | 0.44 | 0.00 | 0.07 | central dependency |
| features/views.py | 0.00 | 0.00 | 0.40 | 0.00 | 0.00 | 0.06 | high complexity |
| analysis/diagnostics.py | 0.00 | 0.00 | 0.22 | 0.00 | 0.16 | 0.06 | high complexity, scaling risk |
| analysis/historical_outcomes.py | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 0.05 | scaling risk |
| analysis/situation_clustering.py | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 0.05 | scaling risk |
| domain/features/technical.py | 0.14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | runtime hotspot |
| pipeline/service_runtime.py | 0.01 | 0.05 | 0.00 | 0.22 | 0.00 | 0.05 | runtime hotspot, memory hotspot, central dependency |
| workflows/features.py | 0.13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | runtime hotspot |
| analysis/insights.py | 0.00 | 0.00 | 0.28 | 0.00 | 0.00 | 0.04 | high complexity |
| pipeline/service_jobs_modeling.py | 0.00 | 0.00 | 0.26 | 0.00 | 0.00 | 0.04 | high complexity |
