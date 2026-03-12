# Dependency Hotspots

- Engine: `networkx`

| Module | Path | Fan In | Fan Out | PageRank | Betweenness | Score |
| --- | --- | --- | --- | --- | --- | --- |
| pipeline.models | pipeline/models.py | 43 | 0 | 0.0346 | 0.0000 | 89.463 |
| fmp.models | fmp/models.py | 37 | 0 | 0.0423 | 0.0000 | 78.231 |
| fmp.endpoints.base | fmp/endpoints/base.py | 31 | 0 | 0.0297 | 0.0000 | 64.975 |
| tools.product_quality_analysis.models | tools/product_quality_analysis/models.py | 23 | 0 | 0.0267 | 0.0000 | 48.670 |
| pipeline.services | pipeline/services.py | 13 | 9 | 0.0074 | 0.0188 | 36.210 |
| fmp.endpoints.helpers | fmp/endpoints/helpers.py | 17 | 0 | 0.0117 | 0.0000 | 35.167 |
| features.section_utils | features/section_utils.py | 16 | 1 | 0.0109 | 0.0000 | 34.086 |
| pipeline.service_runtime | pipeline/service_runtime.py | 15 | 2 | 0.0077 | 0.0006 | 32.790 |
| pipeline.tests | pipeline/tests.py | 0 | 32 | 0.0014 | 0.0000 | 32.141 |
| fmp.endpoints.registry | fmp/endpoints/registry.py | 0 | 31 | 0.0014 | 0.0000 | 31.141 |
| tools.performance_analysis.models | tools/performance_analysis/models.py | 14 | 0 | 0.0086 | 0.0000 | 28.862 |
| ml.execution | ml/execution.py | 8 | 8 | 0.0124 | 0.0071 | 25.418 |
| tools.product_quality_analysis.cli | tools/product_quality_analysis/cli.py | 0 | 25 | 0.0014 | 0.0000 | 25.141 |
| ml.base | ml/base.py | 10 | 2 | 0.0116 | 0.0006 | 23.176 |
| tools.performance_analysis.utils.report_utils | tools/performance_analysis/utils/report_utils.py | 11 | 0 | 0.0057 | 0.0000 | 22.567 |
| features.feature_builders | features/feature_builders.py | 2 | 18 | 0.0021 | 0.0050 | 22.334 |
| analysis.market_state | analysis/market_state.py | 8 | 3 | 0.0126 | 0.0023 | 20.319 |
| pipeline.research_suite | pipeline/research_suite.py | 8 | 3 | 0.0052 | 0.0018 | 19.564 |
| analysis.market_insight_schema | analysis/market_insight_schema.py | 8 | 2 | 0.0093 | 0.0010 | 18.959 |
| pipeline.cohort_runner | pipeline/cohort_runner.py | 6 | 5 | 0.0059 | 0.0046 | 17.711 |
| pipeline.feature_presentation | pipeline/feature_presentation.py | 7 | 2 | 0.0103 | 0.0017 | 17.069 |
| ml.models | ml/models.py | 8 | 0 | 0.0090 | 0.0000 | 16.896 |
| domain.features.specs | domain/features/specs.py | 8 | 0 | 0.0085 | 0.0000 | 16.853 |
| tools.performance_analysis.config | tools/performance_analysis/config.py | 8 | 0 | 0.0051 | 0.0000 | 16.508 |
| pipeline.contracts | pipeline/contracts.py | 8 | 0 | 0.0049 | 0.0000 | 16.492 |

## Cycles

- fmp.views -> fmp.tasks
