# Dependency Hotspots

- Engine: `networkx`

| Module | Path | Fan In | Fan Out | PageRank | Betweenness | Score |
| --- | --- | --- | --- | --- | --- | --- |
| pipeline.models | pipeline/models.py | 42 | 0 | 0.0367 | 0.0000 | 87.670 |
| fmp.models | fmp/models.py | 36 | 0 | 0.0459 | 0.0000 | 76.588 |
| fmp.endpoints.base | fmp/endpoints/base.py | 31 | 0 | 0.0327 | 0.0000 | 65.274 |
| pipeline.services | pipeline/services.py | 13 | 9 | 0.0081 | 0.0219 | 36.353 |
| fmp.endpoints.helpers | fmp/endpoints/helpers.py | 17 | 0 | 0.0128 | 0.0000 | 35.284 |
| features.section_utils | features/section_utils.py | 16 | 1 | 0.0119 | 0.0000 | 34.195 |
| pipeline.service_runtime | pipeline/service_runtime.py | 15 | 2 | 0.0085 | 0.0008 | 32.870 |
| pipeline.tests | pipeline/tests.py | 0 | 32 | 0.0016 | 0.0000 | 32.155 |
| fmp.endpoints.registry | fmp/endpoints/registry.py | 0 | 31 | 0.0016 | 0.0000 | 31.155 |
| tools.performance_analysis.models | tools/performance_analysis/models.py | 14 | 0 | 0.0095 | 0.0000 | 28.948 |
| ml.execution | ml/execution.py | 8 | 8 | 0.0136 | 0.0084 | 25.572 |
| ml.base | ml/base.py | 10 | 2 | 0.0107 | 0.0007 | 23.091 |
| tools.performance_analysis.utils.report_utils | tools/performance_analysis/utils/report_utils.py | 11 | 0 | 0.0062 | 0.0000 | 22.624 |
| features.feature_builders | features/feature_builders.py | 2 | 18 | 0.0023 | 0.0059 | 22.377 |
| analysis.market_state | analysis/market_state.py | 8 | 3 | 0.0139 | 0.0027 | 20.458 |
| pipeline.research_suite | pipeline/research_suite.py | 8 | 3 | 0.0057 | 0.0025 | 19.635 |
| analysis.market_insight_schema | analysis/market_insight_schema.py | 8 | 2 | 0.0103 | 0.0013 | 19.059 |
| pipeline.feature_presentation | pipeline/feature_presentation.py | 7 | 2 | 0.0113 | 0.0021 | 17.182 |
| ml.models | ml/models.py | 8 | 0 | 0.0099 | 0.0000 | 16.985 |
| domain.features.specs | domain/features/specs.py | 8 | 0 | 0.0093 | 0.0000 | 16.935 |
| tools.performance_analysis.config | tools/performance_analysis/config.py | 8 | 0 | 0.0056 | 0.0000 | 16.559 |
| pipeline.contracts | pipeline/contracts.py | 8 | 0 | 0.0054 | 0.0000 | 16.541 |
| pipeline.views | pipeline/views.py | 2 | 12 | 0.0029 | 0.0012 | 16.322 |
| tools.performance_analysis.cli | tools/performance_analysis/cli.py | 0 | 16 | 0.0016 | 0.0000 | 16.155 |
| tools.code_analysis.repository | tools/code_analysis/repository.py | 7 | 1 | 0.0091 | 0.0001 | 15.914 |

## Cycles

- fmp.tasks -> fmp.views
