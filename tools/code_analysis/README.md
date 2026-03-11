# Code Analysis Toolkit

This toolkit generates repo-wide analysis reports for refactoring and architectural review.

It is library-backed rather than purely heuristic:

- `grimp` + `networkx` for import graphs and cycle detection
- AST inventory for module/function/class extraction
- `vulture` for dead code detection
- `radon` for complexity and maintainability metrics
- `sentence-transformers` + `faiss` when a local model is available, with a local hashing-vectorizer fallback for fully offline runs
- `typer` + `rich` for CLI execution and readable terminal output

## Install

Run inside the project environment:

```bash
/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m pip install \
  vulture radon grimp sentence-transformers faiss-cpu pydeps typer rich networkx matplotlib
```

The toolkit now degrades gracefully when some optional libraries are missing:

- no `radon` -> AST fallback metrics
- no `grimp` -> inventory import graph only
- no `vulture` -> import-graph dead-module detection only
- no `faiss` -> numpy similarity fallback
- no `networkx` -> custom graph analysis with simplified SVG output

## Main Commands

Run from the repository root:

```bash
/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis analyze_repo --root . --output data/code_analysis
/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis generate_dependency_graph --root . --output data/code_analysis
/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis generate_call_graph --root . --output data/code_analysis
/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis detect_duplicate_code --root . --output data/code_analysis
/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis detect_dead_code --root . --output data/code_analysis
/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis analyze_complexity --root . --output data/code_analysis
/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis build_semantic_index --root . --output data/code_analysis
/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis search_code "model training pipeline" --output data/code_analysis
/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis generate_repo_overview --output data/code_analysis
```

## Outputs

The default output directory is `data/code_analysis/`.

Primary reports:

- `inventory.json`
- `dependency_graph.json`
- `dependency_graph.md`
- `dependency_graph.svg`
- `call_graph.json`
- `call_graph.md`
- `duplicate_code_report.json`
- `duplicate_code_report.md`
- `dead_code_report.json`
- `dead_code_report.md`
- `code_metrics_report.json`
- `code_metrics_report.md`
- `module_responsibility_report.json`
- `module_responsibility_report.md`
- `semantic_index.faiss`
- `semantic_chunks.json`
- `semantic_embeddings.npy`
- `semantic_index.md`
- `repo_overview.json`
- `repo_overview.md`
- `refactoring_hints.json`
- `refactoring_hints.md`
- `semantic_search_<query>.json`
- `semantic_search_<query>.md`

## Report Intent

- `dependency_graph.*`
  - module dependencies
  - strongly connected dependency groups
  - central modules
- `call_graph.*`
  - function call relationships
  - high fan-in / fan-out nodes
  - major call pipelines
- `duplicate_code_report.*`
  - semantically similar functions/classes
  - clusters of repeated workflows
- `dead_code_report.*`
  - Vulture-based unused items
  - inbound-import-free module candidates
- `code_metrics_report.*`
  - cyclomatic complexity
  - maintainability index
  - largest files
- `module_responsibility_report.*`
  - mixed-concern modules
  - likely split candidates
- `repo_overview.*`
  - top-level architectural summary
  - refactor targets
- `refactoring_hints.*`
  - prioritized actions from the combined reports

## Notes

- Semantic duplicate detection and semantic search share the same embedding model.
- Offline default: the toolkit uses a local hashing-vectorizer embedding backend. To force a local `sentence-transformers` model, set `CODE_ANALYSIS_USE_SENTENCE_TRANSFORMERS=1` and pass a local model path or a cached model name.
- `dependency_graph.svg` is rendered with `matplotlib` because the Graphviz `dot` binary is not required.
- Dead code output is advisory. Framework entrypoints and reflection-heavy code still need human review.
