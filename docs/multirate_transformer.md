# Multi-rate multi-asset transformer

The reusable implementation lives under `scripts/multirate_transformer/`.
It provides:

- annual bidirectional issuer memory;
- quarterly bidirectional issuer memory conditioned on annual memory;
- a shared causal daily decoder with separate annual and quarterly cross-attention;
- asset-specific adapters for equity, corporate bonds/`note_bond`, preferred shares,
  warrants, and the other non-option classes represented by the existing related
  asset panels;
- dynamic task-head construction from the current MTL registry;
- raw HITS plus same-issuer and same-asset-class rank-label generation;
- point-in-time as-of alignment, masked ranking losses, and date-aware batches.

The authoritative feature-family audit is generated from the warehouse panel
`index.csv` and family metadata. For the current 1T cache, the generated report is:

`artifacts/multirate_transformer/feature_family_audit_1t.csv`

The resolved task inventory is:

`artifacts/multirate_transformer/task_inventory.csv`

The new warehouse adapter is `warehouse_data.py`. It consumes the same persisted
daily feature-family panels as the existing models, outer-joins family coverage,
filters the existing TA families, and derives causal annual/quarterly cadence
views from each row's preserved warehouse observation date. For raw warehouse
fundamental frames, `filing_date`/`accepted_date` can be normalized into that
observation date before constructing windows; fiscal period end dates are never
used as a substitute for availability.

Run the audits with:

```bash
python scripts/audit_multirate_transformer.py \
  --index <feature-family-cache>/index.csv \
  --output artifacts/multirate_transformer/feature_family_audit.csv

python scripts/audit_multirate_tasks.py \
  --output artifacts/multirate_transformer/task_inventory.csv
```
