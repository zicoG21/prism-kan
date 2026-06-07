# Task Cards

Task cards are the fixed benchmark inputs.  A task card declares:

- the formula or covariate generator;
- train/test seeds and task dimensions;
- the known active support;
- legal structural claims for the task;
- official scorers and predicates for those claims.

Different valid interpretations of one formula are separate task cards.  A
submission should not reinterpret a task after seeing a method's output.

ClaimTransfer-Bench 1.0 includes a public diagnostic registry, an offline
hidden-template registry, and a standard-formula claim suite.  The historical
file names retain `v0`/`v1` for compatibility with earlier result packs, but the
release contract treats them as the 1.0 benchmark registries.  Validate the task
cards with:

```bash
python scripts/validate_task_cards.py
```

The public registry is a diagnostic split.  Hidden cards or private seeds use
the same task-card schema for offline private evaluation.

Contract files:

- `task_card_schema.json`: machine-readable schema.
- `claimtransfer_v0_public.json`: public diagnostic registry.
- `claimtransfer_v0_hidden_template.json`: offline hidden-split template.
- `claimtransfer_v1_standard_formula_public.json`: 30 base standard formulas,
  expandable to 90 standard-formula settings with noise and high-dimensional
  variants.
- `claimtransfer_v1_scientific_templates.json`: scientific/private-card
  templates.
- `docs/task_card_authoring_protocol.md`: authoring and versioning rules.
