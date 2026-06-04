# Claim Records

`claim_record.csv` is the row-level official-scoring output.  A workflow
adapter exposes raw evidence objects first; the official scorer then recomputes
rank, margin, predicate, and pass/fail fields.

The workshop quick path writes two files:

- `released_adapter_outputs.csv`: raw standardized evidence from available
  experiment CSVs.
- `released_claim_records.csv`: official scorer output derived from those raw
  evidence rows.

Core claim-record columns:

```text
task_id, adapter, seed, evidence_object, claim_type, target,
scorer, rank, margin, predicate, pass, protocol
```

Each row answers one structural question for one evidence object.  A method can
therefore pass a fitted-function pair claim while failing an exposed-readout
endpoint claim on the same task and seed.  The pass/fail column should be read
as an official scorer decision, not as an adapter-authored label.
