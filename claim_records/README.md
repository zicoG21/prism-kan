# Claim Records

`claim_record.csv` is the row-level submission format.

Required columns:

```text
task_id, adapter, seed, evidence_object, claim_type, target,
scorer, rank, margin, predicate, pass, protocol
```

Each row answers one structural question for one evidence object.  A method can
therefore pass a fitted-function pair claim while failing an exposed-readout
endpoint claim on the same task and seed.

