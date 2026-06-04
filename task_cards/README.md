# Task Cards

Task cards are the fixed benchmark inputs.  A task card declares:

- the formula or covariate generator;
- train/test seeds and task dimensions;
- the known active support;
- legal structural claims for the task;
- official scorers and predicates for those claims.

Different valid interpretations of one formula are separate task cards.  A
submission should not reinterpret a task after seeing a method's output.

Workshop v0 includes `claimtransfer_v0_public.json`, a compact public registry
for the formula and semi-synthetic cards used in the paper.  Validate the task
cards with:

```bash
python scripts/validate_task_cards.py
```

The public registry is a diagnostic split.  Hidden cards or private seeds can
use the same task-card schema for leaderboard-style evaluation.
