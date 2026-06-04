# ClaimTransfer Task-Card Authoring Protocol

This protocol freezes how benchmark cards declare structural claims.  A method
may disagree with a card, but it may not reinterpret the card after seeing its
own output.

## Required Authoring Rules

1. **Support rule.** The active support is the set of variables appearing in the
   canonical data-generating expression after deterministic simplification.

2. **Pair rule.** Pair claims are declared only for explicit bivariate terms or
   bivariate components that the card author intends a pair scorer to license.

3. **Multi-pair rule.** If a formula contains multiple declared pair claims,
   each pair appears as a separate legal claim with its own target and official
   scorer.  Multi-pair cards should use `rank_at_budget` rather than silently
   reducing the task to one favorite pair.

4. **Nested/compositional rule.** Nested, compositional, or three-way formulas
   may be marked as scorer-stress cards.  A failed pairwise verdict on such a
   card is a pair-grammar stress result, not a full formula-recovery verdict.

5. **Symbolic rule.** Symbolic status is a separate claim type.  A syntactic
   symbolic event does not license sparse support, endpoint, pair, expression
   equivalence, or operator-recovery claims unless those claims are separately
   scored.

6. **Alternative-grammar rule.** If two structural interpretations are both
   scientifically defensible, they become separate task cards or separate
   scorer-indexed claim records.  They are not merged into one post-hoc label.

7. **Split rule.** Public diagnostic cards may expose formulas, labels, and
   scorers.  Hidden or private cards may withhold seeds, labels, or formulas, but
   they must use the same schema and official scorer implementation.

8. **Versioning rule.** Any change to formula, labels, claim grammar, official
   scorer, or predicate creates a new registry version.  Silent edits are not
   allowed.

## Minimal Card Fields

Every card must declare:

- `task_id`
- `task_family`
- `split`
- `registry_version`
- `formula`
- `covariates`
- `dimension`
- `samples`
- `support`
- `claim_specification`
- `seed_policy`
- `stress_tags`

Validate cards with:

```bash
python scripts/validate_task_cards.py
```

The validator enforces:

- unique `task_id` values inside each registry;
- unique `claim_id` values inside each registry;
- recognized claim buckets and claim types;
- recognized official scorers and predicates;
- support, endpoint, and pair targets inside the declared dimension;
- endpoint and pair targets as subsets of the declared support;
- numeric thresholds for thresholded symbolic/expression predicates;
- positive integer budgets for `top_m_contains_all`.

## Example Interpretations

- Product card: declare support, endpoints, and one product-like pair claim.
- Division/rational card: declare all intended bivariate pair claims separately.
- Nested-trig card: declare support plus pair-scorer stress, not a universal
  pair-recovery failure.
- Three-way card: declare support and optional pair-stress claims; expression or
  higher-order claims should be separate claim types.
