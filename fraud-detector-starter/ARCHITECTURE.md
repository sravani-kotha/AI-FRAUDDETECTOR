# Explainable Checkout Fraud Risk Engine

```text
Checkout transaction
        |
        v
Feature validation and ordering (single FEATURE_COLUMNS contract)
        |
        v
Gradient-boosted fraud classifier
        |
        +--> risk probability
        |
        +--> validation-selected business-cost threshold
                    |
                    +--> approve (< review threshold)
                    +--> review  (between thresholds)
                    +--> decline (>= decision threshold)
        |
        v
Reason codes + audit-friendly JSON response
```

Training uses stratified train/validation/test splits. The validation split
selects the operating threshold using explicit false-positive and
false-negative costs. The untouched test split is used for final precision,
recall, F1, confusion counts, and expected cost.

The system is defense-only. It does not attempt payments, create accounts, or
automate an irreversible action; it returns a bounded recommendation for a
merchant's checkout workflow.
