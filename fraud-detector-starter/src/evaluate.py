"""
Produces the evaluation artifacts the project's success bar asks for:
a precision/recall curve plot and a false-positive-cost readout.

Reads reports/metrics.json (written by train.py) so this can be re-run
without retraining.

Usage:
    python src/evaluate.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from .features import AVG_ORDER_VALUE, AVG_FRAUD_LOSS
except ImportError:  # pragma: no cover
    from features import AVG_ORDER_VALUE, AVG_FRAUD_LOSS

DEFAULT_REPORTS_DIR = Path(__file__).parent.parent / "reports"


def main(reports_dir: str = None):
    reports_dir = reports_dir or DEFAULT_REPORTS_DIR
    metrics_path = Path(reports_dir) / "metrics.json"
    with open(metrics_path) as f:
        m = json.load(f)

    curve = m["precision_recall_curve"]
    recall = [pt["recall"] for pt in curve]
    precision = [pt["precision"] for pt in curve]

    chosen = m["boosted_model"]["at_cost_minimizing_threshold"]
    default = m["boosted_model"]["at_threshold_0.5"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(recall, precision, color="#1D9E75", linewidth=2)
    ax.scatter([chosen["recall"]], [chosen["precision"]], color="#BA7517", zorder=5,
               label=f"validation-selected threshold ({chosen['threshold']})")
    ax.scatter([default["recall"]], [default["precision"]], color="#888780", zorder=5,
               label="default threshold (0.5)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    labels = ["False positives\n(good customers declined)", "False negatives\n(missed fraud)"]
    chosen_counts = [chosen["false_positives"], chosen["false_negatives"]]
    default_counts = [default["false_positives"], default["false_negatives"]]
    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], default_counts, width, label="threshold 0.5", color="#888780")
    ax.bar([i + width / 2 for i in x], chosen_counts, width, label="cost-minimizing threshold", color="#BA7517")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Count on test set")
    ax.set_title("False-positive / false-negative trade-off")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Fraud detector — evaluation on held-out test set", fontsize=12)
    fig.tight_layout()
    out_path = Path(reports_dir) / "evaluation.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")

    print(f"\nAssumed costs: false positive = ${AVG_ORDER_VALUE:.0f} (avg order), "
          f"false negative = ${AVG_FRAUD_LOSS:.0f} (avg fraud loss) — edit features.py to your real numbers.")
    print(f"\nDefault threshold (0.5):          precision {default['precision']:.1%}  "
          f"recall {default['recall']:.1%}  expected cost ${default['expected_cost']:,.0f}")
    print(f"Validation-selected threshold ({chosen['threshold']}), test result: precision {chosen['precision']:.1%}  "
          f"recall {chosen['recall']:.1%}  expected cost ${chosen['expected_cost']:,.0f}")


if __name__ == "__main__":
    main()
