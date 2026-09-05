"""
Generates a synthetic transaction dataset shaped like real checkout data,
so the rest of the pipeline (train / evaluate / serve) is runnable end to
end before you have real labeled fraud data.

Swap this out for a real extract as soon as you have one — the column
names below (`FEATURE_COLUMNS` in features.py) are what the rest of the
pipeline expects, so keep those consistent when you plug in real data.

Usage:
    python src/generate_data.py --n 20000 --fraud-rate 0.015 --out data/transactions.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent


def generate(n: int, fraud_rate: float, seed: int = 7) -> pd.DataFrame:
    if n < 10:
        raise ValueError("n must be at least 10")
    if not 0 < fraud_rate < 1:
        raise ValueError("fraud_rate must be between 0 and 1")
    if int(n * fraud_rate) < 2 or n - int(n * fraud_rate) < 2:
        raise ValueError("n and fraud_rate must produce at least two rows per class")
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_good = n - n_fraud

    def block(size: int, is_fraud: int) -> pd.DataFrame:
        if is_fraud:
            # fraud skews toward higher order-value ratio, newer accounts,
            # more velocity, and more flags — but with real overlap/noise
            # so the classes aren't trivially separable.
            order_value_ratio = rng.lognormal(mean=1.1, sigma=0.7, size=size)
            account_age_days = rng.exponential(scale=25, size=size).clip(0, 900)
            txns_last_hour = rng.poisson(lam=2.4, size=size) + 1
            mismatch = rng.binomial(1, 0.45, size=size)
            vpn = rng.binomial(1, 0.35, size=size)
            new_device = rng.binomial(1, 0.55, size=size)
        else:
            order_value_ratio = rng.lognormal(mean=0.0, sigma=0.35, size=size)
            account_age_days = rng.exponential(scale=220, size=size).clip(0, 2500)
            txns_last_hour = rng.poisson(lam=1.05, size=size) + 1
            mismatch = rng.binomial(1, 0.04, size=size)
            vpn = rng.binomial(1, 0.03, size=size)
            new_device = rng.binomial(1, 0.12, size=size)

        hour_of_day = rng.integers(0, 24, size=size)
        return pd.DataFrame({
            "order_value_ratio": order_value_ratio,
            "account_age_days": account_age_days,
            "txns_last_hour": txns_last_hour,
            "billing_shipping_mismatch": mismatch,
            "vpn_detected": vpn,
            "new_device": new_device,
            "hour_of_day": hour_of_day,
            "is_fraud": is_fraud,
        })

    df = pd.concat([block(n_good, 0), block(n_fraud, 1)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN-{100000+i}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--fraud-rate", type=float, default=0.015)
    ap.add_argument("--out", type=str, default=str(PROJECT_ROOT / "data" / "transactions.csv"))
    args = ap.parse_args()

    df = generate(args.n, args.fraud_rate)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows ({df['is_fraud'].sum()} fraud, "
          f"{df['is_fraud'].mean():.3%}) to {args.out}")
