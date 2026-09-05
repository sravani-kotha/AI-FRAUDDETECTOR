"""
Single source of truth for feature columns and business cost constants.

train.py, evaluate.py, and api.py all import from here so the feature
list used to train can never silently drift from the feature list used
to serve — the most common way real fraud pipelines break in production.
"""

FEATURE_COLUMNS = [
    "order_value_ratio",        # this order's value vs. this customer's historical average
    "account_age_days",         # how long the account has existed
    "txns_last_hour",           # velocity: transactions from this card/device/IP in the last hour
    "billing_shipping_mismatch",  # 1 if billing and shipping address don't match
    "vpn_detected",              # 1 if a VPN/proxy was detected on this session
    "new_device",                 # 1 if this device hasn't been seen on this account before
    "hour_of_day",                # 0-23, local to the merchant
]

LABEL_COLUMN = "is_fraud"

# Business cost constants for the false-positive / false-negative
# trade-off (see evaluate.py). These are placeholders — replace with
# your merchant's real average order value and average confirmed-fraud
# loss (chargeback amount + fee + goods) before trusting the threshold
# it picks.
AVG_ORDER_VALUE = 85.0        # cost of a false positive: a good customer wrongly declined
AVG_FRAUD_LOSS = 210.0        # cost of a false negative: goods lost + chargeback fee
