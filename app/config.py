# config.py

# List of features for the model
# The order and names must be consistent across training and prediction.
FEATURE_COLS = [
    'price_change_1d',
    'price_change_5d',
    'sma_10d',
    'sma_50d',
    'rsi_14d',
    'volatility_10d',
    'volume_change_1d'
]
