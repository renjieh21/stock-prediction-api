# app/schemas.py

from pydantic import BaseModel, create_model
from typing import List
from app.config import FEATURE_COLS

# --- Dynamic Model Creation ---

# 1. Dynamically create an example payload for the API documentation.
example_features = {feature: round(0.01 * (i + 1), 2) for i, feature in enumerate(FEATURE_COLS)}

# 2. Dynamically create the StockFeatures Pydantic model.
# The __config__ argument expects a dictionary, not a class.
# We pass the schema_extra configuration directly as a dictionary.
StockFeatures = create_model(
    'StockFeatures',
    **{feature: (float, ...) for feature in FEATURE_COLS},
    __config__={'schema_extra': {'example': example_features}}
)

# --- Static Schemas (dependent on the dynamic model) ---
# The following schemas now depend on the dynamically created StockFeatures model.

class PredictionResponse(BaseModel):
    """Defines the structure for a single prediction response."""
    predicted_future_return_5d: float


class BatchPredictionRequest(BaseModel):
    """Defines the structure for a batch prediction request."""
    instances: List[StockFeatures]


class BatchPredictionResponse(BaseModel):
    """Defines the structure for a batch prediction response."""
    predictions: List[float]
