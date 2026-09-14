"""
VectorFlow BiLSTM Forecasting Model
====================================
Bidirectional LSTM with Attention Pooling for proactive attack-onset prediction.
Predicts attack probability 30–120 seconds ahead from 10-window flow sequences.
"""

from .inference import InsufficientHistoryError, InferencePipeline, get_inference_pipeline
from .model import ForecastLSTM

__all__ = [
    "ForecastLSTM",
    "InferencePipeline",
    "InsufficientHistoryError",
    "get_inference_pipeline",
]
