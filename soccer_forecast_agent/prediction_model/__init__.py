"""Prediction model tools exposed by the soccer forecast agent."""

from .model import predict_exact_score, predict_match_outcome

__all__ = ["predict_exact_score", "predict_match_outcome"]
