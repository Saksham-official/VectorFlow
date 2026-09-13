"""Application adapter for the forecasting model."""

from datetime import timedelta
from pathlib import Path
from typing import Any

from contracts import ForecastPoint, ForecastResult, NetworkState

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pipeline(artifacts_dir: str | None = None):
    from model.src.inference import get_inference_pipeline

    path = Path(artifacts_dir or REPO_ROOT / "model" / "artifacts")
    return get_inference_pipeline(str(path))


def forecast(states: list[NetworkState], config: dict[str, Any] | None = None) -> ForecastResult:
    """Run inference on the most recent real historical sequence."""
    if not states:
        raise ValueError("forecast requires at least one network state")
    cfg = config or {}
    forecast_cfg = cfg.get("forecasting", cfg)
    pipeline = _pipeline(forecast_cfg.get("artifacts_dir"))
    try:
        result = pipeline.predict_sequence([state.features for state in states])
    except ValueError as exc:
        from model.src.inference import InsufficientHistoryError

        if not isinstance(exc, InsufficientHistoryError):
            raise
        return ForecastResult(
            infiltration_probability=None,
            horizon=[],
            model_name="ForecastModel",
            forecast_ready=False,
            sequence_length=pipeline.sequence_length,
            model_mode="real",
            status_message=str(exc),
        )

    probability = float(result["attack_probability"])
    current = states[-1]
    step = current.window_end - current.window_start
    if step.total_seconds() <= 0:
        step = timedelta(seconds=10)
    # The trained target is one 30–60 second forecast. A single point at the
    # middle of that band avoids inventing an autoregressive future path.
    point = ForecastPoint(
        step=max(1, round(45 / step.total_seconds())),
        time=current.window_end + timedelta(seconds=45),
        probability=probability,
    )
    return ForecastResult(
        infiltration_probability=probability,
        horizon=[point],
        model_name="ForecastModel",
        prediction=bool(result["prediction"]),
        threshold=float(result["threshold"]),
        forecast_horizon_seconds=result["forecast_horizon_seconds"],
        sequence_length=int(result["sequence_length"]),
        forecast_ready=True,
        model_mode="real",
        model_version="model-artifacts",
    )


__all__ = ["forecast"]
