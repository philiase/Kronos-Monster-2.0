from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from .market_structure import MarketStructure


@dataclass(frozen=True)
class SignalConfig:
    min_edge: float = 0.006
    volatility_window: int = 20
    trend_window: int = 30
    min_confidence: float = 0.55
    risk_per_trade: float = 0.01
    stop_atr_multiple: float = 1.5
    target_r_multiple: float = 2.0
    max_position_fraction: float = 0.25
    chop_volatility_ratio: float = 0.65


@dataclass(frozen=True)
class TradeSignal:
    action: str
    trend: str
    regime: str
    confidence: float
    entry: float
    stop_loss: float | None
    take_profit: float | None
    position_fraction: float
    forecast_return: float
    expected_r_multiple: float
    volatility: float
    max_forecast_drawdown: float
    reason: str
    structure_trend: str | None = None
    market_regime: str | None = None
    volume_state: str | None = None
    volume_ratio: float | None = None
    nearest_support: float | None = None
    nearest_resistance: float | None = None
    context_score: float = 0.0
    context_warnings: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_PRICE_COLUMNS = ("open", "high", "low", "close")


def _validate_candles(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame.")
    missing = [col for col in REQUIRED_PRICE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

    candles = df.loc[:, list(REQUIRED_PRICE_COLUMNS)].copy()
    for col in REQUIRED_PRICE_COLUMNS:
        candles[col] = pd.to_numeric(candles[col], errors="coerce")
    if candles.isnull().values.any():
        raise ValueError(f"{name} contains NaN or non-numeric OHLC values.")
    if len(candles) < 2:
        raise ValueError(f"{name} must contain at least two candles.")
    return candles


def _average_true_range(df: pd.DataFrame, window: int) -> float:
    previous_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    true_range = ranges.max(axis=1).dropna()
    if true_range.empty:
        return 0.0
    return float(true_range.tail(window).mean())


def _max_drawdown(path: pd.Series) -> float:
    running_max = path.cummax()
    drawdown = (path / running_max) - 1.0
    return float(drawdown.min())


def _trend_label(value: float, threshold: float) -> str:
    if value > threshold:
        return "bullish"
    if value < -threshold:
        return "bearish"
    return "sideways"


def _structure_to_dict(market_structure: MarketStructure | dict[str, Any] | None) -> dict[str, Any] | None:
    if market_structure is None:
        return None
    if isinstance(market_structure, MarketStructure):
        return market_structure.to_dict()
    return market_structure


def _nearest_level(levels: list[dict[str, Any]], entry: float, side: str) -> dict[str, Any] | None:
    candidates = []
    for level in levels:
        price = float(level["price"])
        if side == "above" and price >= entry:
            candidates.append(level)
        elif side == "below" and price <= entry:
            candidates.append(level)
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(float(item["price"]) - entry))


def _market_context_adjustment(
    forecast_direction: str,
    entry: float,
    noise_floor: float,
    market_structure: MarketStructure | dict[str, Any] | None,
) -> tuple[float, list[str], dict[str, Any]]:
    structure = _structure_to_dict(market_structure)
    if not structure:
        return 0.0, [], {
            "structure_trend": None,
            "market_regime": None,
            "volume_state": None,
            "volume_ratio": None,
            "nearest_support": None,
            "nearest_resistance": None,
            "context_warnings": None,
        }

    reasons: list[str] = []
    context_score = 0.0
    atr_pct = float(structure.get("atr_pct") or 0.0)
    danger_band = max(noise_floor * 1.5, atr_pct * 1.5, 0.001)
    support_levels = structure.get("support_levels") or []
    resistance_levels = structure.get("resistance_levels") or []
    nearest_support = _nearest_level(support_levels, entry, "below")
    nearest_resistance = _nearest_level(resistance_levels, entry, "above")

    structure_trend = structure.get("trend")
    if forecast_direction == "bullish":
        if structure_trend and "bullish" in structure_trend:
            context_score += 0.08
            reasons.append("Market structure agrees with bullish forecast.")
        elif structure_trend and "bearish" in structure_trend:
            context_score -= 0.12
            reasons.append("Bullish forecast is fighting bearish structure.")
        if nearest_resistance:
            distance = (float(nearest_resistance["price"]) / entry) - 1.0
            if 0 <= distance <= danger_band:
                context_score -= 0.18
                reasons.append("Forecast is pushing into nearby resistance.")
    elif forecast_direction == "bearish":
        if structure_trend and "bearish" in structure_trend:
            context_score += 0.08
            reasons.append("Market structure agrees with bearish forecast.")
        elif structure_trend and "bullish" in structure_trend:
            context_score -= 0.12
            reasons.append("Bearish forecast is fighting bullish structure.")
        if nearest_support:
            distance = 1.0 - (float(nearest_support["price"]) / entry)
            if 0 <= distance <= danger_band:
                context_score -= 0.18
                reasons.append("Forecast is pushing into nearby support.")

    regime = structure.get("regime")
    if regime == "range":
        context_score -= 0.08
        reasons.append("Recent candles are range-bound, so directional edge is weaker.")
    elif regime == "trend":
        context_score += 0.04

    volume_state = structure.get("volume_state")
    if volume_state == "expansion":
        context_score += 0.05
        reasons.append("Recent volume expanded, adding confirmation.")
    elif volume_state == "dry" and forecast_direction != "sideways":
        context_score -= 0.07
        reasons.append("Forecast direction has weak volume participation.")

    context_score = max(-0.35, min(0.2, context_score))
    context = {
        "structure_trend": structure_trend,
        "market_regime": regime,
        "volume_state": volume_state,
        "volume_ratio": structure.get("volume_ratio"),
        "nearest_support": float(nearest_support["price"]) if nearest_support else None,
        "nearest_resistance": float(nearest_resistance["price"]) if nearest_resistance else None,
        "context_warnings": structure.get("warnings") or None,
    }
    return context_score, reasons, context


def analyze_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    config: SignalConfig | None = None,
    market_structure: MarketStructure | dict[str, Any] | None = None,
) -> TradeSignal:
    """Convert predicted candles into a risk-aware trading signal.

    The signal is intentionally conservative. A forecast must beat recent
    volatility, avoid a choppy regime, and keep drawdown under control before
    it becomes a long/short idea.
    """

    config = config or SignalConfig()
    history = _validate_candles(historical_df, "historical_df")
    forecast = _validate_candles(forecast_df, "forecast_df")

    last_close = float(history["close"].iloc[-1])
    final_close = float(forecast["close"].iloc[-1])
    forecast_return = (final_close / last_close) - 1.0

    combined = pd.concat([history, forecast], axis=0, ignore_index=True)
    returns = history["close"].pct_change().dropna()
    recent_volatility = float(returns.tail(config.volatility_window).std())
    if not np.isfinite(recent_volatility):
        recent_volatility = 0.0

    atr = _average_true_range(history, config.volatility_window)
    atr_pct = atr / last_close if last_close else 0.0
    noise_floor = max(config.min_edge, atr_pct, recent_volatility)

    trend_start = float(history["close"].tail(config.trend_window).iloc[0])
    historical_trend = (last_close / trend_start) - 1.0 if trend_start else 0.0
    forecast_path = pd.concat(
        [pd.Series([last_close]), forecast["close"].reset_index(drop=True)],
        ignore_index=True,
    )
    max_forecast_drawdown = _max_drawdown(forecast_path)

    forecast_direction = _trend_label(forecast_return, noise_floor)
    historical_direction = _trend_label(historical_trend, noise_floor)

    path_returns = forecast_path.pct_change().dropna()
    positive_ratio = float((path_returns > 0).mean()) if len(path_returns) else 0.5
    negative_ratio = float((path_returns < 0).mean()) if len(path_returns) else 0.5
    direction_consistency = positive_ratio if forecast_return >= 0 else negative_ratio

    momentum_agreement = 1.0 if forecast_direction == historical_direction else 0.65
    edge_score = min(abs(forecast_return) / (noise_floor * 3.0), 1.0) if noise_floor > 0 else 0.0
    drawdown_penalty = min(abs(max_forecast_drawdown) / max(noise_floor * 3.0, 1e-9), 1.0)
    confidence = 0.25 + 0.45 * edge_score + 0.2 * direction_consistency + 0.1 * momentum_agreement
    confidence = max(0.0, min(confidence - 0.2 * drawdown_penalty, 1.0))

    context_score, context_reasons, context = _market_context_adjustment(
        forecast_direction,
        last_close,
        noise_floor,
        market_structure,
    )
    confidence = max(0.0, min(confidence + context_score, 1.0))

    forecast_range = float((forecast["high"].max() - forecast["low"].min()) / last_close)
    is_choppy = (
        abs(forecast_return) < noise_floor
        or direction_consistency < config.chop_volatility_ratio
        or forecast_range > max(abs(forecast_return) * 5.0, noise_floor * 4.0)
    )
    regime = "choppy" if is_choppy else "trending"

    action = "hold"
    reason = "Forecast edge is not strong enough after volatility and regime filters."
    if regime == "trending" and confidence >= config.min_confidence:
        if forecast_return > noise_floor:
            action = "long"
            reason = "Forecast shows upside edge with acceptable trend consistency."
        elif forecast_return < -noise_floor:
            action = "short"
            reason = "Forecast shows downside edge with acceptable trend consistency."

    if context_score <= -0.18 and action != "hold":
        action = "hold"
        reason = "Market structure filter blocked the trade despite the forecast edge."
    if context_reasons:
        reason = f"{reason} {' '.join(context_reasons)}"

    stop_distance = max(atr * config.stop_atr_multiple, last_close * noise_floor)
    if action == "long":
        stop_loss = last_close - stop_distance
        take_profit = last_close + stop_distance * config.target_r_multiple
        risk_per_unit = last_close - stop_loss
    elif action == "short":
        stop_loss = last_close + stop_distance
        take_profit = last_close - stop_distance * config.target_r_multiple
        risk_per_unit = stop_loss - last_close
    else:
        stop_loss = None
        take_profit = None
        risk_per_unit = 0.0

    position_fraction = 0.0
    if action != "hold" and risk_per_unit > 0:
        raw_fraction = config.risk_per_trade / (risk_per_unit / last_close)
        position_fraction = min(raw_fraction * confidence, config.max_position_fraction)
        position_fraction = max(0.0, float(position_fraction))

    expected_r_multiple = 0.0
    if action == "long" and risk_per_unit > 0:
        expected_r_multiple = (final_close - last_close) / risk_per_unit
    elif action == "short" and risk_per_unit > 0:
        expected_r_multiple = (last_close - final_close) / risk_per_unit

    return TradeSignal(
        action=action,
        trend=forecast_direction,
        regime=regime,
        confidence=round(float(confidence), 4),
        entry=round(last_close, 8),
        stop_loss=round(float(stop_loss), 8) if stop_loss is not None else None,
        take_profit=round(float(take_profit), 8) if take_profit is not None else None,
        position_fraction=round(position_fraction, 4),
        forecast_return=round(float(forecast_return), 6),
        expected_r_multiple=round(float(expected_r_multiple), 4),
        volatility=round(float(recent_volatility), 6),
        max_forecast_drawdown=round(float(max_forecast_drawdown), 6),
        reason=reason,
        structure_trend=context["structure_trend"],
        market_regime=context["market_regime"],
        volume_state=context["volume_state"],
        volume_ratio=context["volume_ratio"],
        nearest_support=round(float(context["nearest_support"]), 8) if context["nearest_support"] is not None else None,
        nearest_resistance=round(float(context["nearest_resistance"]), 8) if context["nearest_resistance"] is not None else None,
        context_score=round(float(context_score), 4),
        context_warnings=context["context_warnings"],
    )


def summarize_signal(signal: TradeSignal) -> str:
    action = signal.action.upper()
    confidence = f"{signal.confidence:.0%}"
    forecast_pct = f"{signal.forecast_return:.2%}"
    context_bits = []
    if signal.structure_trend:
        context_bits.append(signal.structure_trend)
    if signal.volume_state and signal.volume_state not in {"missing", "empty"}:
        context_bits.append(f"volume {signal.volume_state}")
    context_text = f" Context: {', '.join(context_bits)}." if context_bits else ""
    if signal.action == "hold":
        return (
            f"{action}: {signal.trend} forecast, {signal.regime} regime, "
            f"{confidence} confidence, expected move {forecast_pct}.{context_text} {signal.reason}"
        )
    return (
        f"{action}: {signal.trend} forecast, {signal.regime} regime, "
        f"{confidence} confidence, expected move {forecast_pct}, "
        f"size up to {signal.position_fraction:.1%}, stop {signal.stop_loss}, "
        f"target {signal.take_profit}.{context_text} {signal.reason}"
    )
