from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioConfig:
    min_score_to_select: float = 0.35
    max_open_gap_atr: float = 0.75
    pullback_depth_atr: float = 0.7
    breakout_buffer_atr: float = 0.35
    structure_weight: float = 0.24
    level_weight: float = 0.24
    liquidity_weight: float = 0.18
    realism_weight: float = 0.2
    volume_weight: float = 0.14


@dataclass(frozen=True)
class ScenarioScore:
    index: int
    score: float
    label: str
    direction: str
    forecast_return: float
    max_drawdown: float
    max_runup: float
    open_gap_atr: float
    structure_score: float
    level_score: float
    liquidity_score: float
    realism_score: float
    volume_score: float
    reasons: list[str]
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _close_series(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["close"], errors="coerce").dropna().reset_index(drop=True)


def _max_drawdown(path: pd.Series) -> float:
    running_max = path.cummax()
    return float(((path / running_max) - 1.0).min())


def _max_runup(path: pd.Series) -> float:
    running_min = path.cummin()
    return float(((path / running_min) - 1.0).max())


def _nearest(levels: list[dict[str, Any]], price: float, side: str) -> dict[str, Any] | None:
    candidates = []
    for level in levels:
        level_price = float(level["price"])
        if side == "above" and level_price >= price:
            candidates.append(level)
        elif side == "below" and level_price <= price:
            candidates.append(level)
    if not candidates:
        return None
    return min(candidates, key=lambda level: abs(float(level["price"]) - price))


def _label_direction(forecast_return: float, atr_pct: float) -> str:
    threshold = max(atr_pct * 0.8, 0.0015)
    if forecast_return > threshold:
        return "bullish"
    if forecast_return < -threshold:
        return "bearish"
    return "sideways"


def _is_rejection_candle(row: pd.Series, direction: str) -> bool:
    high = float(row["high"])
    low = float(row["low"])
    open_ = float(row["open"])
    close = float(row["close"])
    candle_range = max(high - low, 1e-9)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    if direction == "bullish":
        return lower_wick / candle_range >= 0.35 and close >= open_
    if direction == "bearish":
        return upper_wick / candle_range >= 0.35 and close <= open_
    return False


def _score_structure(direction: str, structure_trend: str | None, regime: str | None, reasons: list[str]) -> float:
    score = 0.0
    trend_text = structure_trend or ""
    if direction == "bullish":
        if "bullish" in trend_text:
            score += 0.7
            reasons.append("matches bullish structure")
        elif "bearish" in trend_text:
            score -= 0.5
            reasons.append("fights bearish structure")
    elif direction == "bearish":
        if "bearish" in trend_text:
            score += 0.7
            reasons.append("matches bearish structure")
        elif "bullish" in trend_text:
            score -= 0.5
            reasons.append("fights bullish structure")
    else:
        if regime == "range":
            score += 0.4
            reasons.append("sideways path fits range conditions")

    if regime == "trend" and direction != "sideways":
        score += 0.25
    elif regime == "range" and direction != "sideways":
        score -= 0.25
        reasons.append("directional path inside range regime")
    return max(-1.0, min(1.0, score))


def _score_levels(
    forecast: pd.DataFrame,
    direction: str,
    entry: float,
    atr: float,
    structure: dict[str, Any],
    reasons: list[str],
) -> tuple[float, str]:
    support = _nearest(structure.get("support_levels") or [], entry, "below")
    resistance = _nearest(structure.get("resistance_levels") or [], entry, "above")
    high = float(pd.to_numeric(forecast["high"], errors="coerce").max())
    low = float(pd.to_numeric(forecast["low"], errors="coerce").min())
    close = float(forecast["close"].iloc[-1])
    buffer = max(atr * 0.35, abs(entry) * 0.0007)
    score = 0.0
    label = "balanced"

    if direction == "bullish":
        if resistance:
            r_price = float(resistance["price"])
            if high > r_price + buffer and close > r_price:
                score += 0.35
                label = "breakout continuation"
                reasons.append("breaks and holds above resistance")
            elif high > r_price and close <= r_price:
                score += 0.15
                label = "liquidity probe"
                reasons.append("probes resistance without clean acceptance")
            elif 0 <= (r_price - close) <= buffer:
                score -= 0.45
                reasons.append("ends directly under resistance")
        if support and low >= float(support["lower"]) - buffer:
            score += 0.15
            reasons.append("respects nearby support")
    elif direction == "bearish":
        if support:
            s_price = float(support["price"])
            if low < s_price - buffer and close < s_price:
                score += 0.35
                label = "breakdown continuation"
                reasons.append("breaks and holds below support")
            elif low < s_price and close >= s_price:
                score += 0.15
                label = "liquidity probe"
                reasons.append("probes support without clean acceptance")
            elif 0 <= (close - s_price) <= buffer:
                score -= 0.45
                reasons.append("ends directly above support")
        if resistance and high <= float(resistance["upper"]) + buffer:
            score += 0.15
            reasons.append("respects nearby resistance")
    else:
        if support and resistance:
            if low >= float(support["lower"]) - buffer and high <= float(resistance["upper"]) + buffer:
                score += 0.45
                label = "range rotation"
                reasons.append("stays between support and resistance")

    return max(-1.0, min(1.0, score)), label


def _score_liquidity(
    forecast: pd.DataFrame,
    direction: str,
    atr: float,
    structure: dict[str, Any],
    reasons: list[str],
) -> tuple[float, str | None]:
    high = float(pd.to_numeric(forecast["high"], errors="coerce").max())
    low = float(pd.to_numeric(forecast["low"], errors="coerce").min())
    final = forecast.iloc[-1]
    buffer = max(atr * 0.25, abs(float(final["close"])) * 0.0005)
    score = 0.0
    label = None

    for level in structure.get("liquidity_levels") or []:
        price = float(level["price"])
        if level["kind"].startswith("buy") and high > price + buffer:
            if direction == "bearish" or _is_rejection_candle(final, "bearish"):
                score += 0.45
                label = "buy-side sweep rejection"
                reasons.append("sweeps buy-side liquidity and rejects")
            else:
                score += 0.15
                reasons.append("takes buy-side liquidity")
        if level["kind"].startswith("sell") and low < price - buffer:
            if direction == "bullish" or _is_rejection_candle(final, "bullish"):
                score += 0.45
                label = "sell-side sweep rejection"
                reasons.append("sweeps sell-side liquidity and rejects")
            else:
                score += 0.15
                reasons.append("takes sell-side liquidity")
    return max(-1.0, min(1.0, score)), label


def _score_realism(history: pd.DataFrame, forecast: pd.DataFrame, atr: float, reasons: list[str]) -> tuple[float, float, float]:
    entry = float(history["close"].iloc[-1])
    first_open = float(forecast["open"].iloc[0])
    open_gap_atr = abs(first_open - entry) / max(atr, 1e-9)
    candle_ranges = pd.to_numeric(forecast["high"], errors="coerce") - pd.to_numeric(forecast["low"], errors="coerce")
    median_range_atr = float(candle_ranges.median() / max(atr, 1e-9))
    closes = _close_series(forecast)
    path = pd.concat([pd.Series([entry]), closes], ignore_index=True)
    step_returns = path.pct_change().dropna()
    direction_consistency = float(max((step_returns > 0).mean(), (step_returns < 0).mean())) if len(step_returns) else 0.5

    score = 0.55
    if open_gap_atr <= 0.35:
        score += 0.2
        reasons.append("opens close to last real close")
    elif open_gap_atr > 1.0:
        score -= 0.4
        reasons.append("large forecast open gap")
    if median_range_atr > 2.5:
        score -= 0.25
        reasons.append("forecast candles are unusually wide")
    elif 0.25 <= median_range_atr <= 1.8:
        score += 0.15
    if direction_consistency >= 0.68:
        score += 0.1
    return max(-1.0, min(1.0, score)), open_gap_atr, median_range_atr


def _score_volume(direction: str, structure: dict[str, Any], reasons: list[str]) -> float:
    state = structure.get("volume_state")
    if state == "expansion":
        reasons.append("volume expansion supports active scenario")
        return 0.45 if direction != "sideways" else 0.15
    if state == "dry":
        if direction == "sideways":
            reasons.append("dry volume fits pause scenario")
            return 0.25
        reasons.append("dry volume weakens directional scenario")
        return -0.25
    return 0.0


def score_forecast_scenario(
    index: int,
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    market_structure: dict[str, Any] | None,
    config: ScenarioConfig | None = None,
) -> ScenarioScore:
    config = config or ScenarioConfig()
    structure = market_structure or {}
    reasons: list[str] = []
    entry = float(history["close"].iloc[-1])
    final_close = float(forecast["close"].iloc[-1])
    forecast_return = (final_close / entry) - 1.0 if entry else 0.0
    atr = float(structure.get("atr") or 0.0)
    if atr <= 0:
        atr = float((pd.to_numeric(history["high"], errors="coerce") - pd.to_numeric(history["low"], errors="coerce")).tail(20).mean())
    atr = max(atr, abs(entry) * 0.001, 1e-9)
    atr_pct = atr / entry if entry else 0.0
    direction = _label_direction(forecast_return, atr_pct)

    structure_score = _score_structure(direction, structure.get("trend"), structure.get("regime"), reasons)
    level_score, level_label = _score_levels(forecast, direction, entry, atr, structure, reasons)
    liquidity_score, liquidity_label = _score_liquidity(forecast, direction, atr, structure, reasons)
    realism_score, open_gap_atr, _ = _score_realism(history, forecast, atr, reasons)
    volume_score = _score_volume(direction, structure, reasons)

    score = (
        structure_score * config.structure_weight
        + level_score * config.level_weight
        + liquidity_score * config.liquidity_weight
        + realism_score * config.realism_weight
        + volume_score * config.volume_weight
    )

    closes = _close_series(forecast)
    path = pd.concat([pd.Series([entry]), closes], ignore_index=True)
    label = liquidity_label or level_label
    if label == "balanced":
        label = f"{direction} scenario" if direction != "sideways" else "pause/range scenario"

    return ScenarioScore(
        index=index,
        score=round(float(score), 4),
        label=label,
        direction=direction,
        forecast_return=round(float(forecast_return), 6),
        max_drawdown=round(_max_drawdown(path), 6),
        max_runup=round(_max_runup(path), 6),
        open_gap_atr=round(float(open_gap_atr), 4),
        structure_score=round(float(structure_score), 4),
        level_score=round(float(level_score), 4),
        liquidity_score=round(float(liquidity_score), 4),
        realism_score=round(float(realism_score), 4),
        volume_score=round(float(volume_score), 4),
        reasons=reasons[:8],
    )


def select_best_scenario(
    history: pd.DataFrame,
    forecasts: list[pd.DataFrame],
    market_structure: dict[str, Any] | None,
    config: ScenarioConfig | None = None,
) -> tuple[int, list[ScenarioScore]]:
    if not forecasts:
        raise ValueError("At least one forecast scenario is required.")
    scores = [
        score_forecast_scenario(index, history, forecast, market_structure, config)
        for index, forecast in enumerate(forecasts)
    ]
    best_index = max(range(len(scores)), key=lambda idx: scores[idx].score)
    selected_scores = [
        ScenarioScore(**{**score.to_dict(), "selected": idx == best_index})
        for idx, score in enumerate(scores)
    ]
    return best_index, selected_scores
