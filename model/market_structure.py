from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StructureConfig:
    swing_window: int = 3
    zone_tolerance_atr: float = 0.35
    max_levels: int = 4
    volume_window: int = 40
    volume_spike_ratio: float = 1.6
    liquidity_tolerance_atr: float = 0.25


@dataclass(frozen=True)
class PriceLevel:
    kind: str
    price: float
    lower: float
    upper: float
    touches: int
    strength: float
    last_timestamp: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiquidityLevel:
    kind: str
    price: float
    lower: float
    upper: float
    touches: int
    swept: bool
    last_timestamp: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketStructure:
    trend: str
    regime: str
    atr: float
    atr_pct: float
    volume_state: str
    volume_ratio: float | None
    support_levels: list[PriceLevel]
    resistance_levels: list[PriceLevel]
    liquidity_levels: list[LiquidityLevel]
    volume_spikes: list[dict[str, Any]]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["support_levels"] = [level.to_dict() for level in self.support_levels]
        data["resistance_levels"] = [level.to_dict() for level in self.resistance_levels]
        data["liquidity_levels"] = [level.to_dict() for level in self.liquidity_levels]
        return data


PRICE_COLUMNS = ("open", "high", "low", "close")


def _clean_candles(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in PRICE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    candles = df.copy()
    for col in PRICE_COLUMNS:
        candles[col] = pd.to_numeric(candles[col], errors="coerce")
    if "volume" in candles.columns:
        candles["volume"] = pd.to_numeric(candles["volume"], errors="coerce")
    if "timestamps" in candles.columns:
        candles["timestamps"] = pd.to_datetime(candles["timestamps"], errors="coerce")
    candles = candles.dropna(subset=list(PRICE_COLUMNS)).reset_index(drop=True)
    if len(candles) < 10:
        raise ValueError("Need at least 10 candles for market structure analysis.")
    return candles


def _average_true_range(df: pd.DataFrame, window: int = 20) -> float:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.tail(window).mean()
    return float(atr) if np.isfinite(atr) else 0.0


def _timestamp_at(df: pd.DataFrame, idx: int) -> str | None:
    if "timestamps" not in df.columns or idx < 0 or idx >= len(df):
        return None
    value = df["timestamps"].iloc[idx]
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _find_swings(df: pd.DataFrame, window: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    swing_highs: list[dict[str, Any]] = []
    swing_lows: list[dict[str, Any]] = []
    if len(df) < window * 2 + 1:
        return swing_highs, swing_lows

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    for idx in range(window, len(df) - window):
        high_slice = highs[idx - window : idx + window + 1]
        low_slice = lows[idx - window : idx + window + 1]
        if highs[idx] == np.max(high_slice) and np.count_nonzero(high_slice == highs[idx]) == 1:
            swing_highs.append({"idx": idx, "price": float(highs[idx]), "timestamp": _timestamp_at(df, idx)})
        if lows[idx] == np.min(low_slice) and np.count_nonzero(low_slice == lows[idx]) == 1:
            swing_lows.append({"idx": idx, "price": float(lows[idx]), "timestamp": _timestamp_at(df, idx)})
    return swing_highs, swing_lows


def _cluster_levels(
    swings: list[dict[str, Any]],
    kind: str,
    tolerance: float,
    max_levels: int,
    current_price: float,
) -> list[PriceLevel]:
    clusters: list[dict[str, Any]] = []
    tolerance = max(tolerance, abs(current_price) * 0.0005, 1e-9)

    for swing in sorted(swings, key=lambda item: item["idx"]):
        price = swing["price"]
        match = None
        for cluster in clusters:
            if abs(price - cluster["price"]) <= tolerance:
                match = cluster
                break
        if match is None:
            clusters.append(
                {
                    "prices": [price],
                    "price": price,
                    "touches": 1,
                    "last_idx": swing["idx"],
                    "last_timestamp": swing.get("timestamp"),
                }
            )
        else:
            match["prices"].append(price)
            match["price"] = float(np.mean(match["prices"]))
            match["touches"] += 1
            match["last_idx"] = swing["idx"]
            match["last_timestamp"] = swing.get("timestamp")

    ranked = sorted(
        clusters,
        key=lambda item: (item["touches"], item["last_idx"], -abs(item["price"] - current_price)),
        reverse=True,
    )
    levels: list[PriceLevel] = []
    for cluster in ranked[:max_levels]:
        strength = min(1.0, 0.25 + cluster["touches"] * 0.18 + cluster["last_idx"] / max(len(swings), 1) * 0.02)
        levels.append(
            PriceLevel(
                kind=kind,
                price=round(float(cluster["price"]), 8),
                lower=round(float(cluster["price"] - tolerance), 8),
                upper=round(float(cluster["price"] + tolerance), 8),
                touches=int(cluster["touches"]),
                strength=round(float(strength), 4),
                last_timestamp=cluster["last_timestamp"],
            )
        )
    return sorted(levels, key=lambda level: level.price)


def _detect_trend(df: pd.DataFrame, swing_highs: list[dict[str, Any]], swing_lows: list[dict[str, Any]], atr: float) -> str:
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_highs = swing_highs[-2:]
        last_lows = swing_lows[-2:]
        high_delta = last_highs[-1]["price"] - last_highs[-2]["price"]
        low_delta = last_lows[-1]["price"] - last_lows[-2]["price"]
        threshold = max(atr * 0.2, abs(float(df["close"].iloc[-1])) * 0.0005)
        if high_delta > threshold and low_delta > threshold:
            return "bullish structure"
        if high_delta < -threshold and low_delta < -threshold:
            return "bearish structure"

    close = df["close"].astype(float)
    slope = float(close.tail(min(50, len(close))).iloc[-1] / close.tail(min(50, len(close))).iloc[0] - 1.0)
    threshold_pct = max((atr / close.iloc[-1]) * 1.5 if close.iloc[-1] else 0.0, 0.002)
    if slope > threshold_pct:
        return "bullish drift"
    if slope < -threshold_pct:
        return "bearish drift"
    return "sideways structure"


def _detect_regime(df: pd.DataFrame, atr: float) -> str:
    recent = df.tail(min(80, len(df)))
    close = recent["close"].astype(float)
    price_range = float(recent["high"].max() - recent["low"].min())
    close_move = abs(float(close.iloc[-1] - close.iloc[0]))
    if atr <= 0:
        return "unknown"
    if close_move < atr * 2.0 and price_range < atr * 6.0:
        return "range"
    if close_move > atr * 4.0:
        return "trend"
    return "mixed"


def _volume_context(df: pd.DataFrame, config: StructureConfig) -> tuple[str, float | None, list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if "volume" not in df.columns:
        warnings.append("No volume column found, volume confirmation is disabled.")
        return "missing", None, [], warnings

    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    if float(volume.abs().sum()) <= 0:
        warnings.append("Volume values are all zero, volume confirmation is disabled.")
        return "empty", None, [], warnings

    rolling = volume.rolling(config.volume_window, min_periods=max(5, config.volume_window // 4)).mean()
    baseline = float(rolling.iloc[-2]) if len(rolling) > 1 and np.isfinite(rolling.iloc[-2]) else float(volume.tail(config.volume_window).mean())
    latest = float(volume.iloc[-1])
    ratio = latest / baseline if baseline > 0 else None
    if ratio is None:
        state = "unknown"
    elif ratio >= config.volume_spike_ratio:
        state = "expansion"
    elif ratio <= 0.65:
        state = "dry"
    else:
        state = "normal"

    spike_rows = df.loc[volume >= rolling.fillna(np.inf) * config.volume_spike_ratio].tail(10)
    spikes: list[dict[str, Any]] = []
    for idx, row in spike_rows.iterrows():
        spikes.append(
            {
                "timestamp": _timestamp_at(df, int(idx)),
                "price": round(float(row["close"]), 8),
                "volume": round(float(row["volume"]), 4),
            }
        )
    return state, round(float(ratio), 4) if ratio is not None else None, spikes, warnings


def _liquidity_levels(
    swing_highs: list[dict[str, Any]],
    swing_lows: list[dict[str, Any]],
    df: pd.DataFrame,
    tolerance: float,
) -> list[LiquidityLevel]:
    levels: list[LiquidityLevel] = []
    latest_high = float(df["high"].iloc[-1])
    latest_low = float(df["low"].iloc[-1])
    tolerance = max(tolerance, abs(float(df["close"].iloc[-1])) * 0.0005, 1e-9)

    for swings, kind, latest_extreme in [
        (swing_highs, "buy-side liquidity", latest_high),
        (swing_lows, "sell-side liquidity", latest_low),
    ]:
        sorted_swings = sorted(swings[-20:], key=lambda item: item["price"])
        used: set[int] = set()
        for idx, swing in enumerate(sorted_swings):
            if idx in used:
                continue
            group = [swing]
            for other_idx in range(idx + 1, len(sorted_swings)):
                if other_idx in used:
                    continue
                other = sorted_swings[other_idx]
                if abs(other["price"] - swing["price"]) <= tolerance:
                    group.append(other)
                    used.add(other_idx)
            if len(group) < 2:
                continue
            price = float(np.mean([item["price"] for item in group]))
            swept = latest_extreme > price + tolerance if kind.startswith("buy") else latest_extreme < price - tolerance
            levels.append(
                LiquidityLevel(
                    kind=kind,
                    price=round(price, 8),
                    lower=round(price - tolerance, 8),
                    upper=round(price + tolerance, 8),
                    touches=len(group),
                    swept=bool(swept),
                    last_timestamp=max(item.get("timestamp") or "" for item in group) or None,
                )
            )

    return sorted(levels, key=lambda level: (level.swept, level.touches), reverse=True)[:6]


def analyze_market_structure(df: pd.DataFrame, config: StructureConfig | None = None) -> MarketStructure:
    config = config or StructureConfig()
    candles = _clean_candles(df)
    atr = _average_true_range(candles)
    current_price = float(candles["close"].iloc[-1])
    atr_pct = atr / current_price if current_price else 0.0
    swing_highs, swing_lows = _find_swings(candles, config.swing_window)
    tolerance = max(atr * config.zone_tolerance_atr, current_price * 0.0006)

    support_swings = [swing for swing in swing_lows if swing["price"] <= current_price + tolerance]
    resistance_swings = [swing for swing in swing_highs if swing["price"] >= current_price - tolerance]
    support = _cluster_levels(support_swings, "support", tolerance, config.max_levels, current_price)
    resistance = _cluster_levels(resistance_swings, "resistance", tolerance, config.max_levels, current_price)
    liquidity = _liquidity_levels(swing_highs, swing_lows, candles, max(atr * config.liquidity_tolerance_atr, current_price * 0.0005))
    volume_state, volume_ratio, volume_spikes, warnings = _volume_context(candles, config)

    return MarketStructure(
        trend=_detect_trend(candles, swing_highs, swing_lows, atr),
        regime=_detect_regime(candles, atr),
        atr=round(float(atr), 8),
        atr_pct=round(float(atr_pct), 6),
        volume_state=volume_state,
        volume_ratio=volume_ratio,
        support_levels=support,
        resistance_levels=resistance,
        liquidity_levels=liquidity,
        volume_spikes=volume_spikes,
        warnings=warnings,
    )
