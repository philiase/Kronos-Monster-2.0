import pandas as pd

from model import SignalConfig, analyze_forecast, analyze_market_structure


def make_structure_candles():
    closes = [
        100, 102, 101, 103, 101.2, 104, 102, 105, 103, 104.8,
        101.1, 103.7, 100.9, 104.2, 101.0, 103.8, 101.2, 104.1,
    ]
    rows = []
    for idx, close in enumerate(closes):
        rows.append(
            {
                "timestamps": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=30 * idx),
                "open": close - 0.4,
                "high": close + (1.0 if idx in {7, 9, 13, 17} else 0.5),
                "low": close - (1.0 if idx in {10, 12, 14, 16} else 0.5),
                "close": close,
                "volume": 1000 + (idx * 20) + (1200 if idx in {7, 14} else 0),
            }
        )
    return pd.DataFrame(rows)


def test_market_structure_detects_levels_and_volume():
    structure = analyze_market_structure(make_structure_candles()).to_dict()

    assert structure["support_levels"]
    assert structure["resistance_levels"]
    assert structure["volume_state"] in {"normal", "expansion", "dry"}
    assert structure["atr"] > 0


def test_structure_context_can_block_trade_into_resistance():
    history = make_structure_candles()
    last_close = history["close"].iloc[-1]
    forecast = pd.DataFrame(
        {
            "open": [last_close, last_close * 1.005, last_close * 1.01],
            "high": [last_close * 1.008, last_close * 1.012, last_close * 1.018],
            "low": [last_close * 0.998, last_close * 1.002, last_close * 1.008],
            "close": [last_close * 1.006, last_close * 1.012, last_close * 1.018],
            "volume": [1000, 1100, 1200],
        }
    )
    structure = analyze_market_structure(history).to_dict()
    signal = analyze_forecast(
        history,
        forecast,
        SignalConfig(min_edge=0.003, min_confidence=0.5),
        market_structure=structure,
    )

    assert signal.nearest_resistance is not None
    assert signal.context_score <= 0
