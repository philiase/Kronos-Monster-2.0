import pandas as pd

from model import SignalConfig, analyze_forecast, summarize_signal


def make_candles(closes):
    rows = []
    for close in closes:
        rows.append(
            {
                "open": close * 0.995,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1000,
            }
        )
    return pd.DataFrame(rows)


def test_bullish_forecast_can_become_long_signal():
    history = make_candles([100 + i * 0.2 for i in range(60)])
    last = history["close"].iloc[-1]
    forecast = make_candles([last * (1 + i * 0.01) for i in range(1, 7)])

    signal = analyze_forecast(
        history,
        forecast,
        SignalConfig(min_edge=0.003, min_confidence=0.5),
    )

    assert signal.action == "long"
    assert signal.trend == "bullish"
    assert signal.regime == "trending"
    assert signal.stop_loss is not None
    assert signal.take_profit is not None
    assert signal.position_fraction > 0


def test_sideways_forecast_stays_hold():
    history = make_candles([100, 100.2, 99.9, 100.1, 100.0] * 12)
    forecast = make_candles([100.0, 100.1, 99.95, 100.05, 100.0])

    signal = analyze_forecast(history, forecast, SignalConfig(min_edge=0.01))

    assert signal.action == "hold"
    assert signal.regime == "choppy"
    assert "HOLD" in summarize_signal(signal)


def test_missing_columns_raise_clear_error():
    history = make_candles([100, 101, 102])
    forecast = pd.DataFrame({"close": [103, 104]})

    try:
        analyze_forecast(history, forecast)
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("Expected a clear validation error.")
