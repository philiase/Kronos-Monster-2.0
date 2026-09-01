import pandas as pd

from model import analyze_market_structure, select_best_scenario


def make_history():
    closes = [
        100, 101, 100.4, 102, 101.2, 103, 102.1, 104, 103.0, 105,
        104.0, 106, 105.1, 107, 106.0, 108, 107.2, 109, 108.5, 110,
        109.2, 111, 110.0, 112, 111.2, 113,
    ]
    rows = []
    for idx, close in enumerate(closes):
        rows.append(
            {
                "timestamps": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=30 * idx),
                "open": close - 0.35,
                "high": close + 0.8,
                "low": close - 0.8,
                "close": close,
                "volume": 1200 + idx * 15,
            }
        )
    return pd.DataFrame(rows)


def make_forecast(closes):
    rows = []
    for close in closes:
        rows.append(
            {
                "open": close - 0.25,
                "high": close + 0.6,
                "low": close - 0.6,
                "close": close,
                "volume": 1400,
            }
        )
    return pd.DataFrame(rows)


def test_selector_prefers_structure_aligned_path():
    history = make_history()
    structure = analyze_market_structure(history).to_dict()
    bearish = make_forecast([112.5, 111.8, 111.0, 110.4])
    bullish = make_forecast([113.2, 114.0, 114.8, 115.6])

    best_index, scores = select_best_scenario(history, [bearish, bullish], structure)

    assert best_index == 1
    assert scores[best_index].selected is True
    assert scores[best_index].score > scores[0].score


def test_selector_returns_single_best_index_for_one_path():
    history = make_history()
    structure = analyze_market_structure(history).to_dict()
    forecast = make_forecast([113.1, 113.4, 113.7])

    best_index, scores = select_best_scenario(history, [forecast], structure)

    assert best_index == 0
    assert len(scores) == 1
    assert scores[0].selected is True
