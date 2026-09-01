import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model import SignalConfig, analyze_forecast, summarize_signal


def read_candles(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamps" in df.columns:
        df["timestamps"] = pd.to_datetime(df["timestamps"])
    elif "timestamp" in df.columns:
        df["timestamps"] = pd.to_datetime(df["timestamp"])
    elif "date" in df.columns:
        df["timestamps"] = pd.to_datetime(df["date"])
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn Kronos forecast candles into a trend/regime/trade signal."
    )
    parser.add_argument("--history", required=True, help="CSV with historical OHLC candles.")
    parser.add_argument("--forecast", required=True, help="CSV with forecast OHLC candles.")
    parser.add_argument("--min-edge", type=float, default=0.006, help="Minimum forecast edge, e.g. 0.006 = 0.6%.")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Minimum confidence needed to trade.")
    parser.add_argument("--risk-per-trade", type=float, default=0.01, help="Account risk per trade, e.g. 0.01 = 1%.")
    parser.add_argument("--max-position", type=float, default=0.25, help="Maximum position fraction, e.g. 0.25 = 25%.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    history = read_candles(args.history)
    forecast = read_candles(args.forecast)

    config = SignalConfig(
        min_edge=args.min_edge,
        min_confidence=args.min_confidence,
        risk_per_trade=args.risk_per_trade,
        max_position_fraction=args.max_position,
    )
    signal = analyze_forecast(history, forecast, config)

    if args.json:
        print(json.dumps(signal.to_dict(), indent=2))
        return

    print(summarize_signal(signal))
    print()
    print(f"Action: {signal.action}")
    print(f"Trend: {signal.trend}")
    print(f"Regime: {signal.regime}")
    print(f"Confidence: {signal.confidence:.0%}")
    print(f"Forecast return: {signal.forecast_return:.2%}")
    print(f"Suggested max position: {signal.position_fraction:.1%}")
    if signal.stop_loss is not None:
        print(f"Stop loss: {signal.stop_loss}")
        print(f"Take profit: {signal.take_profit}")


if __name__ == "__main__":
    main()
