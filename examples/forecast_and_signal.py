import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model import Kronos, KronosPredictor, KronosTokenizer, SignalConfig, analyze_forecast, summarize_signal


MODEL_CONFIGS = {
    "mini": {
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "max_context": 2048,
    },
    "small": {
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
    "base": {
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
}


def detect_step(timestamps: pd.Series) -> pd.Timedelta:
    diffs = timestamps.diff().dropna()
    if diffs.empty:
        return pd.Timedelta(minutes=1)
    return diffs.median()


def read_candles(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamps" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "timestamps"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "timestamps"})
        else:
            raise ValueError("CSV must contain timestamps, timestamp, or date column.")

    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").drop_duplicates("timestamps").reset_index(drop=True)

    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    for col in required + [col for col in ["volume", "amount"] if col in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forecast candles with Kronos and turn them into a trade signal.")
    parser.add_argument("--data", required=True, help="CSV containing OHLCV candles.")
    parser.add_argument("--lookback", type=int, default=512, help="Number of historical candles to use.")
    parser.add_argument("--pred-len", type=int, default=32, help="Number of future candles to forecast.")
    parser.add_argument("--model", choices=MODEL_CONFIGS.keys(), default="small", help="Kronos model size.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, or auto.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=1, help="Use 1 for deterministic-ish CPU smoke tests.")
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--min-edge", type=float, default=0.003)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--output", default="outputs/gold_forecast.csv", help="Where to save forecast candles.")
    parser.add_argument("--json-output", default="outputs/gold_signal.json", help="Where to save the signal JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = read_candles(args.data)
    if len(df) < args.lookback:
        raise ValueError(f"Need at least {args.lookback} rows, got {len(df)}.")

    model_config = MODEL_CONFIGS[args.model]
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    lookback = min(args.lookback, model_config["max_context"])
    history = df.iloc[-lookback:].copy()
    feature_cols = ["open", "high", "low", "close"]
    for optional in ["volume", "amount"]:
        if optional in history.columns:
            feature_cols.append(optional)

    step = detect_step(df["timestamps"])
    y_timestamp = pd.Series(
        pd.date_range(
            start=history["timestamps"].iloc[-1] + step,
            periods=args.pred_len,
            freq=step,
        ),
        name="timestamps",
    )

    print(f"Loading {args.model} model on {device}...")
    tokenizer = KronosTokenizer.from_pretrained(model_config["tokenizer_id"])
    model = Kronos.from_pretrained(model_config["model_id"])
    tokenizer.eval()
    model.eval()

    predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config["max_context"])
    with torch.no_grad():
        forecast = predictor.predict(
            df=history[feature_cols].reset_index(drop=True),
            x_timestamp=history["timestamps"].reset_index(drop=True),
            y_timestamp=y_timestamp,
            pred_len=args.pred_len,
            T=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_count=args.sample_count,
            verbose=True,
        )

    forecast = forecast.reset_index().rename(columns={"index": "timestamps"})
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_path, index=False)

    signal = analyze_forecast(
        history,
        forecast,
        SignalConfig(min_edge=args.min_edge, min_confidence=args.min_confidence),
    )
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(signal.to_dict(), indent=2), encoding="utf-8")

    print()
    print(summarize_signal(signal))
    print(f"Forecast saved to: {output_path}")
    print(f"Signal saved to: {json_path}")


if __name__ == "__main__":
    main()
