import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model import Kronos, KronosPredictor, KronosTokenizer, SignalConfig, analyze_forecast


MODEL_CONFIGS = {
    "mini": {
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "max_context": 2048,
        "params": "4.1M",
    },
    "small": {
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
        "params": "24.7M",
    },
}


def read_candles(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamps"])
    required = ["timestamps", "open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestamps").drop_duplicates("timestamps").dropna().reset_index(drop=True)
    return df


def choose_window_starts(row_count: int, lookback: int, pred_len: int, windows: int) -> list[int]:
    first = lookback
    last = row_count - pred_len
    if last <= first:
        raise ValueError(f"Not enough rows for lookback={lookback}, pred_len={pred_len}.")
    if windows <= 1:
        return [last]
    starts = np.linspace(first, last, windows, dtype=int)
    return sorted(set(int(i) for i in starts))


def safe_mape(pred: np.ndarray, actual: np.ndarray) -> float:
    denom = np.where(np.abs(actual) < 1e-12, np.nan, actual)
    return float(np.nanmean(np.abs(pred - actual) / denom) * 100)


def run_one_forecast(
    predictor: KronosPredictor,
    df: pd.DataFrame,
    symbol: str,
    model_name: str,
    start_idx: int,
    lookback: int,
    pred_len: int,
    top_k: int,
    top_p: float,
    temperature: float,
    sample_count: int,
    output_dir: Path,
) -> dict:
    feature_cols = ["open", "high", "low", "close"]
    for optional in ["volume", "amount"]:
        if optional in df.columns:
            feature_cols.append(optional)

    history = df.iloc[start_idx - lookback : start_idx].copy()
    actual = df.iloc[start_idx : start_idx + pred_len].copy()

    with torch.no_grad():
        forecast = predictor.predict(
            df=history[feature_cols].reset_index(drop=True),
            x_timestamp=history["timestamps"].reset_index(drop=True),
            y_timestamp=actual["timestamps"].reset_index(drop=True),
            pred_len=pred_len,
            T=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
        )

    signal = analyze_forecast(history, forecast, SignalConfig(min_edge=0.0015, min_confidence=0.55))

    pred_close = forecast["close"].to_numpy(dtype=float)
    actual_close = actual["close"].to_numpy(dtype=float)
    errors = pred_close - actual_close
    last_close = float(history["close"].iloc[-1])
    pred_ret = float(pred_close[-1] / last_close - 1)
    actual_ret = float(actual_close[-1] / last_close - 1)
    direction_correct = bool(np.sign(pred_ret) == np.sign(actual_ret)) if pred_ret != 0 and actual_ret != 0 else False

    run_name = f"{symbol}_{model_name}_{actual['timestamps'].iloc[0].strftime('%Y%m%d_%H%M')}"
    run_dir = output_dir / "forecasts" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(run_dir / "history.csv", index=False)
    forecast.reset_index().rename(columns={"index": "timestamps"}).to_csv(run_dir / "forecast.csv", index=False)
    actual.to_csv(run_dir / "actual.csv", index=False)
    (run_dir / "signal.json").write_text(json.dumps(signal.to_dict(), indent=2), encoding="utf-8")

    return {
        "symbol": symbol,
        "model": model_name,
        "start_time": actual["timestamps"].iloc[0].isoformat(),
        "end_time": actual["timestamps"].iloc[-1].isoformat(),
        "lookback": lookback,
        "pred_len": pred_len,
        "last_history_close": last_close,
        "forecast_close": float(pred_close[-1]),
        "actual_close": float(actual_close[-1]),
        "forecast_return_pct": pred_ret * 100,
        "actual_return_pct": actual_ret * 100,
        "direction_correct": direction_correct,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mape_pct": safe_mape(pred_close, actual_close),
        "bias": float(np.mean(errors)),
        "signal_action": signal.action,
        "signal_trend": signal.trend,
        "signal_regime": signal.regime,
        "signal_confidence": signal.confidence,
        "signal_position_fraction": signal.position_fraction,
        "signal_reason": signal.reason,
        "run_dir": str(run_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Kronos mini/small across multiple datasets.")
    parser.add_argument("--data", nargs="+", required=True, help="CSV files to test.")
    parser.add_argument("--models", nargs="+", default=["mini", "small"], choices=MODEL_CONFIGS.keys())
    parser.add_argument("--lookback", type=int, default=400)
    parser.add_argument("--pred-len", type=int, default=16)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--output-root", default="outputs/batch_compare")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {Path(path).stem: read_candles(path) for path in args.data}
    all_rows = []

    for model_name in args.models:
        cfg = MODEL_CONFIGS[model_name]
        lookback = min(args.lookback, cfg["max_context"])
        print(f"\nLoading {model_name} ({cfg['params']}) on {args.device}...")
        tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
        model = Kronos.from_pretrained(cfg["model_id"])
        tokenizer.eval()
        model.eval()
        predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=cfg["max_context"])

        for symbol, df in datasets.items():
            starts = choose_window_starts(len(df), lookback, args.pred_len, args.windows)
            print(f"  {symbol}: {len(starts)} windows")
            for idx in starts:
                row = run_one_forecast(
                    predictor=predictor,
                    df=df,
                    symbol=symbol,
                    model_name=model_name,
                    start_idx=idx,
                    lookback=lookback,
                    pred_len=args.pred_len,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    sample_count=args.sample_count,
                    output_dir=output_dir,
                )
                all_rows.append(row)
                print(
                    f"    {row['start_time']} pred={row['forecast_return_pct']:+.2f}% "
                    f"actual={row['actual_return_pct']:+.2f}% "
                    f"dir={row['direction_correct']} action={row['signal_action']}"
                )

    results = pd.DataFrame(all_rows)
    results_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.csv"
    results.to_csv(results_path, index=False)

    summary = (
        results.groupby(["symbol", "model"])
        .agg(
            runs=("symbol", "size"),
            direction_accuracy=("direction_correct", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mape_pct=("mape_pct", "mean"),
            bias=("bias", "mean"),
            avg_forecast_return_pct=("forecast_return_pct", "mean"),
            avg_actual_return_pct=("actual_return_pct", "mean"),
            trade_rate=("signal_action", lambda s: float((s != "hold").mean())),
            avg_signal_confidence=("signal_confidence", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    metadata = {
        "timestamp": timestamp,
        "data_files": args.data,
        "models": args.models,
        "lookback": args.lookback,
        "pred_len": args.pred_len,
        "windows": args.windows,
        "device": args.device,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "sample_count": args.sample_count,
        "results": str(results_path),
        "summary": str(summary_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
