# Kronos Monster 2.0

Kronos Monster 2.0 is an experimental market-forecasting and trading-bias research tool built on top of the open-source Kronos financial candlestick foundation model.

The project keeps Kronos as the forecast engine, then adds a practical decision layer around it:

```text
Kronos candlestick forecast -> market structure -> liquidity and volume context -> scenario selection -> trade bias
```

The goal is not to treat any model as a perfect market predictor. The goal is to evaluate forecast paths through the same kind of context a trader watches: trend, support and resistance, liquidity sweeps, volatility, volume participation, and bad-regime filtering.

## Features

- Kronos mini, small, and base model support through Hugging Face.
- Web UI for loading OHLCV data, running forecasts, and reviewing results.
- Historical comparison mode for checking predictions against known future candles.
- Latest forecast mode for projecting beyond the newest candle in the dataset.
- Trade-bias signal layer: long, short, or hold.
- Support and resistance zone detection from swing highs and lows.
- Buy-side and sell-side liquidity level detection.
- Volume state detection using tick volume or regular volume.
- Volume spike markers on the chart.
- Forecast candle repair for invalid generated OHLC candles.
- Scenario selection engine that runs multiple forecast paths and chooses the path that best fits current market structure.
- Saved evaluation bundles for every run.
- Stretchable Plotly chart with pan, zoom, wide view, and reset zoom.
- Display-only chart time offset to align broker/platform time.

## Project Structure

```text
model/
  kronos.py              Original Kronos model interface
  module.py              Original Kronos model modules
  signals.py             Trade-bias signal layer
  market_structure.py    Support/resistance, liquidity, volume, regime analysis
  scenarios.py           Multi-path forecast scoring and selection

webui/
  app.py                 Flask API, prediction pipeline, chart generation
  serve.py               Simple local server entrypoint
  templates/index.html   Web UI

examples/
  batch_model_compare.py Batch evaluation for mini/small model comparisons
  forecast_and_signal.py CLI forecast plus signal report
  trade_signal_report.py Convert saved forecasts into signal summaries

tests/
  test_signals.py
  test_market_structure.py
  test_scenarios.py
```

## Attribution

This repository is a derivative work based on the original Kronos project:

- Original repository: [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
- Paper: [Kronos: A Foundation Model for the Language of Financial Markets](https://arxiv.org/abs/2508.02739)
- Pretrained models: [NeoQuasar on Hugging Face](https://huggingface.co/NeoQuasar)

The original Kronos code is licensed under the MIT License. The original copyright notice is preserved in `LICENSE`.

## Installation

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r webui\requirements.txt
```

For CPU-only machines, install the CPU PyTorch build if needed:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu
```

## Running the Web UI

Windows:

```powershell
start_monster_ui.bat
```

Manual run:

```powershell
cd webui
..\.venv\Scripts\python.exe serve.py
```

Then open:

```text
http://127.0.0.1:7070
```

## Data Format

Place CSV files in the `data/` directory. The expected format is:

```csv
timestamps,open,high,low,close,volume,amount
2026-01-01 00:00:00,100.0,101.0,99.5,100.5,1200,120600
```

Required columns:

- `timestamps`
- `open`
- `high`
- `low`
- `close`

Recommended columns:

- `volume`
- `amount`

If only tick volume is available, it can be used as `volume`.

Local datasets are ignored by Git by default. Add your own datasets locally when running experiments.

## Recommended CPU Settings

For CPU-only testing:

- Start with `Kronos-mini`.
- Use `Fast scenario mode (3 paths)`.
- Use prediction length `8` or `16`.
- Use lookback `400` or lower.
- Move to `Kronos-small` only when a setup looks promising.
- Avoid large scenario counts for routine testing.

## Scenario Selection

The scenario selector does not modify Kronos internals.

It runs multiple Kronos forecast paths, repairs invalid OHLC candles if needed, then scores each path using:

- structure agreement
- support/resistance behavior
- liquidity sweep or rejection behavior
- candle realism
- volume context

The selected scenario becomes the main forecast displayed in the UI. Alternate scenarios are saved and can be plotted as faint close-path lines.

## Saved Results

Each prediction run can save an evaluation bundle under:

```text
webui/prediction_results/run_YYYYMMDD_HHMMSS/
```

Typical files:

- `history.csv`
- `forecast.csv`
- `actual.csv` when historical comparison is available
- `signal.json`
- `market_structure.json`
- `scenario_scores.json`
- `scenarios/scenario_01.csv`, etc.
- `metadata.json`

These outputs are ignored by Git.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_signals.py tests\test_market_structure.py tests\test_scenarios.py -q
```

## Research Notes

The most useful evaluation is not a single attractive chart. Compare these over many historical windows:

- raw Kronos direction
- selected scenario direction
- final close error
- MAPE
- trade-bias action
- whether hold decisions filtered bad regimes
- whether scenario selection beats the raw forecast

## Disclaimer

This project is for research and experimentation. It is not financial advice and should not be used as an automated trading system without independent validation, risk controls, and forward testing.
