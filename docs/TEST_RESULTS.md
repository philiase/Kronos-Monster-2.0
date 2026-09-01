# Test Results

This page records the first public test history for Kronos Monster 2.0. The raw market CSV files and generated forecast bundles are not committed to the repository, but compact result summaries are preserved under `docs/results/`.

## Data Used

The project was tested with cleaned 30-minute OHLCV data using this schema:

```csv
timestamps,open,high,low,close,volume,amount
```

| Dataset | Rows | Date range | Common interval | Notes |
| --- | ---: | --- | --- | --- |
| `GOLD_M30_6MONTHS_CLEAN.csv` | 5,733 | 2025-11-03 01:00:00 to 2026-04-30 23:30:00 | 30 minutes | Six-month gold test data |
| `US30CASH_M30_6MONTHS_CLEAN.csv` | 5,732 | 2025-11-03 01:00:00 to 2026-04-30 23:30:00 | 30 minutes | Six-month US30 test data |
| `US30CASH_M30_CLEAN.csv` | 1,909 | 2026-04-01 01:00:00 to 2026-05-28 23:30:00 | 30 minutes | Initial cross-market batch |
| `US100CASH_M30_CLEAN.csv` | 1,909 | 2026-04-01 01:00:00 to 2026-05-28 23:30:00 | 30 minutes | Initial cross-market batch |
| `USDCAD_M30_CLEAN.csv` | 2,017 | 2026-04-01 00:00:00 to 2026-05-29 00:00:00 | 30 minutes | Initial cross-market batch |
| `USDJPY_M30_CLEAN.csv` | 2,017 | 2026-04-01 00:00:00 to 2026-05-29 00:00:00 | 30 minutes | Initial cross-market batch |
| `GOLD_M30_2years_clean.csv` | 22,516 | 2024-06-03 01:00:00 to 2026-04-30 23:30:00 | 30 minutes | Larger follow-up dataset prepared for later testing |

Non-common gaps exist because market data has session breaks, weekend gaps, holidays, or missing broker candles. The dominant interval is still 30 minutes.

## Test Runs

| Run date | Output source | Data | Models | Lookback | Prediction length | Windows | Device |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-06-01 14:54:32 | `batch_compare/20260601_145432` | US30, US100, USDCAD, USDJPY M30 | mini, small | 400 | 16 | 5 per symbol/model | CPU |
| 2026-06-01 15:04:53 | `us30_deep_compare_h16/20260601_150453` | US30 M30 | mini, small | 400 | 16 | 12 per model | CPU |
| 2026-06-01 15:41:35 | `six_month_compare_h16/20260601_154135` | GOLD and US30 six-month M30 | mini, small | 400 | 16 | 30 per symbol/model | CPU |
| 2026-06-01 16:09:00 | `six_month_compare_summary_combined.csv` | GOLD and US30 six-month M30 | mini, small | 400 | 8, 16, 32 | 30 per symbol/model | CPU |

## Headline Results

The early tests showed that the project works end to end: data loads, Kronos produces candle forecasts, the Monster layer converts those forecasts into market-structure context, and the UI/batch tools save comparable output.

Best early directional results:

| Test | Model | Direction accuracy | Trade direction accuracy | Notes |
| --- | --- | ---: | ---: | --- |
| US30 initial batch, horizon 16 | mini | 100.0% | No trades fired | Direction was right in all 5 windows, but signal layer stayed conservative |
| US30 initial batch, horizon 16 | small | 100.0% | Not isolated in summary | Small fired more signals than mini in the same batch |
| US30 deeper batch, horizon 16 | mini | 75.0% | Not isolated in summary | 12-window deeper run |
| GOLD six-month, horizon 16 | small | 63.3% | 56.3% | Better than mini on six-month GOLD direction |
| US30 six-month, horizon 16 | mini | 63.3% | 58.3% | Matched small on direction with slightly lower error |
| US30 six-month, horizon 16 | small | 63.3% | 90.0% | Best early trade-only result, but only 10 trades |
| GOLD six-month, horizon 8 | small | 53.3% | 80.0% | Shorter horizon, 10 trade signals |

These results are not enough to prove a profitable strategy. They are evidence that the workflow runs, saves outputs, and can be evaluated across instruments, models, horizons, and historical windows.

## Result Files

- `docs/results/initial_batch_summary.csv`
- `docs/results/us30_deep_h16_summary.csv`
- `docs/results/six_month_compare_summary_combined.csv`
- `docs/results/six_month_compare_trade_only.csv`

## Interpretation

- The `small` model often gives stronger trade-filtered outcomes but can be heavier on CPU.
- The `mini` model is useful for fast screening and sometimes matches or beats `small` on direction.
- Shorter horizons are easier to test quickly and may behave better for bias generation.
- Trade-only accuracy must be read with trade count. A high percentage over a small number of trades is promising, not conclusive.
- Candle-level price accuracy remains difficult. The stronger use case is directional bias plus regime filtering, not blind candle-by-candle automation.
