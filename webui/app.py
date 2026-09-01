import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor, analyze_forecast, analyze_market_structure, select_best_scenario, summarize_signal
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

app = Flask(__name__)
CORS(app)

# Global variables to store models
tokenizer = None
model = None
predictor = None
current_model_info = None

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

def load_data_files():
    """Scan data directory and return available data files"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                })
    
    return data_files

def load_data_file(file_path):
    """Load data file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows containing NaN values
        df = df.dropna()
        df = df.sort_values('timestamps').drop_duplicates('timestamps').reset_index(drop=True)
        
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def _records_to_df(records):
    return pd.DataFrame(records) if records else pd.DataFrame()


def _shift_timestamps(values, offset_hours):
    if not offset_hours:
        return values
    return pd.to_datetime(values) + pd.to_timedelta(offset_hours, unit='h')


def _apply_chart_time_offset(df, offset_hours):
    if df is None or len(df) == 0 or not offset_hours:
        return df
    shifted = df.copy()
    if 'timestamps' in shifted.columns:
        shifted['timestamps'] = _shift_timestamps(shifted['timestamps'], offset_hours)
    elif isinstance(shifted.index, pd.DatetimeIndex):
        shifted.index = _shift_timestamps(shifted.index, offset_hours)
    return shifted


def _repair_ohlc(df):
    repaired = df.copy()
    required = ['open', 'high', 'low', 'close']
    if repaired is None or len(repaired) == 0 or not all(col in repaired.columns for col in required):
        return repaired, {'repaired_rows': 0, 'total_rows': 0}

    for col in required:
        repaired[col] = pd.to_numeric(repaired[col], errors='coerce')

    original_high = repaired['high'].copy()
    original_low = repaired['low'].copy()
    candle_max = repaired[required].max(axis=1)
    candle_min = repaired[required].min(axis=1)
    invalid_mask = (original_high < repaired[['open', 'close', 'low']].max(axis=1)) | (
        original_low > repaired[['open', 'close', 'high']].min(axis=1)
    )
    repaired['high'] = candle_max
    repaired['low'] = candle_min
    return repaired, {
        'repaired_rows': int(invalid_mask.sum()),
        'total_rows': int(len(repaired)),
    }


def _records_from_forecast(forecast_df, timestamps):
    records = []
    timestamp_series = pd.Series(timestamps).reset_index(drop=True)
    for i, (_, row) in enumerate(forecast_df.iterrows()):
        records.append({
            'timestamp': timestamp_series[i].isoformat() if i < len(timestamp_series) else f"T{i}",
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']) if 'volume' in row else 0,
            'amount': float(row['amount']) if 'amount' in row else 0,
        })
    return records


def save_prediction_results(
    file_path,
    prediction_type,
    prediction_results,
    actual_data,
    input_data,
    prediction_params,
    model_info=None,
    historical_data=None,
    forecast_data=None,
    actual_df=None,
    trade_signal=None,
    market_structure=None,
    forecast_quality=None,
    scenarios=None,
):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        run_dir = os.path.join(results_dir, f'run_{timestamp}')
        os.makedirs(run_dir, exist_ok=True)

        history_csv = os.path.join(run_dir, 'history.csv')
        forecast_csv = os.path.join(run_dir, 'forecast.csv')
        actual_csv = os.path.join(run_dir, 'actual.csv')
        signal_json = os.path.join(run_dir, 'signal.json')
        structure_json = os.path.join(run_dir, 'market_structure.json')
        scenarios_dir = os.path.join(run_dir, 'scenarios')
        scenarios_json = os.path.join(run_dir, 'scenario_scores.json')
        metadata_json = os.path.join(run_dir, 'metadata.json')

        if historical_data is not None and len(historical_data) > 0:
            historical_data.to_csv(history_csv, index=False)
        if forecast_data is not None and len(forecast_data) > 0:
            forecast_to_save = forecast_data.copy()
            if isinstance(forecast_to_save.index, pd.DatetimeIndex):
                forecast_to_save = forecast_to_save.reset_index().rename(columns={'index': 'timestamps'})
            forecast_to_save.to_csv(forecast_csv, index=False)
        else:
            _records_to_df(prediction_results).to_csv(forecast_csv, index=False)
        if actual_df is not None and len(actual_df) > 0:
            actual_df.to_csv(actual_csv, index=False)
        elif actual_data:
            _records_to_df(actual_data).to_csv(actual_csv, index=False)
        if trade_signal is not None:
            with open(signal_json, 'w', encoding='utf-8') as f:
                json.dump(trade_signal, f, indent=2, ensure_ascii=False)
        if market_structure is not None:
            with open(structure_json, 'w', encoding='utf-8') as f:
                json.dump(market_structure, f, indent=2, ensure_ascii=False)
        if scenarios:
            os.makedirs(scenarios_dir, exist_ok=True)
            scenario_summaries = []
            for scenario in scenarios:
                scenario_path = os.path.join(scenarios_dir, f"scenario_{scenario['index'] + 1:02d}.csv")
                scenario_df = scenario.get('dataframe')
                if scenario_df is not None and len(scenario_df) > 0:
                    scenario_to_save = scenario_df.copy()
                    if isinstance(scenario_to_save.index, pd.DatetimeIndex):
                        scenario_to_save = scenario_to_save.reset_index().rename(columns={'index': 'timestamps'})
                    scenario_to_save.to_csv(scenario_path, index=False)
                scenario_summaries.append({
                    key: value for key, value in scenario.items()
                    if key not in {'dataframe'}
                } | {'file': scenario_path})
            with open(scenarios_json, 'w', encoding='utf-8') as f:
                json.dump(scenario_summaries, f, indent=2, ensure_ascii=False)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'model_info': model_info,
            'trade_signal': trade_signal,
            'market_structure': market_structure,
            'forecast_quality': forecast_quality,
            'scenarios': [
                {key: value for key, value in scenario.items() if key != 'dataframe'}
                for scenario in (scenarios or [])
            ],
            'saved_files': {
                'run_dir': run_dir,
                'metadata': metadata_json,
                'history': history_csv if historical_data is not None and len(historical_data) > 0 else None,
                'forecast': forecast_csv,
                'actual': actual_csv if (actual_df is not None and len(actual_df) > 0) or actual_data else None,
                'signal': signal_json if trade_signal is not None else None,
                'market_structure': structure_json if market_structure is not None else None,
                'scenarios': scenarios_json if scenarios else None,
                'legacy_json': filepath,
            },
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        with open(metadata_json, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        print(f"Evaluation bundle saved to: {run_dir}")
        return save_data['saved_files']
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None


def _price_axis_range(*frames, market_structure=None):
    price_values = []
    for frame in frames:
        if frame is None or len(frame) == 0:
            continue
        for col in ['high', 'low']:
            if col in frame.columns:
                values = pd.to_numeric(frame[col], errors='coerce').dropna()
                if len(values) > 0:
                    price_values.extend(values.tolist())

    if market_structure:
        for group in ['support_levels', 'resistance_levels', 'liquidity_levels']:
            for level in market_structure.get(group, []):
                for key in ['lower', 'upper', 'price']:
                    if key in level:
                        try:
                            price_values.append(float(level[key]))
                        except (TypeError, ValueError):
                            pass

    if not price_values:
        return None

    low = float(np.nanmin(price_values))
    high = float(np.nanmax(price_values))
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if low == high:
        padding = max(abs(low) * 0.01, 1.0)
    else:
        padding = max((high - low) * 0.08, abs(high) * 0.0008)
    return [low - padding, high + padding]


def create_prediction_chart(historical_df, pred_df, actual_df=None, market_structure=None, scenario_forecasts=None):
    """Create prediction chart"""
    show_volume = 'volume' in historical_df.columns and pd.to_numeric(historical_df['volume'], errors='coerce').fillna(0).abs().sum() > 0
    fig = make_subplots(
        rows=2 if show_volume else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.78, 0.22] if show_volume else [1.0],
    )
    
    # Add historical data (candlestick chart)
    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name=f'Historical Data ({len(historical_df)} data points)',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ), row=1, col=1)
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        if isinstance(pred_df.index, pd.DatetimeIndex):
            pred_timestamps = pred_df.index
        elif 'timestamps' in pred_df.columns:
            pred_timestamps = pd.to_datetime(pred_df['timestamps'])
        else:
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=pred_timestamps,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name=f'Prediction Data ({len(pred_df)} data points)',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043'
        ), row=1, col=1)
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        if 'timestamps' in actual_df.columns:
            actual_timestamps = actual_df['timestamps']
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name=f'Actual Data ({len(actual_df)} data points)',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336'
        ), row=1, col=1)

    if scenario_forecasts:
        for scenario in scenario_forecasts:
            if scenario.get('selected'):
                continue
            scenario_df = scenario.get('dataframe')
            if scenario_df is None or len(scenario_df) == 0:
                continue
            if isinstance(scenario_df.index, pd.DatetimeIndex):
                scenario_timestamps = scenario_df.index
            elif 'timestamps' in scenario_df.columns:
                scenario_timestamps = pd.to_datetime(scenario_df['timestamps'])
            else:
                scenario_timestamps = range(len(historical_df), len(historical_df) + len(scenario_df))
            fig.add_trace(go.Scatter(
                x=scenario_timestamps,
                y=scenario_df['close'],
                mode='lines',
                line=dict(color='rgba(113, 128, 150, 0.35)', width=1, dash='dot'),
                name=f"Alt Scenario {scenario['index'] + 1}",
                hovertemplate=f"Alt Scenario {scenario['index'] + 1}<br>%{{x}}<br>Close %{{y}}<extra></extra>",
            ), row=1, col=1)

    if market_structure:
        for level in market_structure.get('support_levels', []):
            fig.add_hrect(
                y0=level['lower'],
                y1=level['upper'],
                fillcolor='rgba(47, 133, 90, 0.14)',
                line_width=0,
                annotation_text=f"S {level['price']:.4f}",
                annotation_position='left',
                row=1,
                col=1,
            )
        for level in market_structure.get('resistance_levels', []):
            fig.add_hrect(
                y0=level['lower'],
                y1=level['upper'],
                fillcolor='rgba(197, 48, 48, 0.14)',
                line_width=0,
                annotation_text=f"R {level['price']:.4f}",
                annotation_position='left',
                row=1,
                col=1,
            )
        for level in market_structure.get('liquidity_levels', []):
            color = '#805ad5' if level['kind'].startswith('buy') else '#2b6cb0'
            dash = 'dot' if level.get('swept') else 'dash'
            label = 'BSL' if level['kind'].startswith('buy') else 'SSL'
            fig.add_hline(
                y=level['price'],
                line_color=color,
                line_dash=dash,
                line_width=1,
                annotation_text=f"{label} {level['price']:.4f}",
                annotation_position='right',
                row=1,
                col=1,
            )
        spike_times = []
        spike_prices = []
        spike_text = []
        for spike in market_structure.get('volume_spikes', []):
            if spike.get('timestamp') is None:
                continue
            spike_times.append(spike['timestamp'])
            spike_prices.append(spike['price'])
            spike_text.append(f"Volume spike: {spike['volume']}")
        if spike_times:
            fig.add_trace(go.Scatter(
                x=spike_times,
                y=spike_prices,
                mode='markers',
                marker=dict(symbol='diamond', size=9, color='#d69e2e'),
                name='Volume Spikes',
                text=spike_text,
                hovertemplate='%{text}<br>%{x}<br>Price %{y}<extra></extra>',
            ), row=1, col=1)

    if show_volume:
        volume_values = pd.to_numeric(historical_df['volume'], errors='coerce').fillna(0)
        candle_up = historical_df['close'].astype(float) >= historical_df['open'].astype(float)
        colors = np.where(candle_up, 'rgba(38, 166, 154, 0.45)', 'rgba(239, 83, 80, 0.45)')
        fig.add_trace(go.Bar(
            x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
            y=volume_values,
            marker_color=colors,
            name='Historical Volume',
            hovertemplate='%{x}<br>Volume %{y}<extra></extra>',
        ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title='Kronos Financial Prediction Results',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_white',
        height=720 if show_volume else 620,
        showlegend=True
    )
    if show_volume:
        fig.update_yaxes(title_text='Volume', row=2, col=1)

    scenario_frames = [scenario.get('dataframe') for scenario in (scenario_forecasts or [])]
    price_range = _price_axis_range(historical_df, pred_df, actual_df, *scenario_frames, market_structure=market_structure)
    if price_range:
        fig.update_yaxes(
            range=price_range,
            fixedrange=False,
            tickformat=',.2f',
            row=1,
            col=1,
        )
    
    # Ensure x-axis time continuity
    if 'timestamps' in historical_df.columns:
        # Get all timestamps and sort them
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        if scenario_forecasts:
            for scenario in scenario_forecasts:
                scenario_df = scenario.get('dataframe')
                if scenario_df is None or len(scenario_df) == 0:
                    continue
                if isinstance(scenario_df.index, pd.DatetimeIndex):
                    all_timestamps.extend(scenario_df.index)
                elif 'timestamps' in scenario_df.columns:
                    all_timestamps.extend(pd.to_datetime(scenario_df['timestamps']))
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=False,
                type='date'
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'Successfully loaded data, total {len(df)} rows'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform prediction"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        prediction_mode = data.get('prediction_mode', 'historical')
        chart_time_offset_hours = float(data.get('chart_time_offset_hours', 0))
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        scenario_mode = data.get('scenario_mode', 'single')
        scenario_count = int(data.get('scenario_count', 1))
        scenario_count = max(1, min(scenario_count, 5))
        if scenario_mode == 'single':
            scenario_count = 1
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        # Load data
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        if len(df) < lookback:
            return jsonify({'error': f'Insufficient data length, need at least {lookback} rows'}), 400
        
        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                # Use real Kronos model
                # Only use necessary columns: OHLCV, excluding amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                start_date = data.get('start_date')
                start_index = data.get('start_index')
                
                if prediction_mode == 'historical':
                    if start_index is not None:
                        start_index = int(start_index)
                    elif start_date:
                        start_dt = pd.to_datetime(start_date)
                        matching = df.index[df['timestamps'] >= start_dt]
                        start_index = int(matching[0]) if len(matching) else 0
                    else:
                        start_index = 0

                    max_start = len(df) - lookback - pred_len
                    if max_start < 0:
                        return jsonify({'error': f'Insufficient data, need at least {lookback + pred_len} rows, currently only {len(df)} available'}), 400
                    start_index = max(0, min(start_index, max_start))
                    end_index = start_index + lookback + pred_len
                    time_range_df = df.iloc[start_index:end_index]

                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']

                    start_timestamp = time_range_df['timestamps'].iloc[0]
                    end_timestamp = time_range_df['timestamps'].iloc[-1]
                    time_span = end_timestamp - start_timestamp
                    
                    prediction_type = f"Kronos model prediction (historical comparison: rows {start_index}-{end_index - 1}, first {lookback} for prediction, next {pred_len} actual, time span: {time_span})"
                else:
                    # Use the most recent data and forecast beyond the file.
                    x_df = df.iloc[-lookback:][required_cols]
                    x_timestamp = df.iloc[-lookback:]['timestamps']
                    time_diff = df['timestamps'].diff().dropna().median() if len(df) > 1 else pd.Timedelta(hours=1)
                    y_timestamp = pd.Series(
                        pd.date_range(
                            start=df['timestamps'].iloc[-1] + time_diff,
                            periods=pred_len,
                            freq=time_diff
                        ),
                        name='timestamps'
                    )
                    prediction_type = "Kronos model prediction (latest data, no future actual comparison)"
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                forecast_paths = []
                forecast_qualities = []
                for scenario_idx in range(scenario_count):
                    raw_pred_df = predictor.predict(
                        df=x_df,
                        x_timestamp=x_timestamp,
                        y_timestamp=y_timestamp,
                        pred_len=pred_len,
                        T=temperature,
                        top_p=top_p,
                        sample_count=sample_count if scenario_count == 1 else 1
                    )
                    repaired_pred_df, path_quality = _repair_ohlc(raw_pred_df)
                    forecast_paths.append(repaired_pred_df)
                    path_quality['scenario_index'] = scenario_idx
                    forecast_qualities.append(path_quality)
                
            except Exception as e:
                return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400
        
        # Prepare actual data for comparison (if exists)
        actual_data = []
        actual_df = None
        
        if prediction_mode == 'historical':
            actual_df = time_range_df.iloc[lookback:lookback+pred_len]
            actual_df_display = _apply_chart_time_offset(actual_df, chart_time_offset_hours)
            for i, (_, row) in enumerate(actual_df.iterrows()):
                display_timestamp = actual_df_display['timestamps'].iloc[i]
                actual_data.append({
                    'timestamp': display_timestamp.isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']) if 'volume' in row else 0,
                    'amount': float(row['amount']) if 'amount' in row else 0
                })
        
        historical_for_chart = x_df.copy()
        historical_for_chart['timestamps'] = pd.Series(x_timestamp).values
        historical_for_chart = _apply_chart_time_offset(historical_for_chart, chart_time_offset_hours)
        actual_df_for_chart = _apply_chart_time_offset(actual_df, chart_time_offset_hours)

        market_structure = None
        try:
            market_structure = analyze_market_structure(historical_for_chart).to_dict()
        except Exception as e:
            print(f"Failed to analyze market structure: {e}")

        selected_scenario_index = 0
        scenario_scores = []
        if scenario_count > 1:
            try:
                selected_scenario_index, scored_scenarios = select_best_scenario(
                    x_df,
                    forecast_paths,
                    market_structure,
                )
                scenario_scores = [score.to_dict() for score in scored_scenarios]
            except Exception as e:
                print(f"Failed to score forecast scenarios: {e}")
                scenario_scores = []
        if not scenario_scores:
            scenario_scores = [{
                'index': 0,
                'score': None,
                'label': 'single raw forecast',
                'direction': None,
                'forecast_return': None,
                'selected': True,
                'reasons': ['scenario selection disabled'],
            }]

        pred_df = forecast_paths[selected_scenario_index]
        forecast_quality = {
            'selected_scenario_index': selected_scenario_index,
            'repaired_rows': int(sum(item.get('repaired_rows', 0) for item in forecast_qualities)),
            'total_rows': int(sum(item.get('total_rows', 0) for item in forecast_qualities)),
            'paths': forecast_qualities,
        }
        pred_df_for_chart = _apply_chart_time_offset(pred_df, chart_time_offset_hours)

        scenario_payload = []
        for idx, scenario_df in enumerate(forecast_paths):
            display_df = _apply_chart_time_offset(scenario_df, chart_time_offset_hours)
            score_info = next((score for score in scenario_scores if score.get('index') == idx), {})
            scenario_payload.append({
                'index': idx,
                'selected': idx == selected_scenario_index,
                'score': score_info.get('score'),
                'label': score_info.get('label', f'Scenario {idx + 1}'),
                'direction': score_info.get('direction'),
                'forecast_return': score_info.get('forecast_return'),
                'reasons': score_info.get('reasons', []),
                'dataframe': display_df,
            })

        chart_json = create_prediction_chart(
            historical_for_chart,
            pred_df_for_chart,
            actual_df_for_chart,
            market_structure,
            scenario_payload if scenario_count > 1 else None,
        )
        
        # Use the same timestamps that were passed into Kronos and used by the chart.
        if isinstance(pred_df_for_chart.index, pd.DatetimeIndex):
            future_timestamps = pd.Series(pred_df_for_chart.index)
        else:
            future_timestamps = pd.Series(_shift_timestamps(y_timestamp, chart_time_offset_hours))
        
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f"T{i}",
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })

        trade_signal = None
        try:
            signal = analyze_forecast(x_df, pred_df, market_structure=market_structure)
            trade_signal = signal.to_dict()
            trade_signal['summary'] = summarize_signal(signal)
        except Exception as e:
            print(f"Failed to analyze trade signal: {e}")
        
        # Save prediction results to file
        saved_files = None
        try:
            saved_files = save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'scenario_mode': scenario_mode,
                    'scenario_count': scenario_count,
                    'selected_scenario_index': selected_scenario_index,
                    'prediction_mode': prediction_mode,
                    'chart_time_offset_hours': chart_time_offset_hours,
                    'start_date': start_date if start_date else None,
                    'start_index': start_index if prediction_mode == 'historical' else None,
                },
                model_info=current_model_info,
                historical_data=historical_for_chart,
                forecast_data=pred_df_for_chart,
                actual_df=actual_df_for_chart,
                trade_signal=trade_signal,
                market_structure=market_structure,
                forecast_quality=forecast_quality,
                scenarios=scenario_payload,
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")
        
        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart_time_offset_hours': chart_time_offset_hours,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'trade_signal': trade_signal,
            'market_structure': market_structure,
            'forecast_quality': forecast_quality,
            'scenarios': [
                {key: value for key, value in scenario.items() if key != 'dataframe'}
                for scenario in scenario_payload
            ],
            'actual_data': actual_data,
            'saved_files': saved_files,
            'has_comparison': len(actual_data) > 0,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '')
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor, current_model_info
    
    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Load tokenizer and model
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        current_model_info = {
            'key': model_key,
            'name': model_config['name'],
            'model_id': model_config['model_id'],
            'tokenizer_id': model_config['tokenizer_id'],
            'params': model_config['params'],
            'context_length': model_config['context_length'],
            'device': device,
        }
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
            'model_info': {
                'name': model_config['name'],
                'key': model_key,
                'model_id': model_config['model_id'],
                'tokenizer_id': model_config['tokenizer_id'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'device': device,
                'description': model_config['description']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': predictor.model.__class__.__name__,
                    'device': str(next(predictor.model.parameters()).device),
                    'loaded_model': current_model_info,
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")
    
    app.run(debug=True, host='0.0.0.0', port=7070)
