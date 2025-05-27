from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import json
import keras
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config_common as config

app = Flask(__name__)
logger = config.get_logger('forecasting_service')

MODEL = None
SCALER = None

def load_model_and_scaler():
    global MODEL, SCALER
    model_full_path = os.path.join(os.path.dirname(__file__), config.MODEL_PATH_REL)
    scaler_full_path = os.path.join(os.path.dirname(__file__), config.SCALER_PATH_REL)
    try:
        if os.path.exists(model_full_path):
            MODEL = keras.models.load_model(model_full_path)
            logger.info(f"Keras model loaded successfully from {model_full_path}")
        else:
            logger.error(f"Model file not found: {model_full_path}")
            raise FileNotFoundError(f"Model file not found: {model_full_path}")

        if os.path.exists(scaler_full_path):
            SCALER = joblib.load(scaler_full_path)
            logger.info(f"Scaler loaded successfully from {scaler_full_path}")
        else:
            logger.error(f"Scaler file not found: {scaler_full_path}")
            raise FileNotFoundError(f"Scaler file not found: {scaler_full_path}")

    except Exception as e:
        logger.error(f"Failed to load model or scaler: {e}", exc_info=True)

def inverse_transform_scaled(scaled_data, scaler, target_column_index=0):
    if not hasattr(scaler, 'n_features_in_'): raise ValueError("Scaler not fitted.")
    dummy_array = np.zeros((len(scaled_data), scaler.n_features_in_))
    if scaled_data.ndim == 1: dummy_array[:, target_column_index] = scaled_data
    elif scaled_data.ndim == 2 and scaled_data.shape[1] == 1: dummy_array[:, target_column_index] = scaled_data.flatten()
    elif scaled_data.ndim == 2 and scaled_data.shape[1] > 1 and scaler.n_features_in_ == 1: dummy_array[:, target_column_index] = scaled_data.flatten()
    elif scaled_data.ndim == 2 and scaled_data.shape[1] == scaler.n_features_in_: dummy_array = scaled_data
    else: raise ValueError(f"Shape mismatch: Scaled data {scaled_data.shape}, scaler features {scaler.n_features_in_}")
    try:
        inverted_data = scaler.inverse_transform(dummy_array)
        return inverted_data[:, target_column_index]
    except ValueError as e: logger.error(f"Inverse transform error: {e}. Dummy shape: {dummy_array.shape}"); raise

def energy_forecasting_model_core(df, model, scaler, time_steps=config.TIME_STEPS, expected_horizon=config.EXPECTED_FORECAST_HORIZON):
    if not isinstance(df.index, pd.DatetimeIndex):
        if config.DEFAULT_TIME_COL in df.columns:
            df[config.DEFAULT_TIME_COL] = pd.to_datetime(df[config.DEFAULT_TIME_COL], errors='coerce')
            df.dropna(subset=[config.DEFAULT_TIME_COL], inplace=True)
            if df.empty: return np.array([]), None
            df = df.set_index(config.DEFAULT_TIME_COL)
        else:
            raise ValueError("Input DataFrame must have a DatetimeIndex or a 'Reading_Time' column.")

    df = df.sort_index()
    if df.empty or "Ea" not in df.columns: return np.array([]), None
    df['Ea'] = pd.to_numeric(df['Ea'], errors='coerce')
    if df['Ea'].isnull().values.any(): return np.array([]), None

    features = df[['Ea']].astype("float32").to_numpy() # Use double brackets for 2D array
    if features.shape[0] == 0: return np.array([]), None

    try: features_scaled = scaler.transform(features)
    except Exception as e: logger.error(f"Error scaling features: {e}"); return np.array([]), None

    n_samples, n_features = features_scaled.shape[0], features_scaled.shape[1]
    if n_samples < time_steps: logger.warning(f"Not enough samples ({n_samples}) for time_steps ({time_steps})"); return np.array([]), None

    X = np.array([features_scaled[i:(i + time_steps), :] for i in range(n_samples - time_steps + 1)])
    if not X.size: return np.array([]), None # Check if X is empty
    if X.shape[1] != time_steps or X.shape[2] != n_features: raise ValueError("Sequence shape mismatch.")

    y_pred_scaled = model.predict(X)
    logger.info(f"Model prediction output shape: {y_pred_scaled.shape}")

    # Adjust flattening based on model output (samples, horizon) or (samples, horizon, 1)
    if y_pred_scaled.ndim == 2 and y_pred_scaled.shape[0] == X.shape[0] and y_pred_scaled.shape[1] == expected_horizon:
        y_pred_scaled_flat = y_pred_scaled.flatten()
    elif y_pred_scaled.ndim == 3 and y_pred_scaled.shape[0] == X.shape[0] and y_pred_scaled.shape[1] == expected_horizon and y_pred_scaled.shape[2] == 1:
        y_pred_scaled_flat = y_pred_scaled.reshape(-1)
    else:
        logger.error(f"Model output shape {y_pred_scaled.shape} unexpected.")
        return np.array([]), None

    y_pred_real = inverse_transform_scaled(y_pred_scaled_flat, scaler, target_column_index=0)
    if not y_pred_real.size: return np.array([]), None

    last_input_timestamp = df.index[-1]
    freq = pd.infer_freq(df.index) # freq can be a string like "5T" or None
    logger.info(f"Initially inferred frequency: {freq} (type: {type(freq)})") # Good to log this

    if freq is None:
            # Fallback: Calculate median difference
            diffs = df.index.to_series().diff() # Series of Timedeltas (first element is NaT)
            # logger.debug(f"Timestamp differences: {diffs}")
            if len(diffs) > 1: # Need at least two differences to get a median of valid Timedeltas
                median_diff = diffs.median() # This will be a Timedelta object
                if pd.notna(median_diff) and median_diff > pd.Timedelta(0):
                    freq = median_diff # Now freq is a Timedelta object
                    logger.info(f"Using fallback median frequency: {freq} (type: {type(freq)})")
                else:
                    logger.warning(f"Fallback median frequency is NaT or non-positive: {median_diff}. Cannot determine frequency.")
                    freq = None # Ensure freq remains None if fallback also fails
            else:
                logger.warning("Not enough data points to calculate median frequency difference.")
                freq = None
            if freq is None: # This covers pd.isna(freq) for None itself, and cases where fallback set it to None
                logger.error("Cannot determine a valid frequency for the time series.")
                return np.array([]), None # Return empty predictions and no timestamps
        
        # If freq is a Timedelta from fallback, the <= comparison is valid:
            if isinstance(freq, pd.Timedelta) and freq <= pd.Timedelta(0):
                logger.error(f"Determined frequency ({freq}) is zero or negative, which is invalid.")
                return np.array([]), None

        # If freq is a string (e.g., "5T") from infer_freq, it's generally considered valid.
        # The previous TypeError happened because you can't do "5T" <= Timedelta(0).
        # We now handle string freq and Timedelta freq separately or ensure freq is always Timedelta before comparison.

        # At this point, 'freq' is either a valid frequency string OR a valid positive Timedelta object.
        # pd.date_range can accept either a frequency string or a Timedelta for its 'freq' parameter.

    num_predictions = len(y_pred_real)
    # The model predicts for 'expected_horizon' steps *after* each input sequence.

    if y_pred_scaled.ndim == 2 : # (num_sequences, expected_horizon)
        y_pred_scaled_last_window = y_pred_scaled[-1, :] # Shape (expected_horizon,)
    elif y_pred_scaled.ndim == 3 : # (num_sequences, expected_horizon, 1)
        y_pred_scaled_last_window = y_pred_scaled[-1, :, 0] # Shape (expected_horizon,)
    else:
        logger.error("Unhandled y_pred_scaled dimension for last window extraction.")
        return np.array([]), None

    y_pred_real_last_window = inverse_transform_scaled(y_pred_scaled_last_window, scaler, target_column_index=0)
    num_predictions_last_window = len(y_pred_real_last_window)

    # Timestamps for these predictions start *after* the last timestamp of the *last input window*
    # The last input window ends at `df.index[-1]`.
    # So, predictions are for `df.index[-1] + 1*freq`, `df.index[-1] + 2*freq`, ..., `df.index[-1] + num_predictions_last_window*freq`
    pred_timestamps_plus_one = pd.date_range(start=last_input_timestamp, periods=num_predictions_last_window + 1, freq=freq)
    pred_timestamps = pred_timestamps_plus_one[1:] # Exclude the start_ts itself

    if len(pred_timestamps) != num_predictions_last_window:
        logger.error(f"Timestamp generation mismatch for last window. Expected {num_predictions_last_window}, got {len(pred_timestamps)}.")
        return np.array([]), None

    pred_timestamps_ms = (pred_timestamps.astype(np.int64) // 10**6).tolist()
    
    return y_pred_real_last_window, pred_timestamps_ms


def predictions_to_json_list(predictions_values, predictions_timestamps_ms):
    # (Copied from your script)
    if len(predictions_values) != len(predictions_timestamps_ms): return []
    return [{"ts": ts, "values": float(val)} for ts, val in zip(predictions_timestamps_ms, predictions_values)]

def post_single_prediction(telemetry_data_point, target_device_token, telemetry_key="Forcasted_Energy"):
    # (Adapted from your Post_Predictions, uses device token in URL)
    url = f"{config.TARGET_URL}/api/v1/{target_device_token}/telemetry"
    payload = {"ts": telemetry_data_point["ts"], "values": {telemetry_key: telemetry_data_point["values"]}}
    try:
        resp = requests.post(url, json=payload, timeout=config.API_TIMEOUT)
        config.log_http_request_detail(logger, url, 'POST', json_payload=payload, response=resp)
        resp.raise_for_status()
        logger.info(f"Successfully posted prediction to {url.split('/api')[0]}. Status: {resp.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to post prediction to {url.split('/api')[0]}: {e}")
        if hasattr(e, 'response') and e.response is not None: logger.error(f"Failed POST response: {e.response.text}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error posting prediction: {e}", exc_info=True)
        return False

@app.route('/forecast-and-post', methods=['POST'])
def forecast_and_post_endpoint():
    if MODEL is None or SCALER is None:
        logger.error("Model or Scaler not loaded. Cannot forecast.")
        return jsonify({"error": "Forecasting model/scaler not available"}), 503 # Service Unavailable

    payload = request.get_json(silent=True)
    processed_data_json_str = payload.get('processed_data_json')
    target_device_token = payload.get('target_device_access_token') 

    if not processed_data_json_str or not target_device_token:
        return jsonify({"error": "Missing processed_data_json or target_device_access_token"}), 400

    try:
        # orient='split' and index=True was used in preprocessing_service
        # The index will be the first element in 'index' list of the split JSON
        processed_df = pd.read_json(processed_data_json_str, orient="split")
        # The index from `to_json(orient="split", index=True)` needs to be set
        # It's often better to ensure 'Reading_Time' is a column if passing between services
        # or ensure the index is explicitly named and then set.
        # Assuming 'Reading_Time' is the name of the index from preprocessing
        if isinstance(processed_df.index, pd.RangeIndex) and config.DEFAULT_TIME_COL in processed_df.columns:
             processed_df[config.DEFAULT_TIME_COL] = pd.to_datetime(processed_df[config.DEFAULT_TIME_COL])
             processed_df = processed_df.set_index(config.DEFAULT_TIME_COL)
        elif not isinstance(processed_df.index, pd.DatetimeIndex):
             # If index is not datetime and 'Reading_Time' column doesn't exist, this is an issue
             logger.error(f"Processed DataFrame has no DatetimeIndex. Index type: {type(processed_df.index)}, Columns: {processed_df.columns}")
             return jsonify({"error": "Processed data does not have a valid DatetimeIndex."}), 400


        logger.info(f"Received preprocessed data for forecasting. Shape: {processed_df.shape}, Index type: {type(processed_df.index)}")
        if processed_df.empty or 'Ea' not in processed_df.columns or processed_df['Ea'].isnull().all():
            logger.warning("DataFrame for forecasting is empty or lacks 'Ea' data.")
            return jsonify({"message": "No data for forecasting", "predictions_posted": 0}), 200

        predictions_values, predictions_timestamps_ms = energy_forecasting_model_core(
            processed_df.copy(), MODEL, SCALER
        )

        if predictions_values is None or predictions_timestamps_ms is None or predictions_values.size == 0:
            logger.warning("Forecasting model did not return valid predictions/timestamps.")
            return jsonify({"message": "No predictions generated", "predictions_posted": 0}), 200

        formatted_predictions = predictions_to_json_list(predictions_values, predictions_timestamps_ms)
        if not formatted_predictions:
            logger.warning("Failed to format predictions into JSON list.")
            return jsonify({"message": "Failed to format predictions", "predictions_posted": 0}), 200

        logger.info(f"Generated {len(formatted_predictions)} prediction points. Posting...")
        successful_posts = 0
        for pred_item in formatted_predictions:
            if post_single_prediction(pred_item, target_device_token):
                successful_posts += 1
            time.sleep(0.05) # Small delay

        logger.info(f"Finished posting predictions. Successful: {successful_posts}/{len(formatted_predictions)}")
        return jsonify({
            "message": "Forecasting and posting complete.",
            "predictions_generated": len(formatted_predictions),
            "predictions_posted": successful_posts
        }), 200

    except FileNotFoundError as e: # Catch if model/scaler path was wrong during call (should be caught at startup)
        logger.error(f"Model/Scaler file not found: {e}", exc_info=True)
        return jsonify({"error": f"Configuration error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Error during forecasting or posting: {e}", exc_info=True)
        return jsonify({"error": f"Forecasting/posting failed: {str(e)}"}), 500

if __name__ == '__main__':
    load_model_and_scaler() # Load model when service starts
    if MODEL is None or SCALER is None:
        logger.warning("Forecasting Service starting WITH ERRORS (Model/Scaler not loaded). Forecasts will fail.")
    else:
        logger.info("Forecasting Service starting on port 5003 with model and scaler loaded.")
    app.run(port=5003, debug=False, host='0.0.0.0')