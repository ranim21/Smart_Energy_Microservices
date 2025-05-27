from flask import Flask, request, jsonify, current_app # Add current_app here
# ... other imports like pandas, your Preprocessor, config, etc.
import pandas as pd
import numpy as np # For telemetry_json_to_rows
from collections import defaultdict

# Assuming config_common.py is in the parent directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config_common as config

from Preprocessing import Preprocessor # Your Preprocessor class

app = Flask(__name__)
logger = config.get_logger('preprocessing_service')

def telemetry_json_to_rows(telemetry_json, keys_to_process, time_col_name=config.DEFAULT_TIME_COL): # Added time_col_name
    all_ts = set()
    logger.info(f"telemetry_json_to_rows: Processing keys: {keys_to_process}, Time column name: {time_col_name}")

    if not isinstance(telemetry_json, dict) or not telemetry_json:
        logger.warning("telemetry_json_to_rows: Received empty or invalid telemetry_json.")
        return pd.DataFrame(columns=[time_col_name] + [k for k in keys_to_process if k != 'ts']) # Exclude 'ts' if it's also a key
    valid_data_keys = [k for k in keys_to_process if k in telemetry_json and telemetry_json[k] and k != 'ts'] # Exclude 'ts' here too
    
    if not valid_data_keys:
        logger.warning(f"telemetry_json_to_rows: No valid data found for keys {keys_to_process} in telemetry_json.")
        cols_for_empty_df = [time_col_name] + [k for k in keys_to_process if k != 'ts' and k != time_col_name]
        return pd.DataFrame(columns=list(dict.fromkeys(cols_for_empty_df))) # Ensure unique columns

    for key in valid_data_keys:
        for dp in telemetry_json[key]:
            if isinstance(dp, dict) and 'ts' in dp:
                all_ts.add(dp['ts'])
            else:
                logger.warning(f"telemetry_json_to_rows: Skipping invalid data point for key '{key}': {dp}")

    if not all_ts:
        logger.warning("telemetry_json_to_rows: No timestamps found in the telemetry data.")
        cols_for_empty_df = [time_col_name] + [k for k in keys_to_process if k != 'ts' and k != time_col_name]
        return pd.DataFrame(columns=list(dict.fromkeys(cols_for_empty_df)))


    sorted_ts = sorted(list(all_ts))
    data_by_ts = {ts: {key: None for key in valid_data_keys} for ts in sorted_ts}

    for key in valid_data_keys:
        for dp in telemetry_json[key]: # dp is like {"ts": ..., "value": "100.5"}
            if isinstance(dp, dict) and 'ts' in dp and 'value' in dp and dp['ts'] in data_by_ts: # Added check for 'value' in dp
                # Assign the actual value from the telemetry data point
                # The value is already a string like "100.5" from your JSON.
                # The Preprocessor will handle converting it to numeric.
                data_by_ts[dp['ts']][key] = dp['value']
            elif isinstance(dp, dict) and 'ts' in dp and dp['ts'] in data_by_ts:
                logger.warning(f"telemetry_json_to_rows: Data point for key '{key}' at ts {dp['ts']} is missing 'value' field: {dp}. Setting to None for this key/ts.")
                # It will remain None as per initialization, which is fine if 'value' is missing.
            elif not (isinstance(dp, dict) and 'ts' in dp):
                 logger.warning(f"telemetry_json_to_rows: Skipping invalid data point structure for key '{key}': {dp}")
    rows = []
    for ts_val in sorted_ts:
        row_dict = {"ts_internal": ts_val} # Use a temporary internal name for the raw timestamp
        row_dict.update(data_by_ts[ts_val])
        rows.append(row_dict)

    df = pd.DataFrame(rows)

    # Create the primary time column (e.g., 'Reading_Time') from 'ts_internal'
    if "ts_internal" in df.columns:
        df[time_col_name] = pd.to_datetime(df['ts_internal'], unit='ms')
        df.drop(columns=['ts_internal'], inplace=True) # Drop the temporary internal ts column
    else: # Should not happen if all_ts was populated
        df[time_col_name] = pd.Series(dtype='datetime64[ns]')


    # Define the final list of columns, ensuring time_col_name is first and unique
    # And all other valid_data_keys are present
    final_column_order = [time_col_name]
    for key in valid_data_keys: # Iterate over keys that actually had data
        if key not in final_column_order: # Add only if not already the time_col_name
            final_column_order.append(key)
    
    # Ensure all columns in final_column_order exist in df, adding them with NaNs if not
    for col in final_column_order:
        if col not in df.columns:
            df[col] = np.nan
            
    df = df[final_column_order] # Reorder and select final columns

    logger.info(f"telemetry_json_to_rows: Created DataFrame with shape {df.shape}. Columns: {df.columns.tolist()}")
    logger.info(f"DEBUG telemetry_json_to_rows: DataFrame BEFORE return. Head:\n{df.head().to_string()}")
    logger.info(f"DEBUG telemetry_json_to_rows: DataFrame dtypes BEFORE return:\n{df.dtypes}")

    return df
@app.route('/preprocess', methods=['POST'])
def preprocess_data_endpoint():
    logger = current_app.logger # Or your specific logger instance
    logger.info("--- PREPROCESSING ENDPOINT HIT ---")

    payload = request.json # This will raise 400 if not JSON or bad Content-Type
    if payload is None: # Should not be strictly necessary if request.json is used without force=True
        logger.error("Payload is None, though request.json should have raised error.")
        return jsonify({"error": "Invalid JSON payload"}), 400

    telemetry_json = payload.get('telemetry_json')
    # Use keys from payload if provided, otherwise default from config or a hardcoded default
    keys_to_process_from_request = payload.get('keys_to_process')
    
    # Decide which keys to ultimately process.
    # Option 1: Always use keys from config if you want a standard set
    # keys_for_telemetry_to_rows = config.TELEMETRY_KEYS
    # Option 2: Use keys from request, fallback to config/default if not provided
    if keys_to_process_from_request:
        keys_for_telemetry_to_rows = keys_to_process_from_request
        logger.info(f"Using keys_to_process from request: {keys_for_telemetry_to_rows}")
    elif hasattr(config, 'DEFAULT_KEYS_TO_PROCESS'):
        keys_for_telemetry_to_rows = config.DEFAULT_KEYS_TO_PROCESS
        logger.info(f"Using default keys_to_process from config: {keys_for_telemetry_to_rows}")
    else:
        keys_for_telemetry_to_rows = ['Ea'] # Hardcoded fallback
        logger.info(f"Using hardcoded fallback keys_to_process: {keys_for_telemetry_to_rows}")


    if not telemetry_json:
        logger.error("Missing 'telemetry_json' in payload.")
        return jsonify({"error": "Missing telemetry_json"}), 400

    logger.info(f"Received telemetry for preprocessing. Keys in source dict: {list(telemetry_json.keys())}. Processing keys: {keys_for_telemetry_to_rows}")

    # 1. Convert JSON to DataFrame
    raw_df = telemetry_json_to_rows(telemetry_json, keys_to_process=keys_for_telemetry_to_rows, time_col_name=config.DEFAULT_TIME_COL)
    
    # Check if raw_df is empty AND anomaly generation is off
    # You'll need a way to control anomaly generation (e.g., Flask app config)
    should_generate_anomalies = current_app.config.get("GENERATE_SYNTHETIC_ANOMALIES", True) # Default to True for testing

    if raw_df.empty and not should_generate_anomalies:
        logger.warning("Initial DataFrame is empty and anomaly generation is off. Returning empty.")
        empty_df_json = pd.DataFrame().to_json(orient="split", date_format="iso", index=False) # Usually index=False if df is truly empty
        return jsonify({"message": "No processable data and anomaly generation off", "processed_data_json": empty_df_json}), 200

    logger.info(f"Raw DataFrame shape: {raw_df.shape}. Columns: {raw_df.columns.tolist()}")

    # 2. Use Preprocessor
    try:
        # numeric_cols_to_convert should be the actual data keys you expect to be numeric
        preprocessor = Preprocessor(df=raw_df.copy(), numeric_cols_to_convert=keys_for_telemetry_to_rows, time_column_name=config.DEFAULT_TIME_COL)
        logger.info(f"Preprocessor initialized. self.df shape: {preprocessor.df.shape if preprocessor.df is not None else 'None'}")

        # --- INTEGRATE ANOMALY GENERATION ---
        if should_generate_anomalies:
            logger.info("Attempting to generate and append anomalous data...")
            # Determine anomaly columns - use keys that were actually processed and are in the df
            # or a default like ['Ea'] if it exists.
            current_df_cols = preprocessor.df.columns.tolist() if preprocessor.df is not None else []
            
            # Select anomaly columns: prefer keys_for_telemetry_to_rows that are in current_df_cols
            anomaly_cols_to_generate = [k for k in keys_for_telemetry_to_rows if k in current_df_cols]
            if not anomaly_cols_to_generate and current_df_cols: # If no overlap, use first available numeric-like column
                anomaly_cols_to_generate = current_df_cols[:1] 
            elif not anomaly_cols_to_generate: # Still no columns (e.g. raw_df was empty, but we want to generate)
                 anomaly_cols_to_generate = ['Ea'] # Fallback if preprocessor.df is empty

            anomaly_start_time = "2025-02-01 00:00:00" # Make configurable or dynamic
            if preprocessor.df is not None and not preprocessor.df.empty and isinstance(preprocessor.df.index, pd.DatetimeIndex):
                latest_real_time = preprocessor.df.index.max()
                # Start anomalies one day after the latest real data point
                anomaly_start_time = (latest_real_time + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

            preprocessor.generate_anomalous_time_series_data(
                num_samples=current_app.config.get("ANOMALY_SAMPLES_COUNT", 50), # Make count configurable
                start_time_str=anomaly_start_time,
                anomaly_columns=anomaly_cols_to_generate
            )
            logger.info(f"After (optional) anomaly generation, self.df shape: {preprocessor.df.shape if preprocessor.df is not None else 'None'}")
        else:
            logger.info("Skipping synthetic anomaly generation.")

        # Continue with pipeline if df is not empty
        if preprocessor.df is None or preprocessor.df.empty:
            logger.warning("DataFrame is empty before resampling (possibly after anomaly gen).")
            empty_df_json = pd.DataFrame().to_json(orient="split", date_format="iso", index=False)
            return jsonify({"message": "Data became empty before resampling", "processed_data_json": empty_df_json}), 200

        # Resample data
        processed_df = preprocessor.resample_data() # This updates preprocessor.df and returns it
        if processed_df is None or processed_df.empty:
            logger.warning("Resampling resulted in no data.")
            empty_df_json = pd.DataFrame().to_json(orient="split", date_format="iso", index=False)
            return jsonify({"message": "Resampling resulted in empty data", "processed_data_json": empty_df_json}), 200
        logger.info(f"After resampling, processed_df shape: {processed_df.shape}")

        # Handle missing values
        processed_df = preprocessor.handle_missing_values() # Updates preprocessor.df and returns it
        if processed_df is None or processed_df.empty:
            logger.warning("Handling missing values resulted in no data.")
            empty_df_json = pd.DataFrame().to_json(orient="split", date_format="iso", index=False)
            return jsonify({"message": "Handling missing values resulted in empty data", "processed_data_json": empty_df_json}), 200
        logger.info(f"After missing value handling, processed_df shape: {processed_df.shape}")
        
        # Final check for DatetimeIndex
        if not isinstance(processed_df.index, pd.DatetimeIndex):
            logger.error("Preprocessing finished, but index is not DatetimeIndex!")
            # Attempt to fix (this part is okay, but ideally Preprocessor methods ensure this)
            if config.DEFAULT_TIME_COL in processed_df.columns: # This should not happen if index is set
                processed_df[config.DEFAULT_TIME_COL] = pd.to_datetime(processed_df[config.DEFAULT_TIME_COL])
                processed_df = processed_df.set_index(config.DEFAULT_TIME_COL)
                logger.info("Attempted to fix index to DatetimeIndex.")
            else:
                 logger.error("Could not fix index. Returning error.")
                 return jsonify({"error": "Processed data index is not DatetimeIndex and could not be fixed"}), 500
        
        logger.info(f"Preprocessing complete. Processed DataFrame shape: {processed_df.shape}. Index type: {type(processed_df.index)}")
        processed_data_json_str = processed_df.to_json(orient="split", date_format="iso", index=True)
        return jsonify({"processed_data_json": processed_data_json_str}), 200

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("Preprocessing Service starting on port 5002...")
    app.run(port=5002, debug=False, host='0.0.0.0')