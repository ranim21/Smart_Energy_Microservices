from io import StringIO
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # Changed from discriminant_analysis
import requests
import json
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config_common as config

app = Flask(__name__)
logger = config.get_logger('anomaly_service')

def create_formatted_alarm_json(alarm_type, originator_id, severity, start_ts, details_dict):
    return {
        "type": alarm_type, "originator": {"id": originator_id, "entityType": "DEVICE"},
        "severity": severity, "acknowledged": False, "cleared": False,
        "startTs": int(start_ts), "endTs": int(start_ts), # endTs same as startTs for point anomalies
        "details": details_dict, "propagate": True
    }

def detect_anomalies_and_generate_alarms_core(
    input_df: pd.DataFrame, originator_id: str,
    time_column_name: str = config.DEFAULT_TIME_COL,
    feature_columns_for_kmeans= None,
    optimal_k: int = 2, threshold_percentile: float = 97.0,
    nan_fill_strategy: str = 'mean'
) -> list:
    if feature_columns_for_kmeans is None: feature_columns_for_kmeans = config.DEFAULT_KMEANS_FEATURES.copy()
    if not isinstance(input_df, pd.DataFrame) or input_df.empty: return []
    
    df = input_df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and time_column_name not in df.columns:
        df.reset_index(inplace=True) 
    if time_column_name not in df.columns: logger.error(f"Time column '{time_column_name}' missing."); return []
    
    df['timestamp_dt'] = pd.to_datetime(df[time_column_name], errors='coerce')
    df.dropna(subset=['timestamp_dt'], inplace=True)
    if df.empty: return []
    df['startTs_ms'] = (df['timestamp_dt'].astype(np.int64) // 10**6)

    missing_features = [col for col in feature_columns_for_kmeans if col not in df.columns]
    if missing_features: logger.error(f"Missing K-Means features: {missing_features}"); return []

    df_to_scale = df[feature_columns_for_kmeans].copy()
    if df_to_scale.isnull().values.any():
        if nan_fill_strategy == 'mean': df_to_scale = df_to_scale.fillna(df_to_scale.mean())
        elif nan_fill_strategy == 'ffill': df_to_scale = df_to_scale.ffill().bfill() 
        else: df_to_scale = df_to_scale.fillna(0)
        if df_to_scale.isnull().values.any(): df_to_scale = df_to_scale.fillna(0) 

    if df_to_scale.empty: return []
        
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_to_scale)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    try: kmeans.fit(df_scaled)
    except ValueError: logger.error("Error fitting KMeans (check NaNs/empty df_scaled)"); return []

    df['Distance_to_Cluster_Center'] = kmeans.transform(df_scaled).min(axis=1)
    distance_threshold = np.percentile(df['Distance_to_Cluster_Center'], threshold_percentile)
    df['Is_Anomaly_KMeans'] = df['Distance_to_Cluster_Center'] > distance_threshold
    anomalies_df = df[df['Is_Anomaly_KMeans']].copy()

    if anomalies_df.empty: return []
    
    generated_alarms = []
    for _, anomaly_row in anomalies_df.iterrows():
        feature_name = "Ea" 
        feature_value = anomaly_row[feature_name]
        if pd.isna(feature_value): continue

        severity = "WARNING"
        if feature_value < df[feature_name].quantile(0.05) or feature_value > df[feature_name].quantile(0.95):
            severity = "CRITICAL"
        
        direction = "High" if feature_value > df[feature_name].mean() else "Low"
        alarm_type = f"{direction} {feature_name} Anomaly"
        details = {
            "message": f"{feature_name} value {feature_value:.2f} is anomalous.",
            "feature": feature_name, "value": f"{feature_value:.2f}",
            "kmeans_distance": f"{anomaly_row['Distance_to_Cluster_Center']:.4f}"
        }
        alarm_json = create_formatted_alarm_json(alarm_type, originator_id, severity, anomaly_row['startTs_ms'], details)
        generated_alarms.append(alarm_json)
    return generated_alarms

def post_alarms_to_target(alarms_list, target_alarm_url=config.TARGET_ALARM_URL, headers=config.TARGET_HEADERS):
    if not alarms_list: return 0
    successful_posts = 0
    for alarm_payload in alarms_list:
        try:
            response = requests.post(target_alarm_url, data=json.dumps(alarm_payload), headers=headers, timeout=config.API_TIMEOUT)
            config.log_http_request_detail(logger, target_alarm_url, 'POST', json_payload=alarm_payload, response=response, headers=headers)
            response.raise_for_status()
            successful_posts += 1
        except requests.exceptions.RequestException as e:
            logger.error(f"Error posting alarm: {e}")
            if hasattr(e, 'response') and e.response is not None: logger.error(f"Failed alarm POST response: {e.response.text}")
        except Exception as e: logger.error(f"Unexpected error posting alarm: {e}", exc_info=True)
        time.sleep(0.1)
    return successful_posts

def get_recommendation_rule_based_core(alarm_json):
    alarm_type = alarm_json.get("type")
    feature = alarm_json.get("details", {}).get("feature")
    value = alarm_json.get("details", {}).get("value")
    ts = alarm_json.get("endTs") # or startTs

    if feature == "Ea":
        if "Low" in alarm_type:
            return {"Ts": ts, "Recommendations": f"Energy 'Ea' ({value}) is low. Check sensor, calibration, and related components."}
        elif "High" in alarm_type:
            return {"Ts": ts, "Recommendations": f"Energy 'Ea' ({value}) is high. Review usage patterns, check for meter issues."}
    return {"Ts": ts, "Recommendations": "General check recommended for anomalous reading."}


def post_single_recommendation(recommendation_payload, target_device_token, telemetry_key="Recommendation"):
    url = f"{config.TARGET_URL}/api/v1/{target_device_token}/telemetry"
    payload = {"ts": recommendation_payload["Ts"], "values": {telemetry_key: recommendation_payload["Recommendations"]}}
    try:
        resp = requests.post(url, json=payload, timeout=config.API_TIMEOUT)
        config.log_http_request_detail(logger, url, 'POST', json_payload=payload, response=resp)
        resp.raise_for_status()
        logger.info(f"Successfully posted recommendation to {url.split('/api')[0]}.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to post recommendation: {e}")
        if hasattr(e, 'response') and e.response is not None: logger.error(f"Failed recommendation POST: {e.response.text}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error posting recommendation: {e}", exc_info=True)
        return False

@app.route('/detect-alarms-recommendations', methods=['POST'])
def detect_alarms_recommendations_endpoint():
    payload = request.get_json()
    if not payload:
        logger.error("Request payload is empty or not valid JSON.")
        return jsonify({"error": "Request payload is empty or not valid JSON."}), 400

    processed_data_json_str = payload.get('processed_data_json')
    originator_id = payload.get('originator_id')
    target_device_access_token = payload.get('target_device_access_token') # Using this from payload

    if not all([processed_data_json_str, originator_id, target_device_access_token]):
        logger.error("Missing one or more required fields: processed_data_json, originator_id, target_device_access_token")
        return jsonify({"error": "Missing processed_data_json, originator_id, or target_device_access_token"}), 400

    if not isinstance(processed_data_json_str, str):
        logger.error(f"Expected 'processed_data_json' to be a string, got {type(processed_data_json_str)}")
        return jsonify({"error": "processed_data_json must be a JSON string"}), 400

    try:
        processed_df = pd.read_json(StringIO(processed_data_json_str), orient="split")
        logger.info(f"Successfully parsed JSON to DataFrame. Shape: {processed_df.shape}")
        logger.debug(f"DF head after read_json:\n{processed_df.head().to_string()}")
        logger.debug(f"DF index type after read_json: {type(processed_df.index)}")
        logger.debug(f"DF index name after read_json: {processed_df.index.name}") # Might be None initially
        logger.debug(f"DF columns after read_json: {processed_df.columns.tolist()}")

        if not isinstance(processed_df.index, pd.DatetimeIndex):
            logger.info(f"DataFrame index is currently of type: {type(processed_df.index)}. Attempting conversion to DatetimeIndex.")
            try:
                processed_df.index = pd.to_datetime(processed_df.index, unit='ms')
                logger.info("Successfully converted DataFrame index to DatetimeIndex (assumed unit='ms').")
            except ValueError as ve:
                logger.warning(f"Could not convert index to DatetimeIndex using unit='ms' ({ve}). Trying auto-parsing.")
                try:
                    processed_df.index = pd.to_datetime(processed_df.index)
                    logger.info("Successfully converted DataFrame index to DatetimeIndex (auto-parsed).")
                except Exception as e_auto:
                    logger.error(f"Failed to convert index to DatetimeIndex after multiple attempts: {e_auto}", exc_info=True)
                    return jsonify({"error": "Failed to parse time index in processed data. Check data format."}), 500
        else:
            logger.info("DataFrame index is already a DatetimeIndex.")

        processed_df.index.name = 'Reading_Time' 
        logger.info(f"DataFrame index name set to 'Reading_Time'. Current index type: {type(processed_df.index)}")

        df_for_detection = processed_df.reset_index()
        logger.info(f"DataFrame prepared for detection (after reset_index). Columns: {df_for_detection.columns.tolist()}")

        kmeans_features = config.DEFAULT_KMEANS_FEATURES.copy() 
        actual_kmeans_features = [f for f in kmeans_features if f in df_for_detection.columns] 

        if not actual_kmeans_features:
            logger.warning(f"None of the specified K-Means features ({kmeans_features}) found in DataFrame columns: {df_for_detection.columns.tolist()}. Skipping anomaly detection.")
            return jsonify({"message": "K-Means features not found.", "alarms_generated": 0, "recommendations_posted": 0}), 200

        alarms = detect_anomalies_and_generate_alarms_core(
            df_for_detection,
            originator_id,
            feature_columns_for_kmeans=actual_kmeans_features,
            optimal_k=10, 
            threshold_percentile=96.0 
        )

        # --- Post-detection logging and processing ---
        # This debugging block can now be more informative after the core call
        if 'Reading_Time' in df_for_detection.columns:
            logger.info("Verification: 'Reading_Time' exists as a column in df_for_detection.")
        elif df_for_detection.index.name == 'Reading_Time' and isinstance(df_for_detection.index, pd.DatetimeIndex):
            logger.warning("Verification: 'Reading_Time' is index in df_for_detection (should be column after reset_index).")
        else:
            logger.error("Verification: 'Reading_Time' still not found appropriately in df_for_detection.")

        alarms_posted_count = 0
        if alarms:
            logger.info(f"Generated {len(alarms)} alarms. Posting...")
            # Ensure post_alarms_to_target is defined and imported
            alarms_posted_count = post_alarms_to_target(alarms, target_device_access_token) # Pass token if needed by this function
            logger.info(f"Posted {alarms_posted_count}/{len(alarms)} alarms.")
        else:
            logger.info("No alarms generated.")

        recommendations_posted_count = 0
        if alarms:
            logger.info("Generating and posting recommendations for alarms...")
            for alarm_json in alarms:
                # Ensure get_recommendation_rule_based_core is defined and imported
                recommendation = get_recommendation_rule_based_core(alarm_json)
                # Check for a meaningful recommendation before posting
                if recommendation and recommendation.get("Recommendations") not in [None, "No specific recommendation found. Please consult the system manual or an expert."]:
                    # Ensure post_single_recommendation is defined and imported
                    # The original code used `target_device_token` here. I'm using `target_device_access_token` from payload. Clarify which is correct.
                    if post_single_recommendation(recommendation, target_device_access_token):
                        recommendations_posted_count +=1
                    if len(alarms) > 1 : # Only sleep if there are multiple alarms to avoid unnecessary delay for single alarm
                        time.sleep(0.05) # Be cautious with sleeps in request handlers
            logger.info(f"Posted {recommendations_posted_count} recommendations.")

        return jsonify({
            "message": "Anomaly detection and recommendation processing complete.",
            "alarms_generated": len(alarms) if alarms else 0, # Handle if alarms is None
            "alarms_posted": alarms_posted_count,
            "recommendations_posted": recommendations_posted_count
        }), 200

    except Exception as e:
        logger.error(f"Error during anomaly detection/posting: {e}", exc_info=True)
        return jsonify({"error": f"Anomaly detection/posting failed: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("Anomaly Service starting on port 5004...")
    app.run(port=5004, debug=False, host='0.0.0.0')