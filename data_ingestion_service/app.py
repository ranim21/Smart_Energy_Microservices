from flask import Flask, request, jsonify
import requests
import datetime
import time
import json
from collections import defaultdict
import pandas as pd 
import numpy
import pandas
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config_common as config

app = Flask(__name__)
logger = config.get_logger('data_ingestion_service')

def get_timeseries(entity_id, entity_type, keys, start_ts, end_ts,
                   base_url=config.SOURCE_URL, api_prefix=config.SOURCE_API_PREFIX,
                   headers=config.SOURCE_HEADERS, limit=20000, 
                   timeout=config.API_TIMEOUT):
    url = f"{base_url}{api_prefix}/plugins/telemetry/{entity_type}/{entity_id}/values/timeseries"
    params = {
        "keys": ",".join(keys),
        "startTs": start_ts,
        "endTs": end_ts,
        "limit": limit,
        "agg": "NONE",
        "useStrictDataTypes": "false" 
    }
    logger.info(f"Fetching telemetry for {entity_type} {entity_id} from {base_url} (keys: {keys})")
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        config.log_http_request_detail(logger, url, 'GET', params=params, response=resp, headers=headers)
        resp.raise_for_status()
        if not resp.text.strip(): return {}
        data = resp.json()
        if not any(data.get(key) for key in keys): return {}
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed fetching telemetry for {entity_id} from {base_url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Error response: {e.response.text}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error fetching telemetry for {entity_id}: {e}", exc_info=True)
        return {}

def format_fetched_telemetry_for_post(telemetry_json, keys_to_post):
    data_by_ts = defaultdict(dict)
    if not isinstance(telemetry_json, dict): return []
    for key in keys_to_post:
        if key in telemetry_json and telemetry_json[key]:
            for datapoint in telemetry_json[key]:
                if isinstance(datapoint, dict) and "ts" in datapoint and "value" in datapoint:
                    ts = datapoint["ts"]
                    raw_value = datapoint["value"]
                    try:
                        if isinstance(raw_value, str) and raw_value.lower() in ['true', 'false']:
                            value_to_store = raw_value.lower() == 'true'
                        elif raw_value is None: value_to_store = None
                        else: value_to_store = float(raw_value)
                    except (ValueError, TypeError):
                        value_to_store = str(raw_value)
                    if value_to_store is not None: data_by_ts[ts][key] = value_to_store
    return [{"ts": ts, "values": values_dict} for ts, values_dict in sorted(data_by_ts.items()) if values_dict]


def post_telemetry_item_to_target(telemetry_item, url=config.TARGET_DEVICE_TELEMETRY_URL, timeout=config.API_TIMEOUT):
    if not telemetry_item or "ts" not in telemetry_item or "values" not in telemetry_item:
        logger.warning(f"Invalid telemetry item for posting: {telemetry_item}")
        return {}
    try:
        resp = requests.post(url, json=telemetry_item, timeout=timeout)
        config.log_http_request_detail(logger, url, 'POST', json_payload=telemetry_item, response=resp)
        resp.raise_for_status()
        logger.info(f"Successfully posted telemetry to {url.split('/api')[0]}. Status: {resp.status_code}")
        try: return resp.json()
        except requests.exceptions.JSONDecodeError: return {"status": "success", "body": resp.text}
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to post telemetry to {url.split('/api')[0]}: {e}")
        if hasattr(e, 'response') and e.response is not None: logger.error(f"Failed POST response: {e.response.text}")
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error posting telemetry: {e}", exc_info=True)
        return {"status": "error", "message": f"Unexpected error: {e}"}

# --- Flask Endpoints ---
@app.route('/process-day', methods=['POST'])
def process_day_pipeline():
    data = request.json
    date_str = data.get('date_str')  # type: ignore
    entity_id = data.get('entity_id')  # type: ignore
    entity_type = data.get('entity_type', "DEVICE")   # type: ignore
    originator_id_for_alarms = entity_id 
    if not all([date_str, entity_id]):
        return jsonify({"error": "Missing date_str or entity_id"}), 400

    logger.info(f"--- PIPELINE START: Processing Day: {date_str} for {entity_type} {entity_id} ---")
    actions_summary = {"day": date_str, "entity_id": entity_id, "steps": []}

    try:
        current_day = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        next_day = current_day + datetime.timedelta(days=1)
        start_ts_ms = int(current_day.timestamp() * 1000)
        end_ts_ms = int(next_day.timestamp() * 1000)
    except ValueError:
        logger.error(f"Invalid date format: {date_str}")
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    # 1. Fetch Telemetry
    logger.info(f"Step 1: Fetching telemetry for day {date_str}")
    # Fetch all keys that might be needed downstream (preprocessing, anomaly detection)
    telemetry_json = get_timeseries(entity_id, entity_type, config.TELEMETRY_KEYS, start_ts_ms, end_ts_ms)
    if not telemetry_json or not any(telemetry_json.values()):
        msg = f"No telemetry data found for {entity_id} on {date_str}."
        logger.warning(msg)
        actions_summary["steps"].append({"step": "fetch_telemetry", "status": "no_data", "message": msg})
        return jsonify(actions_summary), 200 # Or 404 if preferred

    actions_summary["steps"].append({"step": "fetch_telemetry", "status": "success", "keys_found": list(telemetry_json.keys())})

    # fetched_telemetry_to_post = format_fetched_telemetry_for_post(telemetry_json, config.TELEMETRY_KEYS)
    # if fetched_telemetry_to_post:
    #     logger.info(f"Step 2: Mirroring {len(fetched_telemetry_to_post)} telemetry items to target.")
    #     for item in fetched_telemetry_to_post:
    #         post_telemetry_item_to_target(item) # Errors logged within function
    #         time.sleep(0.05)
    #     actions_summary["steps"].append({"step": "mirror_telemetry", "status": "attempted", "items": len(fetched_telemetry_to_post)})


    # 3. Call Preprocessing Service
    logger.info(f"Step 3: Calling Preprocessing Service for day {date_str}")
    processed_data_json_str = None
    try:
        preprocess_payload = {"telemetry_json": telemetry_json, "keys_to_process": ["Ea"]} # Focus on 'Ea' for forecasting
        resp_preprocess = requests.post(f"{config.PREPROCESSING_SERVICE_URL}/preprocess", json=preprocess_payload, timeout=config.API_TIMEOUT*2) # Longer timeout for processing
        resp_preprocess.raise_for_status()
        processed_data_json_str = resp_preprocess.json().get("processed_data_json")
        if not processed_data_json_str: raise ValueError("Preprocessing returned no data.")
        actions_summary["steps"].append({"step": "preprocess_data", "status": "success"})
        temp_df = pd.read_json(processed_data_json_str, orient='split')
        logger.debug(f"Preprocessed data head:\n{temp_df.head()}")

    except Exception as e:
        msg = f"Error calling or processing data from Preprocessing Service: {e}"
        logger.error(msg, exc_info=True)
        actions_summary["steps"].append({"step": "preprocess_data", "status": "failed", "error": str(e)})
        return jsonify(actions_summary), 500

    # 4. Call Forecasting Service
    # logger.info(f"Step 4: Calling Forecasting Service for day {date_str}")
    # try:
    #     forecast_payload = {
    #         "processed_data_json": processed_data_json_str,
    #         "entity_id": entity_id, 
    #         "target_device_access_token": config.TARGET_DEVICE_ACCESS_TOKEN_FOR_POSTING # Pass token for posting predictions
    #     }
    #     resp_forecast = requests.post(f"{config.FORECASTING_SERVICE_URL}/forecast-and-post", json=forecast_payload, timeout=config.API_TIMEOUT*3) # Model prediction can be long
    #     resp_forecast.raise_for_status()
    #     forecast_result = resp_forecast.json()
    #     actions_summary["steps"].append({"step": "forecast_data", "status": "success", "details": forecast_result})
    # except Exception as e:
    #     msg = f"Error calling Forecasting Service: {e}"
    #     logger.error(msg, exc_info=True)
    #     actions_summary["steps"].append({"step": "forecast_data", "status": "failed", "error": str(e)})

    # 5. Call Anomaly Detection Service
    logger.info(f"Step 5: Calling Anomaly Detection Service for day {date_str}")
    try:
        anomaly_payload = {
            "processed_data_json": processed_data_json_str,
            "originator_id": originator_id_for_alarms, # Actual ID of the device
            "target_device_access_token": config.TARGET_DEVICE_ACCESS_TOKEN_FOR_POSTING # For posting recommendations
        }
        resp_anomaly = requests.post(f"{config.ANOMALY_SERVICE_URL}/detect-alarms-recommendations", json=anomaly_payload, timeout=config.API_TIMEOUT*2)
        resp_anomaly.raise_for_status()
        anomaly_result = resp_anomaly.json()
        actions_summary["steps"].append({"step": "detect_anomalies", "status": "success", "details": anomaly_result})
    except Exception as e:
        msg = f"Error calling Anomaly Detection Service: {e}"
        logger.error(msg, exc_info=True)
        actions_summary["steps"].append({"step": "detect_anomalies", "status": "failed", "error": str(e)})

    logger.info(f"--- PIPELINE END: Finished processing for day {date_str} ---")
    return jsonify(actions_summary), 200


if __name__ == '__main__':
    logger.info("Data Ingestion Service starting on port 5001...")
    app.run(port=5001, debug=False, host='0.0.0.0') # Use debug=False for production-like behavior