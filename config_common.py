import os
import logging
import time
import requests 
SOURCE_URL = os.environ.get("SOURCE_URL", "https://things.sofia-networks.com")
SOURCE_API_PREFIX = "/api"
SOURCE_TOKEN = os.environ.get("SOURCE_TOKEN", "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJTdGFnZUBTdGFnZS5jb20iLCJzY29wZXMiOlsiQ1VTVE9NRVJfVVNFUiJdLCJ1c2VySWQiOiIyNTIzYzE3MC1lODUxLTExZWYtYTVjYy0yOWFiMDViY2IwNzMiLCJmaXJzdE5hbWUiOiJTdGFnZSIsImxhc3ROYW1lIjoiU3RhZ2UiLCJlbmFibGVkIjp0cnVlLCJpc1B1YmxpYyI6ZmFsc2UsInRlbmFudElkIjoiYTBiMjUzYzAtMDEyMi0xMWVjLTg4MmMtY2JlYjdjMmE2NWFlIiwiY3VzdG9tZXJJZCI6ImVlMjkxYzUwLTg4ZTYtMTFlYy1iOGM1LTBiMWQ5MzE3YjlhOSIsImlzcyI6InRoaW5nc2JvYXJkLmlvIiwiaWF0IjoxNzQ4MzM1MzgyLCJleHAiOjE3NDgzNDQzODJ9.DSQblXVmMI2Mf0aKaojsVO-WBuUN7lJsqVsPtu45HFH49-NguVU3gR0HxQW7COWX50hH2YhRX_dEjG6HBgREQg")
SOURCE_HEADERS = {
   'X-Authorization': f"Bearer {SOURCE_TOKEN}",
   'Content-Type': "application/json",
   'Accept': 'application/json'
}

# --- Target ThingsBoard Configuration ---
TARGET_URL = os.environ.get("TARGET_URL", "http://localhost:8080")
TARGET_API_PREFIX = "/api"
TARGET_TOKEN = os.environ.get("TARGET_TOKEN", "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0ZW5hbnRAdGhpbmdzYm9hcmQub3JnIiwidXNlcklkIjoiZDFmZjViMzAtZTE5My0xMWVmLTliMjItYzlkMTY5MzdjMDMzIiwic2NvcGVzIjpbIlRFTkFOVF9BRE1JTiJdLCJzZXNzaW9uSWQiOiI4NjFlMWM5Mi02MDgwLTQ2OGYtOWYwYS1hZmU1ZTQ2MWMxZDMiLCJleHAiOjE3NDgzNDI4NjIsImlzcyI6InRoaW5nc2JvYXJkLmlvIiwiaWF0IjoxNzQ4MzMzODYyLCJlbmFibGVkIjp0cnVlLCJpc1B1YmxpYyI6ZmFsc2UsInRlbmFudElkIjoiZDE5NDhiNzAtZTE5My0xMWVmLTliMjItYzlkMTY5MzdjMDMzIiwiY3VzdG9tZXJJZCI6IjEzODE0MDAwLTFkZDItMTFiMi04MDgwLTgwODA4MDgwODA4MCJ9.dD55EEWx_r9T4HEmfOuKHHaI6o_TeB0Wso4cbnyv3dTf2BitFEz7yVqMtvUoLssLbfp6_DaS5O1PaBcodIUr0w")
TARGET_HEADERS = {
   'X-Authorization': f"Bearer {TARGET_TOKEN}",
   'Content-Type': "application/json",
   'Accept': 'application/json'
}
TARGET_DEVICE_ACCESS_TOKEN_FOR_POSTING = os.environ.get("TARGET_DEVICE_ACCESS_TOKEN_FOR_POSTING", "vZhSgD0h19Hjdy4MwrW4") 
TARGET_DEVICE_TELEMETRY_URL = f"{TARGET_URL}/api/v1/{TARGET_DEVICE_ACCESS_TOKEN_FOR_POSTING}/telemetry"
TARGET_ALARM_URL = f"{TARGET_URL}{TARGET_API_PREFIX}/alarm"


DEFAULT_KMEANS_FEATURES = ["Ea"]
DEFAULT_TIME_COL = 'Reading_Time'
TELEMETRY_KEYS =  ["Reading_Time",
    "U12","U23","U31",
    "V1","V2","V3",
    "I1","I2","I3","ID","In",
    "Ea","Er","F",
    "THD_In","THD_U12","THD_U23","THD_U31",
    "THD_V1","THD_V2","THD_V3",
    "THD_I1","THD_I2","THD_I3",
    "P1","P2","P3",
    "PF1","PF2","PF3","PFTot","PTot",
    "Q1","Q2","Q3","QTot",
    "S1","S2","S3","STot",
    "phases_shift_I","phases_shift_P","phases_shift_V"] 

API_TIMEOUT = 60 
MODEL_PATH_REL = "Forecasting_files/2nd_try_seq2seq_LSTM_best.keras"
SCALER_PATH_REL = "Forecasting_files/my_scaler.joblib"
TIME_STEPS = 288 
EXPECTED_FORECAST_HORIZON = 288 

# --- Logging Configuration ---
LOG_FOLDER_BASE = os.environ.get("LOG_FOLDER_BASE", os.path.join(os.getcwd(), "service_logs"))
os.makedirs(LOG_FOLDER_BASE, exist_ok=True)

def get_logger(service_name, log_folder_base=LOG_FOLDER_BASE):
    logger = logging.getLogger(service_name)
    if not logger.handlers: 
        logger.setLevel(logging.DEBUG)
        log_dir = os.path.join(log_folder_base, service_name)
        os.makedirs(log_dir, exist_ok=True)
        
        current_time_str = time.strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(log_dir, f'{service_name}_{current_time_str}.log'))
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO) 
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def log_http_request_detail(logger, url, method, params=None, data=None, json_payload=None, response=None, headers=None):
    log_msg = f"Request - Method: {method}, URL: {url}"
    if params:
        try:
            url_with_params = f"{url}?{params if isinstance(params, str) else requests}" 
            log_msg = f"Request - Method: {method}, URL: {url_with_params}"
        except Exception: 
             log_msg += f", Params: {str(params)[:100]}..."


    logged_headers = {}
    if headers:
        for k, v in headers.items():
            if k.lower() == 'x-authorization' and isinstance(v, str) and len(v) > 10:
                logged_headers[k] = v[:10] + "****" + v[-4:]
            else:
                logged_headers[k] = v
        log_msg += f", Headers: {logged_headers}"


    if json_payload:
        log_msg += f", JSON Payload Snippet: {str(json_payload)[:200]}..."
    elif data:
        log_msg += f", Form Data Snippet: {str(data)[:200]}..."

    if response is not None:
        status_code = response.status_code
        try:
            response_text_snippet = response.text[:200] + '...' if len(response.text) > 200 else response.text
        except Exception:
            response_text_snippet = "[Could not decode response text]"
        log_msg += f" | Response - Status: {status_code}, Body Snippet: {response_text_snippet}"
    logger.debug(log_msg)


DATA_INGESTION_SERVICE_URL = os.environ.get("DATA_INGESTION_SERVICE_URL", "http://localhost:5001")
PREPROCESSING_SERVICE_URL = os.environ.get("PREPROCESSING_SERVICE_URL", "http://localhost:5002")
FORECASTING_SERVICE_URL = os.environ.get("FORECASTING_SERVICE_URL", "http://localhost:5003")
ANOMALY_SERVICE_URL = os.environ.get("ANOMALY_SERVICE_URL", "http://localhost:5004")