import requests
import datetime
import time
import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
try:
    import config_common as config
except ImportError:
    print("Error: Could not import config_common. Ensure it's in the same directory or sys.path is correct.")
    sys.exit(1)


START_DATE_STR = "2025-01-01"  
END_DATE_STR = "2025-01-02"    
DEVICE_NAME_FOR_LOGGING = "Canalis D1,3" 
ENTITY_ID = "7114c6e0-88e8-11ec-b8c5-0b1d9317b9a9" 
ENTITY_TYPE = "DEVICE"

orchestrator_logger = config.get_logger('orchestrator_script')

def run_daily_pipeline(date_str, entity_id, entity_type):
    orchestrator_logger.info(f"Orchestrator: Triggering pipeline for date: {date_str}, Entity ID: {entity_id}")
    
    process_day_url = f"{config.DATA_INGESTION_SERVICE_URL}/process-day"
    payload = {
        "date_str": date_str,
        "entity_id": entity_id,
        "entity_type": entity_type
    }

    try:
        response = requests.post(process_day_url, json=payload, timeout=config.API_TIMEOUT * 5) # Long timeout for full pipeline
        response.raise_for_status()
        orchestrator_logger.info(f"Pipeline for {date_str} triggered successfully. Response: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        orchestrator_logger.error(f"Failed to trigger pipeline for {date_str}. Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            orchestrator_logger.error(f"Error response from service: {e.response.text}")
        return {"error": str(e)}
    except Exception as e:
        orchestrator_logger.error(f"An unexpected error occurred during pipeline trigger for {date_str}: {e}", exc_info=True)
        return {"error": f"Unexpected error: {str(e)}"}


if __name__ == "__main__":
    orchestrator_logger.info("--- Orchestrator Script Starting ---")
    
    if "YOUR_SOURCE_BEARER_TOKEN_HERE" in config.SOURCE_TOKEN or \
       "YOUR_TARGET_BEARER_TOKEN_HERE" in config.TARGET_TOKEN or \
       config.TARGET_DEVICE_ACCESS_TOKEN_FOR_POSTING == "YOUR_TARGET_DEVICE_ACCESS_TOKEN_HERE":
        orchestrator_logger.critical("CRITICAL: Placeholder tokens found in config_common.py. Please update them!")
        print("\nCRITICAL: Placeholder tokens found in config_common.py. Please update them before running!\n")
        sys.exit(1)


    try:
        start_date = datetime.datetime.strptime(START_DATE_STR, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(END_DATE_STR, "%Y-%m-%d")
        orchestrator_logger.info(f"Processing data from {START_DATE_STR} up to (but not including) {END_DATE_STR} for {DEVICE_NAME_FOR_LOGGING} (ID: {ENTITY_ID})")
    except ValueError:
        orchestrator_logger.error(f"Invalid date format in START_DATE_STR or END_DATE_STR. Use YYYY-MM-DD.")
        sys.exit(1)

    current_day = start_date
    while current_day < end_date:
        day_str = current_day.strftime('%Y-%m-%d')
        orchestrator_logger.info(f"--- Orchestrating for Day: {day_str} ---")
        
        result = run_daily_pipeline(day_str, ENTITY_ID, ENTITY_TYPE)
        orchestrator_logger.info(f"Result for day {day_str}: {result}")
        
        current_day += datetime.timedelta(days=1)
        time.sleep(1) 

    orchestrator_logger.info(f"--- Orchestrator Script Finished for Entity: {DEVICE_NAME_FOR_LOGGING} (ID: {ENTITY_ID}) ---")