import requests
import json
import pandas as pd
import os
import time

# --- Configuration ---
API_BASE_URL = "https://api.data.gov.in/resource/ee03643a-ee4c-48c2-ac30-9f2ff26ab722"
# IMPORTANT: Replace with your actual API key for higher limits if you generate one.
# For now, we'll use the sample key provided in the documentation.
API_KEY = "579b464db66ec23bdd000001439ecc26eba147cd72a20ba01b5e57c2"

# Folder structure
PROJECT_ROOT = "YourName_RollNumber_ML_CA1"
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "Datasets", "Raw")
CLEANED_DATA_DIR = os.path.join(PROJECT_ROOT, "Datasets", "Cleaned_Preprocessed")
OUTPUT_CONSOLIDATED_JSON_FILE = os.path.join(RAW_DATA_DIR, "mgnrega_data_raw.json")
OUTPUT_CLEANED_CSV_FILE = os.path.join(CLEANED_DATA_DIR, "mgnrega_data_cleaned.csv")
TEMP_BATCH_DIR = os.path.join(RAW_DATA_DIR, "batches") # Directory to store temporary batch files
SOURCE_CODE_DIR = os.path.join(PROJECT_ROOT, "Source_Code") # For your scripts
EDA_REPORT_DIR = os.path.join(PROJECT_ROOT, "EDA_Report") # For your report

# Pagination and API call settings
LIMIT_PER_REQUEST = 500  # A reasonable batch size. If issues, reduce it.
MAX_RETRIES = 5          # Number of times to retry a failed API request
RETRY_DELAY = 5          # Seconds to wait before retrying

# --- Functions ---

def setup_project_folders():
    """Sets up the required project folder structure."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    os.makedirs(TEMP_BATCH_DIR, exist_ok=True) # Create batch directory
    os.makedirs(SOURCE_CODE_DIR, exist_ok=True)
    os.makedirs(EDA_REPORT_DIR, exist_ok=True)
    print(f"Project folder structure created at: {PROJECT_ROOT}")

def fetch_mgnrega_batch(api_key, limit, offset, attempt=1):
    """
    Fetches a batch of MGNREGA data from the API with retry mechanism.
    """
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": limit,
        "offset": offset
    }
    try:
        print(f"Attempt {attempt}: Fetching batch from offset {offset} with limit {limit}...")
        response = requests.get(API_BASE_URL, params=params, timeout=30) # Add timeout
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error {e.response.status_code} for offset {offset}: {e}")
        if e.response.status_code == 403: # Forbidden - likely API key issue or rate limit
            print("Received 403 Forbidden. Check your API key or wait for rate limit reset.")
            return None
        elif attempt < MAX_RETRIES:
            print(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            return fetch_mgnrega_batch(api_key, limit, offset, attempt + 1)
        else:
            print(f"Max retries reached for offset {offset}. Skipping this batch.")
            return None
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error for offset {offset}: {e}")
        if attempt < MAX_RETRIES:
            print(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            return fetch_mgnrega_batch(api_key, limit, offset, attempt + 1)
        else:
            print(f"Max retries reached for offset {offset}. Skipping this batch.")
            return None
    except requests.exceptions.Timeout as e:
        print(f"Timeout Error for offset {offset}: {e}")
        if attempt < MAX_RETRIES:
            print(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            return fetch_mgnrega_batch(api_key, limit, offset, attempt + 1)
        else:
            print(f"Max retries reached for offset {offset}. Skipping this batch.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An unexpected request error occurred for offset {offset}: {e}")
        return None

def get_and_save_all_mgnrega_data_batched(api_key, limit_per_request, temp_batch_dir):
    """
    Fetches all MGNREGA data using pagination and saves each batch to a temporary JSON file.
    """
    offset = 0
    batch_count = 0
    total_records_fetched = 0

    print("\n--- Starting Batched Data Fetching ---")
    while True:
        batch_data = fetch_mgnrega_batch(api_key, limit=limit_per_request, offset=offset)

        if batch_data and 'records' in batch_data:
            records = batch_data['records']
            if not records:
                print("No more records found. All data batches fetched.")
                break # Exit loop if no records are returned

            # Save the current batch to a temporary JSON file
            batch_filename = os.path.join(temp_batch_dir, f"mgnrega_batch_{batch_count:04d}.json")
            try:
                with open(batch_filename, 'w', encoding='utf-8') as f:
                    json.dump(records, f, ensure_ascii=False, indent=4)
                print(f"Saved batch {batch_count} ({len(records)} records) to {batch_filename}")
            except IOError as e:
                print(f"Error saving batch {batch_count} to file {batch_filename}: {e}")
                # Decide how to handle this: skip, retry, or exit
                # For now, we'll continue, but this might need a more robust strategy
                pass

            total_records_fetched += len(records)
            offset += limit_per_request
            batch_count += 1
            time.sleep(1) # Be polite to the API, wait a bit between requests (e.g., 1 second)
        else:
            # If batch_data is None (due to severe error or max retries)
            # Or if 'records' key is missing in response
            print("Failed to fetch data or 'records' key missing in response. Stopping batch fetch.")
            break
    print(f"--- Finished Batched Data Fetching. Total records attempted to fetch: {total_records_fetched} ---")
    return total_records_fetched # Return count for confirmation

def consolidate_batches(temp_batch_dir, consolidated_json_path):
    """
    Reads all temporary batch JSON files and consolidates them into a single JSON file.
    """
    all_records = []
    print(f"\n--- Consolidating batches from {temp_batch_dir} ---")
    for filename in sorted(os.listdir(temp_batch_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(temp_batch_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    batch = json.load(f)
                    all_records.extend(batch)
                print(f"Loaded {len(batch)} records from {filename}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading or decoding JSON from {filepath}: {e}. Skipping file.")
    
    if all_records:
        try:
            with open(consolidated_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_records, f, ensure_ascii=False, indent=4)
            print(f"Consolidated {len(all_records)} records to {consolidated_json_path}")
        except IOError as e:
            print(f"Error saving consolidated JSON to {consolidated_json_path}: {e}")
            return False
        return True
    else:
        print("No records found in any batch files to consolidate.")
        return False

def convert_json_to_cleaned_csv(consolidated_json_path, cleaned_csv_path):
    """
    Loads consolidated JSON data, performs basic cleaning, and saves to CSV.
    """
    if not os.path.exists(consolidated_json_path):
        print(f"Consolidated JSON file not found at: {consolidated_json_path}. Cannot convert to CSV.")
        return

    print(f"\n--- Converting consolidated JSON to cleaned CSV ---")
    try:
        with open(consolidated_json_path, 'r', encoding='utf-8') as f:
            data_records = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading consolidated JSON from {consolidated_json_path}: {e}. Cannot convert to CSV.")
        return

    if not data_records:
        print("No data in consolidated JSON file to convert.")
        return

    df = pd.DataFrame(data_records)

    # Basic cleaning/preprocessing (as discussed, more will go into your ETL script)
    numeric_cols = ['Total No. of JobCards issued', 'Total No. of Workers']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0) # Fill NaN with 0 for numerical columns

    # Rename columns for easier use
    df.rename(columns={
        'SNo.': 'Serial_No',
        'state_name': 'State',
        'district_name': 'District',
        'Total No. of JobCards issued': 'Total_JobCards_Issued',
        'Total No. of Workers': 'Total_Workers'
    }, inplace=True)

    try:
        df.to_csv(cleaned_csv_path, index=False, encoding='utf-8')
        print(f"Cleaned and preprocessed data saved to: {cleaned_csv_path}")
        print("\nInitial Data Head (first 5 rows):")
        print(df.head())
        print("\nInitial Data Info:")
        df.info()
    except IOError as e:
        print(f"Error saving cleaned CSV to {cleaned_csv_path}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    setup_project_folders()

    # Step 1: Fetch data in batches and save temporary files
    total_fetched = get_and_save_all_mgnrega_data_batched(API_KEY, LIMIT_PER_REQUEST, TEMP_BATCH_DIR)

    if total_fetched > 0:
        # Step 2: Consolidate all temporary batch files into one raw JSON
        if consolidate_batches(TEMP_BATCH_DIR, OUTPUT_CONSOLIDATED_JSON_FILE):
            # Step 3: Convert the consolidated raw JSON to a cleaned CSV
            convert_json_to_cleaned_csv(OUTPUT_CONSOLIDATED_JSON_FILE, OUTPUT_CLEANED_CSV_FILE)
        else:
            print("Consolidation failed. CSV conversion skipped.")
    else:
        print("No data was fetched in batches. Check API key, connection, or parameters.")

    # Optional: Clean up temporary batch files after successful consolidation
    # For a real project, you might keep them for debugging or re-runs.
    # For this assignment, it's fine to leave them or add a cleanup step.
    # if total_fetched > 0 and os.path.exists(OUTPUT_CONSOLIDATED_JSON_FILE):
    #     print("\nCleaning up temporary batch files...")
    #     for filename in os.listdir(TEMP_BATCH_DIR):
    #         filepath = os.path.join(TEMP_BATCH_DIR, filename)
    #         try:
    #             os.remove(filepath)
    #         except OSError as e:
    #             print(f"Error removing {filepath}: {e}")
    #     os.rmdir(TEMP_BATCH_DIR)
    #     print("Temporary batch files removed.")