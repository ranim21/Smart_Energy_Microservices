# flask_thingsboard_pipeline/preprocessing_service/Preprocessing.py
import pandas as pd
import numpy as np # For np.number, np.nan, np.inf
import logging
from sklearn.impute import KNNImputer

# Get a logger instance
logger = logging.getLogger("PreprocessorClass") # Use a distinct name
# Configure logger if not already configured (e.g., by a root logger)
if not logger.handlers and not logging.getLogger().handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
# If you have a global logging setup (e.g., in app.py), 
# you might not need to add handlers here, just get the logger.
# To be safe and avoid duplicate logs if this module is reloaded:
elif not logger.handlers: # Add handler only if this specific logger has no handlers
    logger.info(f"Logger '{logger.name}' already configured by parent or has handlers. Not re-adding StreamHandler.")


class Preprocessor:
    def __init__(self, df: pd.DataFrame, time_column_name: str = 'Reading_Time', numeric_cols_to_convert = None):
        logger.info(f"Preprocessor initialized. Initial input type: {type(df)}")
        
        if not isinstance(df, pd.DataFrame):
            logger.error("Preprocessor received non-DataFrame input. Initializing with an empty DataFrame.")
            self.df = pd.DataFrame()
            return

        if df.empty:
            logger.info("Preprocessor initialized with an empty DataFrame.")
            self.df = pd.DataFrame() # Standardize to an empty DF
            return
            
        self.df = df.copy() # Work on a copy
        logger.info(f"Initial DataFrame shape: {self.df.shape}, Columns: {self.df.columns.tolist()}")

        # 1. Handle Time Index
        if time_column_name in self.df.columns:
            try:
                self.df[time_column_name] = pd.to_datetime(self.df[time_column_name], errors='coerce')
                if self.df[time_column_name].isnull().any():
                    original_rows = len(self.df)
                    self.df.dropna(subset=[time_column_name], inplace=True)
                    logger.warning(f"Preprocessor: Coerced some '{time_column_name}' to NaT. Dropped {original_rows - len(self.df)} rows.")
                
                if not self.df.empty:
                    self.df = self.df.set_index(time_column_name)
                    logger.info(f"Preprocessor: '{time_column_name}' column converted to DatetimeIndex.")
                else:
                    logger.warning(f"Preprocessor: DataFrame became empty after '{time_column_name}' NaT drop. Index not set.")
            except Exception as e:
                logger.error(f"Preprocessor: Error setting '{time_column_name}' as DatetimeIndex: {e}")
        elif isinstance(self.df.index, pd.DatetimeIndex):
            logger.info("Preprocessor: DataFrame already has a DatetimeIndex.")
            if self.df.index.name != time_column_name and time_column_name:
                logger.info(f"Renaming existing DatetimeIndex to '{time_column_name}'.")
                self.df.index.name = time_column_name
        else: # Try to convert existing index if it's not time_column_name column or DatetimeIndex
            logger.info(f"Preprocessor: '{time_column_name}' not in columns and index is not DatetimeIndex. Attempting to convert current index.")
            try:
                original_index_name = self.df.index.name
                temp_index_col = original_index_name if original_index_name else 'temp_time_index'
                self.df[temp_index_col] = pd.to_datetime(self.df.index, errors='coerce')

                if self.df[temp_index_col].isnull().any():
                    original_rows = len(self.df)
                    self.df.dropna(subset=[temp_index_col], inplace=True)
                    logger.warning(f"Preprocessor: Coerced some existing index values to NaT. Dropped {original_rows - len(self.df)} rows.")

                if not self.df.empty:
                    self.df = self.df.set_index(temp_index_col)
                    if temp_index_col == 'temp_time_index' and time_column_name:
                        self.df.index.name = time_column_name
                    logger.info(f"Preprocessor: Existing index converted to DatetimeIndex and named '{self.df.index.name}'.")
                else:
                    logger.warning("Preprocessor: DataFrame became empty after index NaT drop.")
            except Exception as e:
                logger.error(f"Preprocessor: Could not convert existing index to DatetimeIndex: {e}")
        
        # 2. Convert specified columns to numeric
        if numeric_cols_to_convert is None:
            numeric_cols_to_convert = ['Ea'] 
            
        logger.info(f"DEBUG __init__: Entering numeric conversion loop. self.df.columns: {self.df.columns.tolist()}")
        for col in numeric_cols_to_convert:
            logger.info(f"DEBUG __init__: Processing column '{col}' for numeric conversion.")
            if col in self.df.columns:
                logger.info(f"DEBUG __init__: Column '{col}' found. Current values (first 5):\n{self.df[col].head().to_string()}")
                logger.info(f"DEBUG __init__: Dtype of '{col}' before pd.to_numeric: {self.df[col].dtype}")
                
                # Store original for comparison (optional, but good for debugging)
                # original_series = self.df[col].copy() 

                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
                logger.info(f"DEBUG __init__: Dtype of '{col}' AFTER pd.to_numeric: {self.df[col].dtype}")
                logger.info(f"DEBUG __init__: Values of '{col}' AFTER pd.to_numeric (first 5):\n{self.df[col].head().to_string()}")
                logger.info(f"DEBUG __init__: NaNs in '{col}' AFTER pd.to_numeric: {self.df[col].isnull().sum()}")

                if self.df[col].isnull().all() and col in ['Ea']: # Specific check for 'Ea'
                    logger.error(f"CRITICAL DEBUG __init__: Column '{col}' became ALL NaNs immediately after pd.to_numeric!")
                
                # Your existing logging for success/coercion
                if not pd.api.types.is_numeric_dtype(self.df[col].dtype): # Should not happen if pd.to_numeric ran
                     logger.warning(f"Preprocessor: Column '{col}' is NOT numeric even after pd.to_numeric.")
                # Check if coercion happened
                # This logic might need refinement based on original_series if you use it
                # For now, we rely on the "successfully converted" vs actual NaN check
                logger.info(f"Preprocessor: Column '{col}' successfully converted to numeric (or was already).")

            else:
                logger.warning(f"Preprocessor __init__: Column '{col}' for numeric conversion NOT FOUND in self.df.columns ({self.df.columns.tolist()}) after index setting.")

        # The existing logs after the loop:
        logger.info(f"DEBUG __init__: self.df AFTER numeric conversion attempts. Head:\n{self.df.head().to_string()}")
        logger.info(f"DEBUG __init__: self.df dtypes AFTER numeric conversion attempts:\n{self.df.dtypes}")
        logger.info(f"DEBUG __init__: self.df NaNs AFTER numeric conversion attempts:\n{self.df.isnull().sum().to_string()}")

        logger.info(f"Preprocessor: DataFrame shape after __init__ type conversions and index setting: {self.df.shape if not self.df.empty else '(0,0)'}")

    def get_dataframe(self):
        return self.df.copy() # Return a copy to prevent external modification

    def generate_anomalous_time_series_data(self, num_samples=1500, start_time_str="2025-01-02 00:00:00", anomaly_columns=None):
        if anomaly_columns is None:
            anomaly_columns = ["Ea"] # Default anomaly column
        logger.info(f"Generating {num_samples} samples of anomalous data for columns {anomaly_columns} starting from {start_time_str}.")
        
        time_index_name = self.df.index.name if self.df.index.name else 'Reading_Time'
        timestamps = pd.date_range(start=start_time_str, periods=num_samples, freq='5T', name=time_index_name)
        
        generated_data = pd.DataFrame(np.random.randn(num_samples, len(anomaly_columns)), columns=anomaly_columns, index=timestamps)

        anomaly_fraction = 0.05
        num_anomalies = int(num_samples * anomaly_fraction)
        # Ensure we don't try to pick more anomalies than available data points if num_samples is small
        if num_anomalies > 0 and num_samples > 0:
            anomaly_indices = np.random.choice(generated_data.index, size=min(num_anomalies, num_samples), replace=False)

            for ts_idx in anomaly_indices:
                column_to_affect = np.random.choice(anomaly_columns)
                anomaly_type = np.random.choice([np.nan, np.inf, -np.inf, 9999, -9999,
                                                 np.random.uniform(10000, 20000), 
                                                 np.random.uniform(-20000, -10000)])
                generated_data.loc[ts_idx, column_to_affect] = anomaly_type
            logger.info(f"Introduced anomalies to {len(anomaly_indices)} data points in generated data.")
        else:
            logger.info("No anomalies introduced (num_anomalies or num_samples is zero).")


        if not self.df.empty:
            logger.info(f"Concatenating generated anomalous data with existing DataFrame. Existing shape: {self.df.shape}, Generated shape: {generated_data.shape}")
            self.df = pd.concat([self.df, generated_data])
            # Handle duplicate indices by keeping the first occurrence
            self.df = self.df[~self.df.index.duplicated(keep='first')]
            logger.info(f"DataFrame shape after concatenation and duplicate index removal: {self.df.shape}")
        else:
            logger.info("Existing DataFrame is empty. Using generated anomalous data.")
            self.df = generated_data
        
        self.df.sort_index(inplace=True)
        return self.df

    def resample_data(self, frequency='5min', aggregation_method='median'):
        if self.df.empty:
            logger.warning("Preprocessor - resample_data: DataFrame is empty. Cannot resample.")
            return self.df
            
        if not isinstance(self.df.index, pd.DatetimeIndex):
            logger.error("Preprocessor - resample_data: DataFrame index is not DatetimeIndex. Resampling skipped. Call __init__ or set index appropriately.")
            return self.df 
        
        logger.info(f"Resampling data to {frequency} frequency using {aggregation_method}. Current shape: {self.df.shape}")
        logger.info(f"Resampling data to {frequency} frequency using {aggregation_method}. Current shape: {self.df.shape}")
        logger.info(f"DEBUG: self.df BEFORE resampling. Head:\n{self.df.head().to_string()}") # Use INFO for visibility
        logger.info(f"DEBUG: self.df dtypes BEFORE resampling:\n{self.df.dtypes}")
        logger.info(f"DEBUG: self.df NaNs per column BEFORE resampling:\n{self.df.isnull().sum().to_string()}")
        
        numeric_df = self.df.select_dtypes(include=np.number)
        non_numeric_df = self.df.select_dtypes(exclude=np.number)

        if numeric_df.empty:
            logger.warning("Preprocessor - resample_data: No numeric columns to resample. Returning original DataFrame structure.")
            return self.df # self.df still contains original data (possibly non-numeric)
        logger.info(f"DEBUG: numeric_df for resampling. Head:\n{numeric_df.head().to_string()}") # Use INFO
        resampled_numeric_df = numeric_df.resample(frequency).agg(aggregation_method)  
        logger.info(f"DEBUG: resampled_numeric_df AFTER resampling. Head:\n{resampled_numeric_df.head().to_string()}") # Use INFO
        logger.info(f"DEBUG: resampled_numeric_df dtypes AFTER resampling:\n{resampled_numeric_df.dtypes}")
        logger.info(f"DEBUG: resampled_numeric_df NaNs AFTER resampling:\n{resampled_numeric_df.isnull().sum().to_string()}")
            
        resampled_numeric_df = numeric_df.resample(frequency).agg(aggregation_method)  
        
        # If there were non-numeric columns, decide how to handle them.
        # For simplicity, this version will only keep the resampled numeric data.
        # If non-numeric data needs to be preserved and aligned, it's more complex.
        if not non_numeric_df.empty:
            logger.warning(f"Preprocessor - resample_data: Non-numeric columns {non_numeric_df.columns.tolist()} were present. They are not included in the resampled output of this version.")

        self.df = resampled_numeric_df 
        logger.info(f"Data resampled. Shape is now {self.df.shape}. Index name: '{self.df.index.name}'.")
        return self.df

    def handle_missing_values(self, n_neighbors=3):
        if self.df.empty:
            logger.warning("Preprocessor - handle_missing_values: DataFrame is empty. Cannot impute.")
            return self.df

        logger.info(f"Handling missing values using KNN (n_neighbors={n_neighbors}). Current shape: {self.df.shape}")
        
        # Replace Inf values before selecting numeric types, as Inf is numeric
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        features_to_impute_df = self.df.select_dtypes(include=np.number)

        if features_to_impute_df.empty:
            logger.warning("Preprocessor - handle_missing_values: No numeric features to impute.")
            return self.df # self.df might contain non-numeric data or was already empty

        logger.debug(f"DEBUG: features_to_impute_df going into imputer. Head:\n{features_to_impute_df.head().to_string()}")
        logger.debug(f"DEBUG: features_to_impute_df dtypes:\n{features_to_impute_df.dtypes}")
        logger.debug(f"DEBUG: features_to_impute_df NaNs per column:\n{features_to_impute_df.isnull().sum().to_string()}")
       
        original_columns = features_to_impute_df.columns
        original_index = features_to_impute_df.index # Use index from the actual df being imputed

        if features_to_impute_df.isnull().all().all(): # Check if ALL values in ALL numeric columns are NaN
             logger.warning("All numeric features selected for imputation are entirely NaN. KNNImputer cannot run. Skipping imputation.")
             return self.df # Return df with NaNs

        imputer = KNNImputer(n_neighbors=min(n_neighbors, len(features_to_impute_df) -1 ) if len(features_to_impute_df) >1 else 1) # Ensure n_neighbors <= n_samples -1

        try:
            imputed_values_array = imputer.fit_transform(features_to_impute_df)
            logger.debug(f"DEBUG: Shape of imputed_values_array after imputer: {imputed_values_array.shape}")
            logger.debug(f"DEBUG: original_columns for imputed_df_part: {original_columns.tolist()}")

            # Determine columns for the resulting DataFrame
            # Standard KNNImputer does not change column order or count
            if imputed_values_array.shape[1] == len(original_columns):
                imputed_df_cols = original_columns
            elif imputed_values_array.shape[1] == 0 and len(original_columns) > 0:
                # This case was causing the original error. KNNImputer shouldn't do this unless 
                # features_to_impute_df was already 0 columns, or a custom imputer.
                logger.warning(f"Imputer returned 0 columns from {len(original_columns)} input columns ({original_columns.tolist()}). "
                               "This implies columns were dropped by the imputer. Check imputer behavior.")
                imputed_df_cols = []
            else: # Unexpected change in column count
                logger.error(f"Imputer unexpectedly changed column count from {len(original_columns)} to {imputed_values_array.shape[1]}. "
                               "Using generic column names. This may lead to misaligned data!")
                imputed_df_cols = [f"imputed_col_{i}" for i in range(imputed_values_array.shape[1])]
            
            imputed_df_part = pd.DataFrame(imputed_values_array, columns=imputed_df_cols, index=original_index)

            # Update the numeric columns in self.df with imputed data
            if not imputed_df_part.empty:
                for col in imputed_df_part.columns:
                    if col in self.df.columns: # Ensure column exists in self.df (it should if from original_columns)
                        self.df[col] = imputed_df_part[col]
                    else:
                        # This would happen if imputed_df_cols contains names not in original_columns (e.g. generic names)
                        # and we want to add them as new columns. For now, we only update existing ones.
                        logger.warning(f"Imputed column '{col}' not originally in self.df to update. This is unexpected if columns weren't generic.")
            elif len(original_columns) > 0 and imputed_values_array.shape[1] == 0 :
                 logger.info(f"No data returned by imputer for columns {original_columns.tolist()}. These columns in self.df remain as they were (likely all NaN).")

            logger.info(f"Missing values handled. Shape after imputation: {self.df.shape}")

        except ValueError as e:
            # Catch specific errors from KNNImputer related to all-NaN columns or insufficient data
            if "Input X contains column" in str(e) and "only NaN values" in str(e) or \
               "No training data" in str(e) or "must be denominator" in str(e) or \
               "n_neighbors <= n_samples" in str(e): # common KNNImputer errors
                logger.warning(f"KNNImputer could not process data: {e}. Numeric columns with issues remain unimputed.")
            else:
                logger.error(f"An unexpected ValueError occurred during KNN imputation: {e}")
                raise # Re-raise other ValueErrors
        except Exception as e:
            logger.error(f"An unexpected error occurred during KNN imputation: {e}")
            raise # Re-raise other unexpected errors
            
        return self.df