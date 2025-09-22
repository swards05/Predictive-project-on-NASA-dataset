# Predictive Maintenance of Jet Engines (NASA CMAPSS - FD001)

This project uses NASA's C-MAPSS dataset (FD001) to predict the **Remaining Useful Life (RUL)** of jet engines using Gradient Boosting.  
An **alert system** is implemented to flag engines that are close to failure.

## Project Workflow
1. Data Loading
2. Data Exploration & Visualization
3. Preprocessing & Feature Engineering
4. Model Training (Gradient Boosting)
5. Evaluation
6. Maintenance Alert System
7. Results & Visualizations

## Dataset
The FD001 dataset contains multiple engines run until failure. Each record has:
- Engine ID
- Cycle number
- Operational settings (3 features)
- 21 sensor measurements
- Remaining Useful Life (RUL, computed)

## Requirements
Install dependencies with:
        **pip install -r requirements.txt**



## step1: (data_loading)
        Assigns column names (engine ID, cycle, 3 settings, 21 sensors).
        Loads train, test, and RUL files.
        Cleans any extra empty columns.
        Prints dataset shapes and previews.

## step2:(eda_fd001)
        Preview dataset → first rows, summary, null values.
        Sensor values over time → check degradation trends.
        Correlation heatmap → see which sensors carry useful info.
        RUL distribution (for train).
        Example engine plot → show how RUL decreases with cycles.

## step3:(preprocessing)
        Compute RUL for training set.
        Compute RUL for test set using provided RUL_FD001.txt.
        Cap RUL (so outliers don’t dominate training).
        Drop constant/unused sensors.
        Apply scaling.
        Return cleaned train/test datasets.

## step4:(feature_engineering)
        Drops useless sensors
        Keeps op_settings + useful sensors.
        Adds rolling statistics (mean & std over window sizes, e.g., 5, 10, 20 cycles).
        Still outputs X_train, y_train, X_test, y_test cleanly.

## step5: (model_training)
        Responsible for training the models and saving them.

## step6: (evaluation)
        Handles metrics and visualization.
        
## step7: (main)
        Load data
        Preprocess + Feature engineering
        Train the model (or load if already saved)
        Evaluate performance
        I also added saving/loading with joblib so you don’t need to retrain every time.