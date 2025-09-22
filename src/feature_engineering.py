import pandas as pd

def select_useful_features(df):
    """
    Drop constant/useless sensors and keep only useful ones.
    """
    drop_sensors = [
        'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
        'sensor_16', 'sensor_18', 'sensor_19'
    ]

    feature_cols = [c for c in df.columns if c.startswith("sensor_") and c not in drop_sensors]
    op_cols = [c for c in df.columns if c.startswith("op_setting")]
    
    selected_cols = ['engine_id', 'cycle'] + op_cols + feature_cols + ['RUL']
    return df[selected_cols]


def add_rolling_features(df, window_sizes=[5, 10, 20]):
    """
    Add rolling mean and std features for selected sensors.
    """
    sensors = [c for c in df.columns if c.startswith("sensor_")]

    for sensor in sensors:
        for w in window_sizes:
            df[f"{sensor}_mean{w}"] = (
                df.groupby("engine_id")[sensor].transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
            df[f"{sensor}_std{w}"] = (
                df.groupby("engine_id")[sensor].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
            )

    return df


def get_train_test_features(train_df, test_df, window_sizes=[5, 10, 20]):
    """
    Return X_train, y_train, X_test, y_test with advanced rolling features
    """
    train_df = select_useful_features(train_df)
    test_df = select_useful_features(test_df)

    # Add rolling features
    train_df = add_rolling_features(train_df, window_sizes)
    test_df = add_rolling_features(test_df, window_sizes)

    # Features (exclude engine_id, cycle, RUL)
    X_train = train_df.drop(columns=['engine_id', 'cycle', 'RUL'])
    y_train = train_df['RUL']

    X_test = test_df.drop(columns=['engine_id', 'cycle', 'RUL'])
    y_test = test_df['RUL']

    # Summary
    base_features = len([c for c in X_train.columns if not any(s in c for s in ["mean", "std"])])
    rolling_features = len(X_train.columns) - base_features
    print(f"\n[SUMMARY] Base features: {base_features}, Rolling features: {rolling_features}, Total: {X_train.shape[1]}")

    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print("\n[INFO] Example features:")
    print(X_train.head())

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    from data_loading import load_fd001
    from preprocessing import add_train_rul, add_test_rul, cap_rul, scale_features

    base_path = r"C:\Users\HP\Desktop\apa_proj\data"
    train_path = base_path + "\\train_FD001.txt"
    test_path = base_path + "\\test_FD001.txt"
    rul_path = base_path + "\\RUL_FD001.txt"

    # Load
    train_df, test_df, rul_df = load_fd001(train_path, test_path, rul_path)

    # Preprocess
    train_df = add_train_rul(train_df)
    test_df = add_test_rul(test_df, rul_df)
    train_df = cap_rul(train_df, cap=125)
    test_df = cap_rul(test_df, cap=125)
    train_df, test_df, scaler = scale_features(train_df, test_df)

    # Feature Engineering
    X_train, y_train, X_test, y_test = get_train_test_features(train_df, test_df)
