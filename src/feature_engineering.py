import pandas as pd

DROP_SENSORS = ['sensor_1','sensor_5','sensor_6','sensor_10','sensor_16','sensor_18','sensor_19']

def select_useful_features(df):
    df = df.copy()
    feature_cols = [c for c in df.columns if c.startswith('sensor_') and c not in DROP_SENSORS]
    op_cols = [c for c in df.columns if c.startswith('op_setting')]
    selected_cols = ['engine_id','cycle'] + op_cols + feature_cols + ['RUL']
    return df[selected_cols]

def add_rolling_features(df, window_sizes=[5,10,20]):
    df = df.copy()
    sensors = [c for c in df.columns if c.startswith('sensor_')]
    rolled_frames = []
    for w in window_sizes:
        mean_df = df.groupby('engine_id')[sensors].rolling(window=w, min_periods=1).mean()
        mean_df.index = mean_df.index.droplevel(0)
        mean_df = mean_df.add_suffix(f'_mean{w}')
        std_df = df.groupby('engine_id')[sensors].rolling(window=w, min_periods=1).std().fillna(0)
        std_df.index = std_df.index.droplevel(0)
        std_df = std_df.add_suffix(f'_std{w}')
        rolled_frames.append(mean_df)
        rolled_frames.append(std_df)

    rolled = pd.concat(rolled_frames, axis=1)
    df = pd.concat([df.reset_index(drop=True), rolled.reset_index(drop=True)], axis=1)
    return df

def get_train_test_features(train_df, test_df, window_sizes=[5,10,20]):
    train_df = select_useful_features(train_df)
    test_df = select_useful_features(test_df)
    train_df = add_rolling_features(train_df, window_sizes)
    test_df = add_rolling_features(test_df, window_sizes)
    X_train = train_df.drop(columns=['engine_id','cycle','RUL'])
    y_train = train_df['RUL']
    X_test = test_df.drop(columns=['engine_id','cycle','RUL'])
    y_test = test_df['RUL']

    print(f"[SUMMARY] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def feature_engineering(train_df, test_df, window_sizes=[5,10,20]):
    return get_train_test_features(train_df, test_df, window_sizes)
