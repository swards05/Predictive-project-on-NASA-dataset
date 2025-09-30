import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from .data_loading import load_fd001
from .config import SCALER_PATH

def add_train_rul(train_df):
    max_cycle = train_df.groupby('engine_id')['cycle'].transform('max')
    train_df = train_df.copy()
    train_df['RUL'] = max_cycle - train_df['cycle']
    return train_df

def add_test_rul(test_df, rul_df):
    test_max = test_df.groupby('engine_id')['cycle'].max().reset_index()
    test_max.columns = ['engine_id', 'max_cycle']
    test_max['true_RUL'] = rul_df['RUL'].values
    test_max['fail_cycle'] = test_max['max_cycle'] + test_max['true_RUL']
    fail_map = dict(zip(test_max['engine_id'], test_max['fail_cycle']))
    test_df = test_df.copy()
    test_df['fail_cycle'] = test_df['engine_id'].map(fail_map)
    test_df['RUL'] = test_df['fail_cycle'] - test_df['cycle']
    test_df = test_df.drop(columns=['fail_cycle'])
    return test_df

def cap_rul(df, cap=125):
    df = df.copy()
    df['RUL'] = np.minimum(df['RUL'], cap)
    return df

def scale_features(train_df, test_df, save_scaler=True):
    feature_cols = [c for c in train_df.columns if c.startswith('sensor_') or c.startswith('op_setting')]
    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    if save_scaler:
        joblib.dump(scaler, SCALER_PATH)
    return train_df, test_df, scaler

def load_and_preprocess(train_path=None, test_path=None, rul_path=None, cap=125):
    train_df, test_df, rul_df = load_fd001(train_path, test_path, rul_path)
    train_df = add_train_rul(train_df)
    test_df = add_test_rul(test_df, rul_df)
    train_df = cap_rul(train_df, cap)
    test_df = cap_rul(test_df, cap)
    train_df, test_df, scaler = scale_features(train_df, test_df)
    return train_df, test_df, scaler
