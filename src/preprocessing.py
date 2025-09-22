import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def add_train_rul(train_df):
    """
    Compute RUL for each row in training set.
    """
    # Find max cycle per engine
    max_cycle = train_df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']

    # Merge back to train
    train_df = train_df.merge(max_cycle, on='engine_id', how='left')
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
    train_df.drop(columns=['max_cycle'], inplace=True)

    return train_df


def add_test_rul(test_df, rul_df):
    """
    Compute RUL for test set using provided RUL file.
    """
    # Max cycle for each engine in test
    test_max = test_df.groupby('engine_id')['cycle'].max().reset_index()
    test_max.columns = ['engine_id', 'max_cycle']

    # Attach provided RULs (same order)
    test_max['true_RUL'] = rul_df['RUL'].values

    # Compute failure cycle
    test_max['fail_cycle'] = test_max['max_cycle'] + test_max['true_RUL']

    # Map fail_cycle to each row
    fail_cycle_map = dict(zip(test_max['engine_id'], test_max['fail_cycle']))
    test_df['fail_cycle'] = test_df['engine_id'].map(fail_cycle_map)

    # Compute RUL
    test_df['RUL'] = test_df['fail_cycle'] - test_df['cycle']
    test_df.drop(columns=['fail_cycle'], inplace=True)

    return test_df


def cap_rul(df, cap=125):
    """
    Cap RUL values to a max threshold.
    """
    df['RUL'] = np.where(df['RUL'] > cap, cap, df['RUL'])
    return df


def scale_features(train_df, test_df):
    """
    Standardize sensor values and operational settings.
    """
    feature_cols = [c for c in train_df.columns if 'sensor' in c or 'op_setting' in c]

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, test_df, scaler


if __name__ == "__main__":
    from data_loading import load_fd001
    
    base_path = r"C:\Users\HP\Desktop\apa_proj\data"
    train_path = base_path + "\\train_FD001.txt"
    test_path = base_path + "\\test_FD001.txt"
    rul_path = base_path + "\\RUL_FD001.txt"

    # Load
    train_df, test_df, rul_df = load_fd001(train_path, test_path, rul_path)

    # Process
    train_df = add_train_rul(train_df)
    test_df = add_test_rul(test_df, rul_df)

    train_df = cap_rul(train_df, cap=125)
    test_df = cap_rul(test_df, cap=125)

    train_df, test_df, scaler = scale_features(train_df, test_df)

    print("Train ready:", train_df.shape)
    print("Test ready:", test_df.shape)
    print(train_df.head())
