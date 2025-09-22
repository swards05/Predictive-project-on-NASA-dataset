import pandas as pd

def load_fd001(train_path, test_path, rul_path):
    """
    Load the FD001 dataset (train, test, RUL).
    Returns: train_df, test_df, rul_df
    """

    # Column names: engine_id, cycle, 3 settings, 21 sensors
    col_names = ['engine_id', 'cycle'] + \
                [f'op_setting_{i}' for i in range(1, 4)] + \
                [f'sensor_{i}' for i in range(1, 22)]

    # Load train and test
    train_df = pd.read_csv(train_path, sep=r"\s+", names=col_names, header=None)
    test_df = pd.read_csv(test_path, sep=r"\s+", names=col_names, header=None)

    # Load RUL (one row per engine in test set)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=['RUL'])

    # Drop empty columns (sometimes extra spaces create NaN columns)
    train_df = train_df.dropna(axis=1, how='all')
    test_df = test_df.dropna(axis=1, how='all')

    return train_df, test_df, rul_df


if __name__ == "__main__":
    base_path = r"C:\Users\HP\Desktop\apa_proj\data"
    
    train_path = base_path + "\\train_FD001.txt"
    test_path = base_path + "\\test_FD001.txt"
    rul_path = base_path + "\\RUL_FD001.txt"

    train_df, test_df, rul_df = load_fd001(train_path, test_path, rul_path)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("RUL shape:", rul_df.shape)

    print("\nTrain head:\n", train_df.head())
    print("\nTest head:\n", test_df.head())
    print("\nRUL head:\n", rul_df.head())
