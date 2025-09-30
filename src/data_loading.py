import pandas as pd
from .config import DATA_DIR

def load_fd001(train_path=None, test_path=None, rul_path=None):
    """
    Load FD001 train, test, and RUL datasets.
    """
    if train_path is None:
        train_path = f"{DATA_DIR}/train_FD001.txt"
    if test_path is None:
        test_path = f"{DATA_DIR}/test_FD001.txt"
    if rul_path is None:
        rul_path = f"{DATA_DIR}/RUL_FD001.txt"

    col_names = ['engine_id', 'cycle'] + \
                [f'op_setting_{i}' for i in range(1,4)] + \
                [f'sensor_{i}' for i in range(1,22)]

    train_df = pd.read_csv(train_path, sep=r"\s+", names=col_names, header=None)
    test_df = pd.read_csv(test_path, sep=r"\s+", names=col_names, header=None)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", names=['RUL'], header=None)

    train_df = train_df.dropna(axis=1, how='all')
    test_df = test_df.dropna(axis=1, how='all')

    return train_df, test_df, rul_df