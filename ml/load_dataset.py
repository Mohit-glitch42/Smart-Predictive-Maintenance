import pandas as pd

def load_cmaps():
    col_names = ['unit_nr', 'time_cycles'] + \
                [f'op_setting_{i}' for i in range(1, 4)] + \
                [f'sensor_{i}' for i in range(1, 22)]

    train_df = pd.read_csv('dataset/train_FD001.txt', sep=' ', header=None)
    train_df.drop([26, 27], axis=1, inplace=True)
    train_df.columns = col_names

    test_df = pd.read_csv('dataset/test_FD001.txt', sep=' ', header=None)
    test_df.drop([26, 27], axis=1, inplace=True)
    test_df.columns = col_names

    rul_df = pd.read_csv('dataset/RUL_FD001.txt', sep=' ', header=None)

    return train_df, test_df, rul_df

train_df, test_df, rul_df = load_cmaps()
print(train_df.head())
