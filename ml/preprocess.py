import pandas as pd
from load_dataset import load_cmaps

def add_rul(df):
    max_cycle = df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycle.columns = ['unit_nr', 'max_cycle']
    df = df.merge(max_cycle, on='unit_nr', how='left')
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df.drop('max_cycle', axis=1, inplace=True)
    return df

if __name__ == "__main__":
    train_df, test_df, rul_df = load_cmaps()
    train_df = add_rul(train_df)
    
    # Create label column: 1 = failure soon, 0 = healthy
    train_df['label'] = train_df['RUL'].apply(lambda x: 1 if x <= 30 else 0)
    print(train_df['label'].value_counts())

    
    print(train_df.head())
    print(train_df.tail())
