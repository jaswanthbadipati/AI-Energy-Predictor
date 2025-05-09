# data_loader.py

from ucimlrepo import fetch_ucirepo
import pandas as pd

def load_and_preprocess():
    dataset = fetch_ucirepo(id=235)
    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('datetime', inplace=True)
    df.drop(['Date', 'Time'], axis=1, inplace=True)

    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    # Time features
    df['hour'] = df.index.hour
    df['day'] = df.index.dayofweek
    df['month'] = df.index.month

    # Lag feature (previous hour's global active power)
    df['lag_1'] = df['Global_active_power'].shift(1)
    df.dropna(inplace=True)  # Drop row with NaN from lag

    return df
