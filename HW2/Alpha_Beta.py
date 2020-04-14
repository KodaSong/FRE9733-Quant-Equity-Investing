'''
@Author: Koda Song
@Date: 2020-04-08 12:16:00
@LastEditors: Koda Song
@LastEditTime: 2020-04-12 11:03:18
@Description: Calculate Alpha & Beta of Panel Data
'''
import numpy as np
import pandas as pd

def Rolling(df, window):
    '''
    @Description: Use rolling window to calculate alpha, beta (Daily Data)
    @Param: dataframe, window (frequency)
    @Return: new dataframe
    '''
    # Create dataframe to store first ticker's data
    Tickers = list(set(df['Ticker']))
    dataframe = df[df['Ticker'] == Tickers[0]].copy()
    Var = dataframe['SPY Return'].rolling(window=window).var() * 252
    Cov = dataframe['Return'].rolling(window=window).cov(dataframe['SPY Return']) * 252
    dataframe['Beta'] = Cov / Var
    dataframe['Alpha'] = (dataframe['Return'] - dataframe['Rf'] - dataframe['Beta'] * (dataframe['SPY Return'] - dataframe['Rf'])) * 252
    # Merge new df with dataframe
    for i in range(1, len(Tickers)):
        DF = df[df['Ticker'] == Tickers[i]].copy()
        if len(DF) < window:
            continue
        Var = DF['SPY Return'].rolling(window=window).var() * 252
        Cov = DF['Return'].rolling(window=window).cov(DF['SPY Return']) * 252
        DF['Beta'] = Cov / Var
        DF['Alpha'] = (DF['Return'] - DF['Rf'] - DF['Beta'] * (DF['SPY Return'] - DF['Rf'])) * 252

        dataframe = pd.concat([dataframe, DF])

    return dataframe