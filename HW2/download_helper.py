'''
@Author: Koda Song
@Date: 2020-04-10 22:45:29
@LastEditors: Koda Song
@LastEditTime: 2020-04-11 10:49:50
@Description: An automatically helper .py to download Price Data From Yahoo Finance
'''

import yfinance as yf
import pandas_datareader.data as pdr
import numpy as np
import pandas as pd

class Download_Helper():
    def __init__(self):
        return

    def Download(self, Tickers, start, end, freq = '1d', dropna = False, benchmark = False, close_only = False):
        """
        Description: Download stock price data from start till end
        ----------------------------------------------------------
        Parameters:
            Tickers: str, list
            start/end: str
                e.g. "1997-06-26"
            freq: str
                1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                Default is 1d
            dropna: bool
                Whether to dropna missing values
                Default is False
            benchmark: bool
                Whether to download SPY and merge with df
                Default is False
            close_only: bool
                Whether to keep only close price
        -----------------------------------------------------------
        return: a dataframe, whose index is date
        """
        yf.pdr_override()
        # Download one stock
        if type(Tickers) == str:
            df = pdr.get_data_yahoo(Tickers, start, end, interval = freq, as_panel = False)
            df['Return'] = df['Close'] / df['Close'].shift(1) - 1
        # Download several stocks
        elif type(Tickers) == list:
            dict_ = {}
            for ticker in Tickers:
                dict_[ticker] = pdr.get_data_yahoo(Tickers, start, end, interval = freq, as_panel = False)

            keys = list(dict_.keys()) # Make sure only successfully downloaded tickers are included

            df = pd.DataFrame(dict_[keys[0]])
            df['Return'] = df['Close'] / df['Close'].shift(1) - 1
            df['Ticker'] = keys[0]

            for key in keys[1:]:
                a = pd.DataFrame(df[key])
                a['Return'] = a['Close'] / a['Close'].shift(1) - 1
                a['Ticker'] = key
                #a.reset_index(inplace=True)
                df_Close = pd.concat([df, a])

        if close_only == True:
            df = df[['Close', 'Return']]

        if benchmark == True:
            SPY_ = pdr.get_data_yahoo('SPY', start, end, as_panel = False)
            SPY = SPY_[['Close']].copy()
            SPY['SPY Return'] = SPY['Close'] / SPY['Close'].shift(1) - 1
            SPY.rename(columns={'Close': 'SPY Close'})
            df = pd.merge(df, SPY, on = 'Date', how = 'outer')

        df.dropna(inplace = dropna)
        df.reset_index(inplace=True)
        return df