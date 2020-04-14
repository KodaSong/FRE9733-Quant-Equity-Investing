'''
@Author: Koda Song
@Date: 2020-04-10 23:22:25
@LastEditors: Koda Song
@LastEditTime: 2020-04-11 00:25:30
@Description: A helper .py to generate factors and rank them
'''
import numpy as np
import pandas as pd

class Factor_Helper():
    def __init__(self):
        return

    def Generate_Momentum(self, df_, freq, dropna = True, benchmark = False):
        """
        Description: Generate Momentum factors
        --------------------------------------
        Parameters:
            df: pandas dataframe
            freq: str, list
                1W, 1M, 3M, 6M, 1Y
            dropna: bool
                Whether to drop missing values
                Default is True
            benchmark: bool
                Whether to calculate SPY's Momentum
                Default is False
        --------------------------------------
        return:
            a dataframe with factors
        """
        def if_else_stock(freq, df):
            if freq == '1W':
                df['PM_1W'] = df['Close'] / df['Close'].shift(5) - 1
            elif freq == '1M':
                df['PM_1M'] = df['Close'] / df['Close'].shift(22) - 1
            elif freq == '3M':
                df['PM_3M'] = df['Close'] / df['Close'].shift(63) - 1
            elif freq == '6M':
                df['PM_6M'] = df['Close'] / df['Close'].shift(125) - 1

        def if_else_benchmark(freq, df):
            if freq == '1W':
                df['SPY_1W'] = df['SPY Close'] / df['SPY Close'].shift(5) - 1
            elif freq == '1M':
                df['SPY_1M'] = df['SPY Close'] / df['SPY Close'].shift(22) - 1
            elif freq == '3M':
                df['SPY_3M'] = df['SPY Close'] / df['SPY Close'].shift(63) - 1
            elif freq == '6M':
                df['SPY_6M'] = df['SPY Close'] / df['SPY Close'].shift(125) - 1

        df = df_.copy()
        if type(freq) == str:
            if_else_stock(df)
        elif type(freq) == list:
            for f in freq:
                df = if_else_stock(f, df)

        if benchmark == True:
            if type(freq) == str:
                if_else_benchmark(df)
            elif type(freq) == list:
                for f in freq:
                    df = if_else_benchmark(f, df)

        df.dropna(inplace = dropna)
        return df

    def Rank(self, df_, Factors):
        """"
        Description: Transform Percentile to Score from 1 to 9
        ------------------------------------------------------
        Param:
            df_: pandas dataframe
                Has columns of factors
            Factors: str, list
        ------------------------------------------------------
        return:
            a dataframe with new columns
        """
        df = df_.copy()
        Dates = list(set(df['Date'].values)).sort()

        if type(Factors) == str:
            factor_rank = Factors + ' Rank'

            # First, we select the first date's dataframe and rank
            dataframe = df[df['Date'] == Dates[0]].copy()
            dataframe[factor_rank] = dataframe[Factors].rank(pct=True).apply(lambda x: Percentile_to_Score(x))

            for i in range(1, len(Dates)):
                DF = df[df['Date'] == Dates[i]].copy()  # Divide the big df into small DF by date
                DF[factor_rank] = DF[Factors].rank(pct=True).apply(lambda x: Percentile_to_Score(x))
                dataframe = pd.concat([dataframe, DF])

            return dataframe

        if type(Factors) == list:
            f = Factors[0]
            f_r = f + ' Rank'
            dataframe = df[df['Date'] == Dates[0]].copy()
            dataframe[f_r] = dataframe[f].rank(pct=True).apply(lambda x: Percentile_to_Score(x))
            for i in range(1, len(Dates)):
                DF = df[df['Date'] == Dates[i]].copy()  # Divide the big df into small DF by date
                DF[f_r] = DF[f].rank(pct=True).apply(lambda x: Percentile_to_Score(x))
                dataframe = pd.concat([dataframe, DF])
            print("[*********************100%***********************] " + f + " completed")

            for factor in Factors[1:]:
                factor_rank = factor + ' Rank'
                for i in range(len(Dates)):
                    DF = df[df['Date'] == Dates[i]].copy()  # Divide the big df into small DF by date
                    DF[factor_rank] = DF[Factors].rank(pct=True).apply(lambda x: Percentile_to_Score(x))
                    dataframe = pd.concat([dataframe, DF])
                print("[*********************100%***********************] " + factor + " completed")
        return df





def Percentile_to_Score(Percentile):
    """
    Description: Transform Percentile to Score from 1 to 9
    ------------------------------------------------------
    Param:
        Percentile: float
            Range from 0 - 1
    ------------------------------------------------------
    return:
        Score: int
            Range from 1 - 10, 1 is highest and 10 is lowest
    """

    if Percentile >= 0.9: Score = 1
    elif Percentile >= 0.8: Score = 2
    elif Percentile >= 0.7: Score = 3
    elif Percentile >= 0.6: Score = 4
    elif Percentile >= 0.5: Score = 5
    elif Percentile >= 0.4: Score = 6
    elif Percentile >= 0.3: Score = 7
    elif Percentile >= 0.2: Score = 8
    elif Percentile >= 0.1: Score = 9
    elif Percentile >= 0.0: Score = 10

    return Score