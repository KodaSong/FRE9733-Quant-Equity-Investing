'''
@Author: Koda Song
@Date: 2020-04-06 22:51:47
@LastEditors: Koda Song
@LastEditTime: 2020-04-06 23:21:00
@Description: A function to rank for factors
'''

import pandas as pd

def Percentile_to_Score(Percentile):
    '''
    @Description: Transform Percentile to Score from 1 to 9
    @Param: Percentile
    @Return: Score
    '''

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

def Rank(df_, factor, Date):
    '''
    @Description: Rank for factor
    @Param: Dataframe, factor name, Date
    @Return: a new DataFrame
    '''
    factor_rank = factor + ' Rank'
    df = df_.copy()

    # First, we select the first date's dataframe and rank
    dataframe = df[df['Date'] == Date[0]].copy()
    dataframe[factor_rank] = dataframe[factor].rank(pct=True).apply(lambda x: Percentile_to_Score(x))

    for i in range(1, len(Date)):
        DF = df[df['Date'] == Date[i]].copy()  # Divide the big df into small DF by date
        DF[factor_rank] = DF[factor].rank(pct=True).apply(lambda x: Percentile_to_Score(x))
        dataframe = pd.concat([dataframe, DF])

    return dataframe