'''
@Author: Koda Song
@Date: 2020-04-07 20:33:47
@LastEditors: Koda Song
@LastEditTime: 2020-04-10 10:32:39
@Description: Calculate a strategy's statistics
'''

import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
from scipy.stats import t

def Cal_MaxDD(r_):
    '''
    @Description: Calculate Maximum DrawDown
    @Param: a Series of Return (Daily, Monthly...), r should start from 0
    @Return: a float
    '''
    max_val = 0
    max_dd = 0
    r = list(r_.cumsum() + 1.0)
    #r.insert(0, 1.0)
    for i in range(1, len(r)):
        max_val = max(max_val, r[i-1])
        max_dd = max(max_dd, 1 - r[i] / max_val)

    return max_dd

def Description(r_):
    '''
    @Description: Show summary statistics of Return
    @Param: a Series of Return (Daily, Monthly...)
    @Return:
    '''
    print(r_.describe())

def Cal_VaR(pnl):
    '''
    @Description: Calculate 95% VaR of PnL
    @Param: a Series of PnL (Daily, Monthly...)
    @Return: a float
    '''
    from scipy.stats import norm
    Z = norm.ppf(0.95)
    pnl = -pnl # Loss is positive
    avg = pnl.mean()
    std = pnl.std()

    return avg + std * Z

def Cal_IR(r_, rm_, freq):
    '''
    @Description: Calculate Information Ratio
    @Param: Return, Benchmark Return, frequency (d, m, q, y)
    @Return: A float
    '''
    r = r_ - rm_  # Excess Daily Return
    avg = r.mean()
    std = r.std()

    if freq == 'd':
        return avg / std * np.sqrt(252)
    elif freq == 'm':
        return avg / std * np.sqrt(12)
    elif freq == 'q':
        return avg / std * np.sqrt(4)
    elif freq == 'y':
        return avg / std

def TTest(r_):
    # t = ttest_1samp(r_, 0)[0]
    # p = ttest_1samp(r_, 0)[1]
    n = len(r_)
    T = np.mean(r_) / np.std(r_) * np.sqrt(252)
    p = t.sf(np.abs(T), n-1)*2
    print("t-statistics: ", T, "; ", "p-value: ", p)


def Summarize(df, freq):
    '''
    @Description: Summarize return's risk characteristics
    @Param: a Series of Return / PnL (Daily, Monthly...)
    @Return:
    '''
    df_ = df.copy()
    r_ = df_['Total Daily Return']
    rm_ = df_['Daily SPY']
    pnl = df_['Total Daily PnL']
    #alpha = df_['Total Alpha']
    Final_Wealth = df_['Total Cum Return'].iloc[-1] * 50e6
    Total_PnL = Final_Wealth - 50e6

    print("Summary Statistics (Return):")
    Description(r_)
    print("--------------------------------------------")

    print("Summary Statistics (PnL):")
    Description(pnl)
    print("--------------------------------------------")

    # print("Summary Statistics (Yearly Alpha):")
    # print("Mean: ", alpha.mean())
    # TTest(alpha)
    # print("--------------------------------------------")

    print("Summary Statistics (Annualized Return):")
    print("Mean: ", r_.mean() * 252)
    TTest(r_)
    print("--------------------------------------------")

    print("Total PnL: ", Total_PnL)
    print("Final Wealth: ", Final_Wealth)
    print("--------------------------------------------")

    IR = Cal_IR(r_, rm_, freq)
    print("Information Ratio: {:.2f}".format(IR))
    print("--------------------------------------------")

    max_dd = Cal_MaxDD(r_) * 100
    print("Maximum DrawDown: {:.2f}".format(max_dd) + "%")
    print("--------------------------------------------")

    VaR = Cal_VaR(pnl)
    print("VaR at 95% Level: ${:.2f}".format(VaR))


