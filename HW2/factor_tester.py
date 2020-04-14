'''
@Author: Koda Song
@Date: 2020-04-11 00:04:58
@LastEditors: Koda Song
@LastEditTime: 2020-04-12 12:05:52
@Description: BackTest on Momentum Stock Selection, 2 Pairs
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from scipy.stats import t

class Factor_Tester():
    def __init__(self, df, Factors, Option):
        self.df = df
        self.Factors = Factors
        self.Option = Option
        self.PnL = None
        self.First_df = None
        self.BackTest_df = None
        self.Top_List = None
        self.Bottom_List = None

    def Delete(self, df, Tickers, Date):
        """
        Description: Delete Tickers which does not have data on Selection Date
        ----------------------------------------------------------------------
        Param:
            df: pandas dataframe
                df needed to delete unavailable tickers
            Tickers: list
                All tickers in the big total df
            Date: str
                Selection Date / The day before the first day of backtesting
        ------------------------------------------------------
        return:
            a dataframe with data of tickers which are available on selection date
        """
        Tickers_Total = set(Tickers)
        Tickers_Part = set(df[df['Date'] == Date]['Ticker'].values)
        Tickers_Drop = list(Tickers_Total - Tickers_Part)

        for t in Tickers_Drop:
            Index = list(df[df['Ticker'] == t].index.values)
            df.drop(Index, inplace=True)

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df

    def Select_Stock_List(self, N):
        """
        Description: Based on each factor, we individually select 2 stocks for Long and for short.
        ----------------------------------------------------------------------
        Param:
            N: int
                Number of Stocks Selected and Put into List
            Option: str
                'BackTest', 'Trade'
        """
        if self.Option == 'BackTest':
            df = self.First_df.copy()
        elif self.Option == 'Trade':
            df = self.BackTest_df.copy()

        self.Top_List = {}; self.Bottom_List = {}

        # Collect Stock List for BackTesting
        for factor in self.Factors:
            Rank_List = df.groupby('Ticker')[factor].mean().sort_values(ascending = True)
            #N = int(len(Rank_List) / 10)
            self.Top_List[factor] = list(Rank_List.index.values[:N])
            self.Bottom_List[factor] = list(Rank_List.index.values[-N:])

    def Determine_Stock(self):
        """
        Description: Determine 2 stocks to long and 2 stocks to short
        ----------------------------------------------------------------------
        Param:
            Option: str
                'BackTest', 'Trade'
        """
        if self.Option == 'BackTest':
            df = self.First_df.copy()
        elif self.Option == 'Trade':
            df = self.BackTest_df.copy()

        Top = {}; Bottom = {} # Construct a Dictionary to record the frequency of each ticker
        Top_List = []; Bottom_List = []

        for factor in self.Factors:
            Top_List += self.Top_List[factor]
            Bottom_List += self.Bottom_List[factor]

        tmp1 = set(Top_List); tmp2 = set(Bottom_List)
        max1_key = None; max2_key = None  # Extract the most 2 tickers in Top
        max1_val = 0; max2_val = 0
        for item in tmp1:
            r1 = df[df.Ticker == max1_key]['Return'].iloc[-10:].mean() # Previous 10 days' return
            r2 = df[df.Ticker == max2_key]['Return'].iloc[-10:].mean()
            r = df[df.Ticker == item]['Return'].iloc[-10:].mean()

            if (max1_val < Top_List.count(item)) or (max1_val == Top_List.count(item) and r > r1):
                # If item > max1, max1 -> max2, Num -> max1
                # If item = max1 and r > r1, max1 -> max2, Num -> max1
                max2_key = max1_key
                max2_val = max1_val
                max1_key = item
                max1_val = Top_List.count(item)

            elif max1_val == Top_List.count(item) and r < r1:   # if equal, we need to decide which one is Pair1
                if (max1_val > max2_val) or (max1_val == max2_val and r > r2):
                    # if item = max1 > max2 and r < r1,  item -> max2
                    # if item = max1 = max2, r2 < r < r1, item -> max2
                    max2_key = item
                    max2_val = Top_List.count(item)

            elif (max2_val < Top_List.count(item)) or (max2_val == Top_List.count(item) and r > r2):
                # if max2 < item < max1, item -> max2
                # if max2 = item < max1 and r > r2, item -> max2
                max2_key = item
                max2_val = Top_List.count(item)
        self.Top = [max1_key, max2_key]

        max1_key = None; max2_key = None
        max1_val = 0; max2_val = 0  # Extract the most 2 tickers in Bottom
        for item in tmp2:
            r1 = df[df.Ticker == max1_key]['Return'].iloc[-10:].mean() # Previous 10 days' return
            r2 = df[df.Ticker == max2_key]['Return'].iloc[-10:].mean()
            r = df[df.Ticker == item]['Return'].iloc[-10:].mean()

            if (max1_val < Bottom_List.count(item)) or (max1_val == Bottom_List.count(item) and r < r1):
                # If item > max1, max1 -> max2, Num -> max1
                # If item = max1 and r < r1, max1 -> max2, Num -> max1
                max2_key = max1_key
                max2_val = max1_val
                max1_key = item
                max1_val = Bottom_List.count(item)

            elif (max1_val == Bottom_List.count(item)) and (r > r1):
                if (max1_val > max2_val) or (max1_val == max2_val and r < r2): 
                    # If item = max1 > max2, item -> max2
                    # If item = max1 = max2 and r1 < r < r2, item -> max2
                    max2_key = item
                    max2_val = Bottom_List.count(item)

            elif (max2_val < Bottom_List.count(item)) or (max2_val == Bottom_List.count(item) and r < r2):
                # If max2 < item < max1, item -> max2
                # If max1 > max2 = item and r < r2, item -> max2
                max2_key = item
                max2_val = Bottom_List.count(item)

        for item in tmp2:
            Bottom[item] = Bottom_List.count(item)

        self.Bottom = [max1_key, max2_key]


    def Show_Portfolio(self):
        """
        Description: Show Element Stocks and their weight in Each Side (After Optimization)
        """
        if self.Option == 'BackTest':
            df = self.First_df.copy()
        elif self.Option == 'Trade':
            df = self.BackTest_df.copy()
        Wealth = 25e6
        w1_long, w2_long = Optimize_PnL(df, self.Top[0], self.Top[1], Wealth = 25e6) # w1 + w2 = 1
        w1_short, w2_short = Optimize_PnL(df, self.Bottom[0], self.Bottom[1], Wealth = 25e6) # w1 + w2 = 1

        Long = pd.Series(['Long', self.Top[0], w1_long, self.Top[1], w2_long])
        Short = pd.Series(['Short', self.Bottom[0], w1_short, self.Bottom[1], w2_short])
        self.Portfolio_Info = pd.DataFrame([Long, Short])
        self.Portfolio_Info.set_axis(['Side', '1st Stock', '1st Weight', '2nd Stock', '2nd Weight'], axis = 'columns', inplace = True)


    def Cal_PnL(self):
        """
        Description: Calculate PnL and return
        ----------------------------------------------------------------------
        Param:
            Option: str
                'BackTest', 'Trade'
        """
        if self.Option == 'BackTest':
            df = self.BackTest_df.copy()
        elif self.Option == 'Trade':
            df = self.Trading_df.copy()

        # Get Series of each stock's Close Price
        PnL = {}
        for i in range(1,3):
            Position1 = 25e6 * self.Portfolio_Info.loc[i-1, '1st Weight']
            Position2 = 25e6 * self.Portfolio_Info.loc[i-1, '2nd Weight']

            Close1 = df[df['Ticker'] == self.Portfolio_Info.loc[i-1, '1st Stock']]['Close']
            Close2 = df[df['Ticker'] == self.Portfolio_Info.loc[i-1, '2nd Stock']]['Close']

            Share1 = (Position1 / Close1.values[0] // 100) * 100
            Share2 = (Position2 / Close2.values[0] // 100) * 100

            PnL[i] = Share1 * Close1.diff() + Share2 * Close2.diff()

        Daily_PnL = PnL[1] - PnL[2]
        Net_Value = Daily_PnL.cumsum() + 5e7

        Daily_Return = Daily_PnL / 5e7
        Net_Value_Level = Daily_Return.cumsum() + 1

        SPY = df[df['Ticker'] == self.Top[0]]['SPY Close'] / df[df['Ticker'] == self.Top[0]]['SPY Close'].shift(1) - 1
        SPY_Level = SPY.cumsum() + 1.0

        self.PnL = pd.DataFrame({'Daily PnL':Daily_PnL, 'Net Value':Net_Value,
                                 'Daily Return':Daily_Return, 'SPY Daily':SPY,
                                 'Net Value Level':Net_Value_Level, 'SPY Level':SPY_Level})

        self.PnL.iloc[0,:] = [0, 5e7, 0.0, 0.0, 1.0, 1.0]

    def Plot_Return(self):
        R = self.PnL['Net Value Level'].copy()
        RM = self.PnL['SPY Level'].copy()

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        plt.plot(R, label = 'Net Value Level')
        plt.plot(RM, label = 'BenchMark')

        colormap = plt.cm.gist_ncar
        colors = [colormap(i) for i in np.linspace(0.4, 0.9, len(ax.lines))]

        for i,j in enumerate(ax.lines):
            j.set_color(colors[i])

        plt.legend()
        plt.show()

    def Summarize(self, freq):
        """
        Description: Summarize return's risk characteristics
        ---------------------------------------------------------------------
        Params:
            Option: str
                'BackTest', 'Trade'
            freq: str
                'd', 'w', 'm', 'q', 'y'
        ---------------------------------------------------------------------
        return:
        """
        df_ = self.PnL.copy()

        r_ = df_['Daily Return']
        rm_ = df_['SPY Daily']
        pnl = df_['Daily PnL']
        #alpha = df_['Total Alpha']
        Final_Wealth = df_['Net Value'].iloc[-1]
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
        print("Mean: ", np.mean(r_) * 252)
        print("Vol:", np.std(r_) * np.sqrt(252))
        TTest(r_, freq)
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


class BackTest(Factor_Tester):
    def Construct_DF(self, Tickers, Date1, Date2):
        """
        Description: Select DataFames we need
        ----------------------------------------------------------------------
        Param:
            Tickers: list
                All tickers in big total df, used for function "Delete"
            Date1: str
                End date of train set
            Date2: str
                End date of test set
        """
        First_df = self.df[self.df['Date'] <= Date1].copy()  # '2019-02-28'
        BackTest_df = self.df[(self.df['Date'] > Date1) & (self.df['Date'] <= Date2)].copy() # '2020-02-28'

        self.First_df = self.Delete(First_df, Tickers, Date1) # Delete Stocks which don't have data after Day1

        BackTest_df['Date'] = pd.to_datetime(BackTest_df['Date'])
        BackTest_df.set_index('Date', inplace=True)
        self.BackTest_df = BackTest_df

class Trade(Factor_Tester):
    def Construct_DF(self, Tickers, Date1, Date2, Date3):
        """
        Description: Select DataFames we need
        ----------------------------------------------------------------------
        Param:
            Tickers: list
                All tickers in big total df, used for function "Delete"
            Date1: str
                Start date of train set
            Date2: str
                End date of train set/ Start date of test set
            Date3: str
                End date of test set
        """
        BackTest_df = self.df[(self.df['Date'] > Date1) & (self.df['Date'] <= Date2)].copy() # '2020-02-28'
        Trading_df = self.df[(self.df['Date'] > Date2) & (self.df['Date'] <= Date3)].copy() # '2020-03-31'
        self.BackTest_df = self.Delete(BackTest_df, Tickers, Date2) # Delete Stocks which don't have data after Day1

        Trading_df['Date'] = pd.to_datetime(Trading_df['Date'])
        Trading_df.set_index('Date', inplace=True)
        self.Trading_df = Trading_df


def Optimize_Return(r1, r2):
    """
        Description: To minize portfolio's return variance
        ----------------------------------------------------------------------
        Param:
            r1/r2: pandas series
                Return for two Assets
        ----------------------------------------------------------------------
        return:
            w1/w2: float
                weight of asset1, asset2
    """
    # sigma1 = np.std(r1)#sigma1 = r1.std(ddof=0)
    # sigma2 = np.std(r2)#sigma2 = r2.std(ddof=0)
    # Cov = np.cov(r1,r2)[0][1]#rho = r1.corr(r2)
    Cov_Matrix = np.matrix(np.cov(r1,r2))
    w1 = 0
    min_var = 1000
    while w1 <= 1.0:
        w2 = 1 - w1
        w = np.array([w1, w2])
        #var = (w1*sigma1)**2 + (w2*sigma2)**2 + 2*w1*w2*Cov
        var = (w.dot(Cov_Matrix)).dot(w)[0,0]
        if var < min_var:
            min_w1 = w1
            min_w2 = w2
            min_var = var
        w1 += 0.001
    return min_w1, min_w2

def Optimize_PnL(df, Ticker1, Ticker2, Wealth):
    """
        Description: To minize portfolio's pnl variance
        ----------------------------------------------------------------------
        Param:
            Ticker1/Ticker2: str
                Tickers for two assets
            Wealth: float
                Wealth used to invest
        ----------------------------------------------------------------------
        return:
            w1/w2: float
                weight of asset1, asset2
    """
    price1 = df[df.Ticker == Ticker1]['Close'].copy()
    price2 = df[df.Ticker == Ticker2]['Close'].copy()
    r1 = df[df.Ticker == Ticker1]['Return'].copy()
    r2 = df[df.Ticker == Ticker2]['Return'].copy()

    S1 = price1.values[0] # Start Price
    S2 = price2.values[0]
    N1 = 0
    sigma1 = np.std(r1)
    sigma2 = np.std(r2)
    Cov = np.cov(r1, r2)[0][1]
    min_std = 1e20
    min_n1 = 0
    min_n2 = 0
    while(N1 <= Wealth / S1):
        N2 = ( (Wealth - N1*S1) / S2 ) // 100 * 100  # 1 Unit = 100 shares
        var = (N1*S1*sigma1)**2 + (N2*S2*sigma2)**2 + 2*Cov*N1*N2*S1*S2
        if np.sqrt(var) < min_std:
            min_std = np.sqrt(var)
            min_n1 = N1
            min_n2 = N2
        N1 += 100
    #print(sigma1, sigma2, min_std)
    w1 = (min_n1 * S1) / Wealth
    w2 = (min_n2 * S2) / Wealth

    return w1, w2

def Rolling(df_, Rf, window, freq = 'd', dropna = True):
    """
    Description: Use rolling window to calculate alpha, beta (Daily Data)
    ---------------------------------------------------------------------
    Params:
        df: dataframe
        Rf:
            Risk_Free Rate
        window: int
            Nums of rolling window
        freq: str
            'd', 'w', 'm', 'q', 'y'
        dropna: bool
            Whether to drop missing values
            Default is True
    ---------------------------------------------------------------------
    return:
        dataframe with yearly alpha
    """
    if freq == 'd':
        cc = 252  # Conversion Coefficient
    elif freq == 'w':
        cc = 52
    elif freq == 'm':
        cc = 12
    elif freq == 'q':
        cc = 4
    elif freq == 'y':
        cc = 1
    df = df_.copy()
    df = pd.merge(df, Rf, on = 'Date', how = 'outer')
    # Create dataframe to store first ticker's data
    Tickers = list(set(df['Ticker']))
    df = df[df['Ticker'] == Tickers[0]].copy()
    Var = df['SPY Return'].rolling(window=window).var() * cc
    Cov = df['Return'].rolling(window=window).cov(df['SPY Return']) * cc
    df['Beta'] = Cov / Var
    df['Alpha'] = (df['Return'] - df['Rf'] - df['Beta'] * (df['SPY Return'] - df['Rf'])) * cc
    # Merge new df with dataframe
    for i in range(1, len(Tickers)):
        DF = df[df['Ticker'] == Tickers[i]].copy()
        if len(DF) < window:
            continue
        Var = DF['SPY Return'].rolling(window=window).var() * cc
        Cov = DF['Return'].rolling(window=window).cov(DF['SPY Return']) * cc
        DF['Beta'] = Cov / Var
        DF['Alpha'] = (DF['Return'] - DF['Rf'] - DF['Beta'] * (DF['SPY Return'] - DF['Rf'])) * cc

        df = pd.concat([df, DF])

    df.dropna(inplace=dropna)
    return df

def Cal_MaxDD(r_):
    """
    Description: Calculate Maximum DrawDown
    ---------------------------------------------------------------------
    Params:
        r_: pandas series
            Return (Daily, Monthly...), r should start from 0
    ---------------------------------------------------------------------
    return:
        a float
    """
    max_val = 0
    max_dd = 0
    r = list(r_.cumsum() + 1.0)
    #r.insert(0, 1.0)
    for i in range(1, len(r)):
        max_val = max(max_val, r[i-1])
        max_dd = max(max_dd, 1 - r[i] / max_val)

    return max_dd

def Description(r_):
    """
    Description: Show summary statistics of Return
    ---------------------------------------------------------------------
    Params:
        r_: pandas series
            Return (Daily, Monthly...), r should start from 0
    ---------------------------------------------------------------------
    return:
        a float
    """
    print(r_.describe())

def Cal_VaR(pnl):
    """
    Description: Calculate 95% VaR of PnL
    ---------------------------------------------------------------------
    Params:
        r_: pandas series
            PnL (Daily, Monthly...), r should start from 0
    ---------------------------------------------------------------------
    return:
        a float
    """
    from scipy.stats import norm
    Z = norm.ppf(0.95)
    pnl = -pnl # Loss is positive
    avg = pnl.mean()
    std = pnl.std()

    return avg + std * Z

def Cal_IR(r_, rm_, freq = 'd'):
    """
    Description: Calculate Information Ratio
    ---------------------------------------------------------------------
    Params:
        r_: pandas series
            Return (Daily, Monthly...), r should start from 0
        rm_: pandas series
            Benchmark Return (Daily, Monthly...), r should start from 0
        freq: str
            'd', 'm',  'w', 'q', 'y'
    ---------------------------------------------------------------------
    return:
        a float
    """
    r = r_ - rm_  # Excess Daily Return
    avg = r.mean()
    std = r.std()

    if freq == 'd':
        return avg / std * np.sqrt(252)
    if freq == 'w':
        return avg / std * np.sqrt(52)
    elif freq == 'm':
        return avg / std * np.sqrt(12)
    elif freq == 'q':
        return avg / std * np.sqrt(4)
    elif freq == 'y':
        return avg / std

def TTest(r_, freq ='d'):
    """
    Description: T test for Return
    ---------------------------------------------------------------------
    Params:
        r_: pandas series
            Return (Daily, Monthly...), r should start from 0
        rm_: pandas series
            Benchmark Return (Daily, Monthly...), r should start from 0
        freq: str
            'd', 'm', 'w', 'q', 'y'
    ---------------------------------------------------------------------
    return:
        a float
    """
    # t = ttest_1samp(r_, 0)[0]
    # p = ttest_1samp(r_, 0)[1]
    n = len(r_)
    if freq == 'd':
        cc = 252 # Conversion Coefficient
    if freq == 'm':
        cc = 52 # Conversion Coefficient
    if freq == 'q':
        cc = 4 # Conversion Coefficient
    if freq == 'y':
        cc = 1 # Conversion Coefficient

    T = np.mean(r_) / np.std(r_) * np.sqrt(cc)
    p = t.sf(np.abs(T), n-1)*2
    print("t-statistics: ", T, "; ", "p-value: ", p)