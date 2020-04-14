'''
@Author: Koda Song
@Date: 2020-04-08 14:18:12
@LastEditors: Koda Song
@LastEditTime: 2020-04-11 10:48:21
@Description: All the functions needed in notebook
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Base():
    def __init__(self, df, Factors):
        self.df = df 
        self.Factors = Factors
        self.PnL = None
        self.First_df = None
        self.BackTest_df = None
        self.Top_List = None
        self.Bottom_List = None

    def Delete(self, df, Tickers, Date):
        """
        Delete Tickers which does not have data on Date
        """
        Tickers_Part = set(df[df['Date'] == Date]['Ticker'].values)
        Tickers_Total = set(Tickers)
        Tickers_Drop = list(Tickers_Total - Tickers_Part)

        for t in Tickers_Drop:
            Index = list(df[df['Ticker'] == t].index.values)
            df.drop(Index, inplace=True)

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df

    def Select_Stock_List(self, N, Option):
        """
        Based on each factor, we individually select 2 stocks for Long and for short.
        """
        if Option == 'BackTest':
            df = self.First_df.copy()
        elif Option == 'Trade':
            df = self.BackTest_df.copy()

        self.Top_List = {}; self.Bottom_List = {}  # BackTest

        # Collect Stock List for BackTesting
        for factor in self.Factors:
            Rank_List = df.groupby('Ticker')[factor].mean().sort_values(ascending = True)
            #N = int(len(Rank_List) / 10)
            self.Top_List[factor] = list(Rank_List.index.values[:N])
            self.Bottom_List[factor] = list(Rank_List.index.values[-N:])

    def Determine_Stock(self, Option):
        """
        Based on List of BackTest, we select the stocks which appear most in Top List and Bottom List
        """
        if Option == 'BackTest':
            df = self.First_df.copy()
        elif Option == 'Trade':
            df = self.BackTest_df.copy()

        Top = {}; Bottom = {} # Construct a Dictionary to record the frequency of each ticker
        Top_List = []; Bottom_List = []

        for factor in self.Factors:
            Top_List += self.Top_List[factor];
            Bottom_List += self.Bottom_List[factor];

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


    def Show_Pairs(self, Option):
        '''
        @Description: Show Element Stocks and their weight in Each Pair
        @Param:
        @Return:
        '''
        if Option == 'BackTest':
            df = self.First_df.copy()
        elif Option == 'Trade':
            df = self.BackTest_df.copy()

        # r1_long = df[df.Ticker == self.Top[0]].copy()['Return']
        # r2_long = df[df.Ticker == self.Top[1]].copy()['Return']
        # r1_short = df[df.Ticker == self.Bottom[0]].copy()['Return']
        # r2_short = df[df.Ticker == self.Bottom[1]].copy()['Return']
        # w1_long, w2_long = Optimize_Return(r1_long, r2_long) # w1 + w2 = 1
        # w1_short, w2_short = Optimize_Return(r1_short, r2_short)
        Wealth = 25e6
        w1_long, w2_long = Optimize_PnL(df, self.Top[0], self.Top[1], Wealth = 25e6) # w1 + w2 = 1
        w1_short, w2_short = Optimize_PnL(df, self.Bottom[0], self.Bottom[1], Wealth = 25e6) # w1 + w2 = 1

        pair1 = pd.Series(['Pair1', self.Top[0], w1_long, self.Bottom[0], w1_short])
        pair2 = pd.Series(['Pair2', self.Top[1], w2_long, self.Bottom[1], w2_short])
        self.Pair_Information = pd.DataFrame([pair1, pair2])
        self.Pair_Information.set_axis(['Pair', 'Long', 'Long_Weight', 'Short', 'Short_Weight'], axis = 'columns', inplace = True)


    def Cal_PnL(self, Option):
        if Option == 'BackTest':
            df = self.BackTest_df.copy()
        elif Option == 'Trade':
            df = self.Trading_df.copy()

        #S0 = 25e6 / 2 # Start Wealth For each Pair

        # Get Series of each stock's Close Price
        PnL = {}
        for i in range(1,3):
            S_Long = 25e6 * self.Pair_Information.loc[i-1, 'Long_Weight']  # Start Wealth For each Pair; Pair 1 is larger
            S_Short = 25e6 * self.Pair_Information.loc[i-1, 'Short_Weight']

            Long_Close = df[df['Ticker'] == self.Top[i-1]]['Close']
            Short_Close = df[df['Ticker'] == self.Bottom[i-1]]['Close']

            Long_Share = (S_Long / Long_Close.values[0] // 100) * 100
            Short_Share = (S_Short / Short_Close.values[0] // 100) * 100

            pnl = Long_Share * Long_Close.diff() - Short_Share * Short_Close.diff()
            PnL["PnL" + str(i)] = pnl
            # PnL["Cash" + str(i)] = 25e6 - S_Long - S_Short

        self.PnL = pd.DataFrame({Pair:Data for Pair, Data in PnL.items()})

        # self.PnL['Alpha1'] = df[df['Ticker'] == self.Top[0]]['Alpha'] - df[df['Ticker'] == self.Bottom[0]]['Alpha']
        # self.PnL['Alpha2'] = df[df['Ticker'] == self.Top[1]]['Alpha'] - df[df['Ticker'] == self.Bottom[1]]['Alpha']
        # self.PnL['Total Alpha'] = self.PnL['Alpha1'] + self.PnL['Alpha2']

        for i in range(1,3):
            S_Long = 25e6 * self.Pair_Information.loc[i-1, 'Long_Weight']
            S_Short = 25e6 * self.Pair_Information.loc[i-1, 'Short_Weight']
            self.PnL['PnL' + str(i)].iloc[0] = 0
            self.PnL['Daily Return' + str(i)] = self.PnL['PnL' + str(i)] / (S_Long + S_Short)
            #self.PnL['Daily Return' + str(i)] = self.PnL['PnL' + str(i)] / (25e6)
            self.PnL['Cum Return' + str(i)] = self.PnL['Daily Return' + str(i)].cumsum() + 1

        self.PnL['Total Daily PnL'] = self.PnL['PnL1'] + self.PnL['PnL2']
        self.PnL['Total Daily Return'] = self.PnL['Total Daily PnL'] / 5e7
        self.PnL['Total Cum Return'] = self.PnL['Total Daily Return'].cumsum() + 1

        self.PnL['Daily SPY'] = df[df['Ticker'] == self.Top[0]]['SPY Close'] / df[df['Ticker'] == self.Top[0]]['SPY Close'].shift(1) - 1
        self.PnL['Daily SPY'].iloc[0] = 0.0
        self.PnL['Cum SPY'] = self.PnL['Daily SPY'].cumsum() + 1.0

    def Plot_Return(self):
        R1 = self.PnL['Cum Return1'].copy()
        R2 = self.PnL['Cum Return2'].copy()
        R3 = self.PnL['Total Cum Return'].copy()
        RM = self.PnL['Cum SPY'].copy()

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        plt.plot(R1, label = 'Pair1')
        plt.plot(R2, label = 'Pair2')
        plt.plot(R3, label = 'Total')
        plt.plot(RM, label = 'BenchMark')

        colormap = plt.cm.gist_ncar
        colors = [colormap(i) for i in np.linspace(0.4, 0.9, len(ax.lines))]

        for i,j in enumerate(ax.lines):
            j.set_color(colors[i])

        plt.legend()
        plt.show()

class BackTest(Base):
    def Construct_DF(self, Tickers, Date1, Date2):
        """
        Select DataFames we need 
        """
        First_df = self.df[self.df['Date'] <= Date1].copy()  # '2019-02-28'
        BackTest_df = self.df[(self.df['Date'] > Date1) & (self.df['Date'] <= Date2)].copy() # '2020-02-28'

        self.First_df = self.Delete(First_df, Tickers, Date1) # Delete Stocks which don't have data after Day1

        BackTest_df['Date'] = pd.to_datetime(BackTest_df['Date'])
        BackTest_df.set_index('Date', inplace=True)
        self.BackTest_df = BackTest_df

class Trade(Base):
    def Construct_DF(self, Tickers, Date1, Date2, Date3):
        """
        Select DataFames we need 
        """
        BackTest_df = self.df[(self.df['Date'] > Date1) & (self.df['Date'] <= Date2)].copy() # '2020-02-28'
        Trading_df = self.df[(self.df['Date'] > Date2) & (self.df['Date'] <= Date3)].copy() # '2020-03-31'
        self.BackTest_df = self.Delete(BackTest_df, Tickers, Date2) # Delete Stocks which don't have data after Day1

        Trading_df['Date'] = pd.to_datetime(Trading_df['Date'])
        Trading_df.set_index('Date', inplace=True)
        self.Trading_df = Trading_df


def Optimize_Return(r1, r2):
    '''
    @Description: To minize portfolio's return variance
    @Param: r1, r2 are Series of Return for two Assets
    @Return: w1, w2
    '''
    sigma1 = np.std(r1)#sigma1 = r1.std(ddof=0)
    sigma2 = np.std(r2)#sigma2 = r2.std(ddof=0)
    Cov = np.cov(r1,r2)[0][1]#rho = r1.corr(r2)
    w1 = 0
    min_var = 1000
    while w1 <= 1.0:
        w2 = 1 - w1
        var = (w1*sigma1)**2 + (w2*sigma2)**2 + 2*w1*w2*Cov
        if var < min_var:
            min_w1 = w1
            min_w2 = w2
            min_var = var
        w1 += 0.001
    return min_w1, min_w2

def Optimize_PnL(df, Ticker1, Ticker2, Wealth):
    '''
    @Description: To minize portfolio's pnl variance
    @Param: price1, price2 are Series of Close Price for two Assets
    @Return: w1, w2
    '''
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