import numpy as np
import pandas as pd
from backtesting import Backtest
from ta.trend import WMAIndicator
import matplotlib.pyplot as plt
import datetime
import backtesting
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')
excelName = r'/Users/milan/Downloads/ETHUSDT.xlsx'



def MovingAverageRSIStrategy(crypto_symbol='ETH', start_date='2020-02-02', end_date='2021-10-01',
                               short_window=20, long_window=50, moving_avg='SMA', rsi_window = 14, rsi_sell = 45, rsi_buy = 58):
    # crypto_symbol - Npr: 'ETH'
    # start_date - startni datum analize podataka (format: 'YYYY-MM-DD') Npr: '2018-01-01'
    # end_date - krajnji datum (format: 'YYYY-MM-DD') Eg: '2020-01-01'
    # short_window - (int)kraci period za MA indikator
    # long_window - (int)duzi period za MA indikator
    # moving_avg - tip moving average strategije ('SMA' ili 'EMA')
    # rsi_sell - vrednost za RSI koju uzimati za SELL
    # rsi_buy - vrednost za RSI koju uzimati za BUY

    crypto_df = pd.read_excel(excelName, index_col=0)
    crypto_df = pd.DataFrame(crypto_df)[['Close']]
    crypto_df = crypto_df.loc[start_date:end_date]
    crypto_df.dropna(axis=0, inplace=True)  # uklanjanje bilo kojeg null reda

    # naziv za kolone kao kraci window plus naziv SMA ili EMA
    short_window_col = str(short_window) + '_' + moving_avg
    long_window_col = str(long_window) + '_' + moving_avg

    #izracunavanje RSI-a (Relative Strenght Index)
    crypto_df['change'] = crypto_df['Close'].diff(1)
    # izracuvanje gain / loss od svake promene
    crypto_df['gain'] = np.select([crypto_df['change'] > 0, crypto_df['change'].isna()],
                           [crypto_df['change'], np.nan],
                           default=0)
    crypto_df['loss'] = np.select([crypto_df['change'] < 0, crypto_df['change'].isna()],
                           [-crypto_df['change'], np.nan],
                           default=0)
    # kreiranje avg_gain /  avg_loss kolona sa NAN (not-a-number) vrednostima
    crypto_df['avg_gain'] = np.nan
    crypto_df['avg_loss'] = np.nan
    n = rsi_window  # velicina prozora za RSI
    crypto_df['avg_gain'][n] = crypto_df['gain'].rolling(window=n).mean().dropna().iloc[0]
    crypto_df['avg_loss'][n] = crypto_df['loss'].rolling(window=n).mean().dropna().iloc[0]
    for i in range(n + 1, crypto_df.shape[0]):
        crypto_df['avg_gain'].iloc[i] = (crypto_df['avg_gain'].iloc[i - 1] * (n - 1) + crypto_df['gain'].iloc[i]) / n
        crypto_df['avg_loss'].iloc[i] = (crypto_df['avg_loss'].iloc[i - 1] * (n - 1) + crypto_df['loss'].iloc[i]) / n
    # izracuvanje rs and rsi
    crypto_df['rs'] = crypto_df['avg_gain'] / crypto_df['avg_loss']
    crypto_df['rsi'] = 100 - (100 / (1 + crypto_df['rs']))
    #odbacivanje redova gde je rsi veci od rsi_sell i manji od rsi_buy koji nisu od znacaja, tako da se kasnije upotrebi
    #ako je indeks veci od rsi_buy i postoji crossover nad kracim i duzim windowom za MA, onda BUY signal, u suportnom SELL
    crypto_df = crypto_df.drop(crypto_df[(crypto_df['rsi']>rsi_sell) & (crypto_df['rsi']<rsi_buy)].index)

    if moving_avg == 'SMA':
        crypto_df[short_window_col] = crypto_df['Close'].rolling(window=short_window, min_periods=1).mean()
        crypto_df[long_window_col] = crypto_df['Close'].rolling(window=long_window, min_periods=1).mean()

    elif moving_avg == 'EMA':
        crypto_df[short_window_col] = crypto_df['Close'].ewm(span=short_window, adjust=False).mean()
        crypto_df[long_window_col] = crypto_df['Close'].ewm(span=long_window, adjust=False).mean()

    #proba za koriscenje indikatora iz talib (Technical Analysis)
    wmaIndicator = WMAIndicator(close=crypto_df['Close'], window=21, fillna=False)
    #crypto_df['WMA'] = wmaIndicator.wma()


    # Kreiranje kolone 'Signal' po indikatorima, inicijalno setujemo na 0 pa odredujemo vrednost
    crypto_df['Signal'] = 0.0
    crypto_df['Signal'] = np.where((crypto_df[short_window_col] > crypto_df[long_window_col]), 1.0, 0.0)
    # Kreiranje nove kolone 'Position' koja je razlika izmedju satnih podataka tako da kreira signal
    # kad se desi promena po indeksima
    crypto_df['Position'] = crypto_df['Signal'].diff()

    # crtanje cene zatvaranja i kraceg i duzeg period indikatora
    plt.figure(figsize=(14, 6))
    plt.tick_params(axis='both', labelsize=14)
    crypto_df['Close'].plot(color='k', lw=1, label='Close')
    crypto_df[short_window_col].plot(color='b', lw=1, label=short_window_col)
    crypto_df[long_window_col].plot(color='g', lw=1, label=long_window_col)


    # crtanje 'buy' signala
    plt.plot(crypto_df[crypto_df['Position'] == -1].index,
             crypto_df[short_window_col][crypto_df['Position'] == -1],
             '^', markersize=15, color='g', alpha=0.8, label='buy')
    # crtanje 'sell' signala
    plt.plot(crypto_df[crypto_df['Position'] == 1].index,
             crypto_df[short_window_col][crypto_df['Position'] == 1],
             'v', markersize=15, color='r', alpha=0.8, label='sell')


    plt.ylabel('Price in $', fontsize=16)
    plt.xlabel('Date', fontsize=16)
    plt.title(str(crypto_symbol) + ' - ' + str(moving_avg) + ' Crossover with RSI index', fontsize=20)
    plt.legend()
    plt.grid()
    plt.show()

    crypto_df = crypto_df.drop(['change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], axis=1)

    #crtanje tabele u konzoli
    df_pos = crypto_df[(crypto_df['Position'] == 1) | (crypto_df['Position'] == -1)]
    df_pos['Position'] = df_pos['Position'].apply(lambda x: 'Sell' if x == 1 else 'Buy')
    print(tabulate(df_pos, headers="keys", tablefmt='psql'))

    

MovingAverageRSIStrategy(crypto_symbol='ETH', start_date='2021-09-01', end_date='2021-09-30',
                         short_window=7, long_window=21, moving_avg='EMA', rsi_window=14, rsi_sell=40, rsi_buy=60)


