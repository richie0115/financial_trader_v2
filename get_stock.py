################################
#  get_stock.py                #
################################

import yfinance as yf
import datetime
import numpy as np
import pandas as pd
from typing import Callable

today = datetime.date.today()

def get_stock_info(ticker_symbol):
    """
    Fetches stock information for a given ticker symbol using yfinance.

    """
    try:
        stock = yf.Ticker(ticker_symbol)
        stock_info = stock.info

        # Extract relevant information
        result = {
            'currentPrice': stock_info.get('currentPrice'),
            'marketCap': stock_info.get('marketCap'),
            'trailingPE': stock_info.get('trailingPE'),
            'forwardPE': stock_info.get('forwardPE'),
            'dividendYield': stock_info.get('dividendYield'),
            '52WeekHigh': stock_info.get('fiftyTwoWeekHigh'),
            '52WeekLow': stock_info.get('fiftyTwoWeekLow'),
        }

        return result

    except Exception as e:
        print(f"An error occurred: {e}")
        return None 

def get_historical_data(ticker_symbol, period="5y", interval="1d"):
    """
    Fetches historical stock data for a given ticker symbol.

    """
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period=period, interval=interval)
        return hist

    except Exception as e:
        print(f"An error occurred: {e}")
        return None 

def download_stock_data(ticker_symbol, filename):
    """
    Downloads historical stock data and saves it to a CSV file.

    """

    data = yf.download([ticker_symbol], start="2020-01-01", end=today, auto_adjust=False)
    if data is not None:
        data.to_csv(filename)
        df = pd.read_csv(filename,skiprows=(1,2))
        df.rename(columns={'Price': 'Date'}, inplace=True)
        df.insert(0, 'Stock', ticker)
        df.to_csv(filename, index=False)
        print(f"Historical data for {ticker_symbol} saved to {filename}")
    else:
        print("Failed to download or save historical data.")

def calc_rsi(over: pd.Series, fn_roll: Callable, length=14) -> pd.Series:
    # Get the difference in price from previous step
    delta = over.diff()
    # Get rid of the first row, which is NaN since it did not have a previous row to calculate the differences
    delta = delta[1:] 

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.clip(lower=0), delta.clip(upper=0).abs()

    roll_up, roll_down = fn_roll(up), fn_roll(down)
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Avoid division-by-zero if `roll_down` is zero
    # This prevents inf and/or nan values.
    rsi[:] = np.select([roll_down == 0, roll_up == 0, True], [100, 0, rsi])
    # rsi = rsi.case_when([((roll_down == 0), 100), ((roll_up == 0), 0)])  # This alternative to np.select works only for pd.__version__ >= 2.2.0.
    rsi.name = 'rsi'

    # Assert range
    valid_rsi = rsi[length - 1:]
    assert ((0 <= valid_rsi) & (valid_rsi <= 100)).all()
    # Note: rsi[:length - 1] is excluded from above assertion because it is NaN for SMA.

    return rsi

def calculate_ema(data_series: pd.Series, window: int) -> pd.Series: return data_series.ewm(span=window, adjust=False, min_periods=window).mean()

def calculate_vroc(data: pd.Series, length=20, upper_limit=400, lower_limit=-200):
    past = data.shift(length)
    raw_vroc = ((data - past) / past) * 100
    # Handle divisions by zero or missing data
    raw_vroc = raw_vroc.replace([pd.NA, pd.NaT, float('inf'), -float('inf')], pd.NA)
    clipped_vroc = raw_vroc.clip(lower_limit, upper_limit)
    return clipped_vroc

def weightedmovingaverage(close, period):
    wma = []
    weights = np.arange(1, period + 1)
    for i in range(len(close)):
        if i + 1 < period:
            wma.append(np.nan)
        else:
            window = close[i - period + 1 : i + 1]
            wma_val = np.dot(window, weights) / weights.sum()
            wma.append(wma_val)
    return np.array(wma)

def calculate_rvi(data, period=10):
    close_open_diff = data['Close'] - data['Open']
    high_low_diff = data['High'] - data['Low']

    numerator = close_open_diff.rolling(window=period).mean()
    denominator = high_low_diff.rolling(window=period).mean()

    rvi = numerator / denominator
    return rvi


if __name__ == "__main__":  
    ticker = "AAPL"
    store_file_name = f"{ticker}_historical_data.csv"
    download_stock_data(ticker, store_file_name)

    print(f"successfully download {ticker}:")
    
    # start analysis
    data = pd.read_csv(store_file_name)
    close = data['Adj Close']
    rsi14_rma = calc_rsi(close, lambda s: s.ewm(alpha=1 / 14).mean(), length=14) 
    data['RSI14'] = rsi14_rma
    rsi28_rma = calc_rsi(close, lambda s: s.ewm(alpha=1 / 28).mean(), length=28)
    data['RSI28'] = rsi28_rma
    rsi50_rma = calc_rsi(close, lambda s: s.ewm(alpha=1 / 50).mean(), length=50)
    data['RSI50'] = rsi50_rma
    ema_20 = calculate_ema(data['Close'], window=20)
    data['EMA20'] = ema_20
    ema_60 = calculate_ema(data['Close'], window=60)
    data['EMA50'] = ema_60
    ema_90 = calculate_ema(data['Close'], window=90)
    data['EMA90'] = ema_90
    ema_12 = calculate_ema(data['Close'], window=12)
    data['EMA12'] = ema_12
    ema_26 = calculate_ema(data['Close'], window=26)
    data['EMA26'] = ema_26
    data['MACD'] = ema_12 - ema_26

    data['AROON_bull_25'] = 100 * data.High.rolling(25).apply(lambda x: x.argmax()) / (25)
    data['AROON_bear_25'] = 100 * data.Low.rolling(25).apply(lambda x: x.argmin()) / (25)
    data['AROON_bull_50'] = 100 * data.High.rolling(50).apply(lambda x: x.argmax()) / (50)
    data['AROON_bear_50'] = 100 * data.Low.rolling(50).apply(lambda x: x.argmin()) / (50)
    data['VROC_20'] = calculate_vroc(data['Volume'])
    data['daily_return'] = close.pct_change(1)
    data_dji = yf.download('^DJI', start="2020-01-01", end=today, auto_adjust=False)
    dji_close = data_dji['Adj Close']
    data['DJI_Close'] = dji_close.values
    data['WMA_14'] = weightedmovingaverage(close.values, 14)
    data['WMA_20'] = weightedmovingaverage(close.values, 20)
    data['WMA_200'] = weightedmovingaverage(close.values, 200)
    data['STD_20'] = data['Adj Close'].rolling(window=20).std()
    data['SMA_20'] = close.rolling(window=20).mean()
    data['UB'] = data['SMA_20'] + 2 * data['STD_20']
    data['LB'] = data['SMA_20'] - 2 * data['STD_20']
    data['RVI_10'] = calculate_rvi(data, period=10)
    data.to_csv(store_file_name,index=False)
