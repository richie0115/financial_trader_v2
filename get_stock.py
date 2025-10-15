################################
#  get_stock.py                #
################################

import yfinance as yf
import datetime
import numpy as np
import pandas as pd

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
    today = datetime.date.today()
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

if __name__ == "__main__":
    ticker = "AAPL"
    store_file_name = f"{ticker}_historical_data.csv"
    download_stock_data(ticker, store_file_name)

    print(f"successfully download {ticker}:")
    
    # start analysis
    data = pd.read_csv(store_file_name, parse_dates=True)
    close = data['Adj Close']
    print(close)
