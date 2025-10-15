################################
#  get_stock.py                #
################################

import yfinance as yf
import datetime
import numpy as np

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
    data = yf.download([ticker_symbol], start="2020-01-01", end=today)

    if data is not None:
        data.to_csv(filename)
        print(f"Historical data for {ticker_symbol} saved to {filename}")
    else:
        print("Failed to download or save historical data.")
        
if __name__ == "__main__":
    ticker = "AAPL"
    stock_data = get_stock_info(ticker)
    historical_data = get_historical_data(ticker)
    data = download_stock_data(ticker, "AAPL_historical_data.csv")
    if stock_data:
        print(f"Stock information for {ticker}:")
        for key, value in stock_data.items():
            print(f"{key}: {value}")