################################
#  get_stock.py                #
################################

import yfinance as yf

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
    

if __name__ == "__main__":
    ticker = "AAPL"
    stock_data = get_stock_info(ticker)
    if stock_data:
        print(f"Stock information for {ticker}:")
        for key, value in stock_data.items():
            print(f"{key}: {value}")