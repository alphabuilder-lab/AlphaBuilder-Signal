from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import traceback

app = FastAPI(
    title="AlphaBuilder-Dash API",
    description="API for stock price data using Yahoo Finance",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Hello from AlphaBuilder-Dash!"}


@app.get("/stock/{ticker}")
def get_stock_data(ticker: str, period: str = "6mo", interval: str = "1d"):
    """
    Fetch historical price data for a given ticker.
    Example: /stock/AAPL?period=1y&interval=1d
    """

    try:
        # Download data
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

        df = df.reset_index()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        date_col = next((c for c in df.columns if "Date" in c), None)
        close_col = next((c for c in df.columns if "Close" in c), None)

        if not date_col or not close_col:
            raise HTTPException(status_code=500, detail=f"Unexpected columns: {list(df.columns)}")

        close_series = df[close_col]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

        data = {
            "ticker": ticker.upper(),
            "dates": df[date_col].astype(str).tolist(),
            "close": close_series.astype(float).tolist(),
            "meta": {
                "period": period,
                "interval": interval,
                "count": len(df)
            }
        }

        print(f"Data fetched successfully for {ticker}: {len(df)} rows")
        return data

    except Exception as e:
        print("ERROR:", str(e))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")
