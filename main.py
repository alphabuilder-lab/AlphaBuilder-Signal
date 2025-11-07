from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import traceback
import numpy as np

app = FastAPI(
    title="AlphaBuilder-Dash API",
    description="API for stock price data using Yahoo Finance",
    version="1.0",
)

origins = [
    "http://localhost:3000",  
    "https://alphabuilder.xyz",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Hello from AlphaBuilder-Dash!"}


@app.get("/stock/{ticker}")
def get_stock_data(ticker: str, period: str = "20y", interval: str = "1d"):
    """
    Fetch historical price data for a given ticker.
    Example: /stock/AAPL?period=1y&interval=1d
    """

    try:
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

        date_col = "Date"
        close_col = "Close"
        if date_col not in df.columns or close_col not in df.columns:
            raise HTTPException(status_code=500, detail=f"Missing columns in {ticker} data")

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


@app.get("/ivs")
def generate_ivs(
    spot: float = 100.0,
    maturities: int = 10,
    strikes: int = 20
):
    """
    Simulate a smooth implied volatility surface (moneyness × maturity)
    """

    T = np.linspace(0.1, 2.0, maturities)  # 0.1Y to 2Y maturities
    K = np.linspace(60, 140, strikes)       # strike range
    F = spot                                # forward ≈ spot
    M = K / F                               # moneyness

    # Example synthetic IV model: smile + term structure
    base_vol = 0.20 + 0.05 * np.exp(-T)[:, None]
    smile_term = 0.1 * (M[None, :] - 1) ** 2
    iv = base_vol + smile_term

    # Convert to long-format DataFrame
    df = pd.DataFrame(
        [(float(t), float(k), float(m), float(v))
         for t, row in zip(T, iv) for k, m, v in zip(K, M, row)],
        columns=["maturity", "strike", "moneyness", "iv"]
    )

    return {
        "meta": {
            "spot": spot,
            "maturities": maturities,
            "strikes": strikes,
        },
        "data": df.to_dict(orient="records")
    }