import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException

def stock_data_yfinance(ticker: str ='', period: str = '', interval: str = ''):
    try:
        data = yf.download(
            ticker=ticker,
            period=period, 
            interval=interval,
            progress=False,
            auto_adjust=True,
            )
        if data.empty:
            return HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        data = data.reset_index()
        
        
        
        return data
    
    except Exception as e:
        print("ERROR:", str(e))
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")
     