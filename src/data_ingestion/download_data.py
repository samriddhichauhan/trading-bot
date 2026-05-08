import yfinance as yf
import pandas as pd

def download_data():

    df = yf.download(
        "BTC-USD",
        period="2y",
        interval="1h"
    )

    df.to_csv("data/raw/btc_data.csv")

    print("Data Downloaded Successfully!")
    print(df.head())

if __name__ == "__main__":
    download_data()