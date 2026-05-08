import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange


def add_indicators():

    # Load dataset
    df = pd.read_csv("data/raw/btc_data.csv")

    # Lowercase columns
    df.columns = [col.lower() for col in df.columns]

    # Convert numeric columns
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # =========================
    # TECHNICAL INDICATORS
    # =========================

    # RSI
    df['rsi'] = RSIIndicator(
        close=df['close']
    ).rsi()

    # EMA
    df['ema_20'] = EMAIndicator(
        close=df['close'],
        window=20
    ).ema_indicator()

    # MACD
    macd = MACD(close=df['close'])

    df['macd'] = macd.macd()

    # Bollinger Bands
    bb = BollingerBands(close=df['close'])

    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()

    # ATR
    atr = AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close']
    )

    df['atr'] = atr.average_true_range()

    # Returns
    df['returns'] = df['close'].pct_change()

    # Volatility
    df['volatility'] = (
        df['high'] - df['low']
    ) / df['close']

    # Volume Change
    df['volume_change'] = df['volume'].pct_change()

    # Replace infinite values
    df.replace(
        [float('inf'), -float('inf')],
        pd.NA,
        inplace=True
    )

    # Remove missing values
    df.dropna(inplace=True)

    # Save processed data
    df.to_csv(
        "data/processed/btc_features.csv",
        index=False
    )

    print("Advanced Indicators Added!")
    print(df.head())


if __name__ == "__main__":
    add_indicators()