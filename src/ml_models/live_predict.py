import pandas as pd
import yfinance as yf

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def live_prediction():

    # ====================================
    # LOAD HISTORICAL TRAINING DATA
    # ====================================

    df = pd.read_csv("data/processed/btc_features.csv")

    # Create target
    df['target'] = (
        df['close'].shift(-1) > df['close']
    ).astype(int)

    df.dropna(inplace=True)

    # Features
    features = [
        'rsi',
        'ema_20',
        'macd',
        'bb_high',
        'bb_low',
        'atr',
        'returns',
        'volatility',
        'volume_change'
    ]

    X = df[features]
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # ====================================
    # TRAIN MODEL
    # ====================================

    model = XGBClassifier()

    model.fit(X_train, y_train)

    # ====================================
    # DOWNLOAD LATEST BTC DATA
    # ====================================

    live_df = yf.download(
        "BTC-USD",
        period="7d",
        interval="1h"
    )

    live_df.reset_index(inplace=True)

    # Lowercase columns
    live_df.columns = [col[0].lower() for col in live_df.columns]

    # Indicators
    live_df['rsi'] = RSIIndicator(
        close=live_df['close']
    ).rsi()

    live_df['ema_20'] = EMAIndicator(
        close=live_df['close'],
        window=20
    ).ema_indicator()

    macd = MACD(close=live_df['close'])

    live_df['macd'] = macd.macd()

    bb = BollingerBands(close=live_df['close'])

    live_df['bb_high'] = bb.bollinger_hband()
    live_df['bb_low'] = bb.bollinger_lband()

    atr = AverageTrueRange(
        high=live_df['high'],
        low=live_df['low'],
        close=live_df['close']
    )

    live_df['atr'] = atr.average_true_range()

    live_df['returns'] = (
        live_df['close'].pct_change()
    )

    live_df['volatility'] = (
        live_df['high'] - live_df['low']
    ) / live_df['close']

    live_df['volume_change'] = (
        live_df['volume'].pct_change()
    )

    # Clean data
    live_df.replace(
        [float('inf'), -float('inf')],
        pd.NA,
        inplace=True
    )

    live_df.dropna(inplace=True)

   # ====================================
    # LATEST DATA ROW
    # ====================================

    latest_data = live_df[features].iloc[-1:]

    # Convert all features to numeric
    latest_data = latest_data.apply(
        pd.to_numeric,
        errors='coerce'
    )

    # Fill missing values
    latest_data = latest_data.fillna(0)

    # Prediction
    prediction = model.predict(latest_data)[0]
    probability = model.predict_proba(latest_data)[0]

    confidence = max(probability) * 100

    # ====================================
    # PRINT RESULTS
    # ====================================

    print("\n========== LIVE AI PREDICTION ==========\n")

    if prediction == 1:
        print("📈 AI Signal: BUY")
    else:
        print("📉 AI Signal: SELL")

    print(f"\nConfidence: {confidence:.2f}%")

    print("\n========================================\n")


if __name__ == "__main__":
    live_prediction()