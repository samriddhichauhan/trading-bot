import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

def train_model():

    # Load processed data
    df = pd.read_csv("data/processed/btc_features.csv")

    # Create target column
    df['target'] = (
        df['close'].shift(-1) > df['close']
    ).astype(int)

    # Remove missing values
    df.dropna(inplace=True)

    # Features
    X = df[
    [
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
]

    # Labels
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # Create model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    # Train model
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Predictions
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nModel Accuracy: {accuracy:.2f}\n")

    print("Classification Report:\n")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    train_model()