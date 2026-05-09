import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.ensemble import VotingClassifier


def ensemble_model():

    # Load data
    df = pd.read_csv("data/processed/btc_features.csv")

    # Create target
    df['target'] = (
        df['close'].shift(-1) > df['close']
    ).astype(int)

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

    # ==================================
    # MODELS
    # ==================================

    xgb = XGBClassifier()

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    lr = LogisticRegression()

    # ==================================
    # ENSEMBLE
    # ==================================

    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('rf', rf),
            ('lr', lr)
        ],
        voting='soft'
    )

    # Train
    ensemble.fit(X_train, y_train)

    # Predict
    predictions = ensemble.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n========== ENSEMBLE AI RESULTS ==========\n")

    print(f"Ensemble Accuracy: {accuracy:.2f}")

    print("\n=========================================\n")


if __name__ == "__main__":
    ensemble_model()