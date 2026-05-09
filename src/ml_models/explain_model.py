import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def explain_model():

    # Load data
    df = pd.read_csv("data/processed/btc_features.csv")

    # Target
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

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Train model
    model = XGBClassifier()

    model.fit(X_train, y_train)

    # SHAP Explainer
    explainer = shap.Explainer(model)

    shap_values = explainer(X_test)

    # Summary plot
    shap.summary_plot(
        shap_values,
        X_test
    )

    plt.show()


if __name__ == "__main__":
    explain_model()