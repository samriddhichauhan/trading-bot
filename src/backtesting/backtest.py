import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def backtest():

    # Load processed data
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

    # Labels
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # Fill missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Train model
    model = XGBClassifier()

    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Backtesting dataframe
    backtest_df = df.iloc[-len(predictions):].copy()

    backtest_df['prediction'] = predictions

    # Strategy returns
    backtest_df['strategy_return'] = (
        backtest_df['returns'] *
        backtest_df['prediction']
    )

    # Cumulative returns
    backtest_df['cumulative_market'] = (
        1 + backtest_df['returns']
    ).cumprod()

    backtest_df['cumulative_strategy'] = (
        1 + backtest_df['strategy_return']
    ).cumprod()

    # ==================================
    # QUANT METRICS
    # ==================================

    # Total returns
    market_return = (
        backtest_df['cumulative_market'].iloc[-1] - 1
    ) * 100

    strategy_return = (
        backtest_df['cumulative_strategy'].iloc[-1] - 1
    ) * 100

    # Win rate
    wins = backtest_df[
        backtest_df['strategy_return'] > 0
    ]

    win_rate = (
        len(wins) / len(backtest_df)
    ) * 100

    # Volatility
    volatility = (
        backtest_df['strategy_return'].std()
    ) * np.sqrt(252)

    # Sharpe Ratio
    sharpe_ratio = (
        backtest_df['strategy_return'].mean() /
        backtest_df['strategy_return'].std()
    ) * np.sqrt(252)

    # Max Drawdown
    rolling_max = (
        backtest_df['cumulative_strategy']
        .cummax()
    )

    drawdown = (
        backtest_df['cumulative_strategy'] -
        rolling_max
    ) / rolling_max

    max_drawdown = drawdown.min() * 100

    # ==================================
    # PRINT RESULTS
    # ==================================

    print("\n========== QUANT BACKTEST RESULTS ==========\n")

    print(f"Market Return: {market_return:.2f}%")
    print(f"Strategy Return: {strategy_return:.2f}%")

    print(f"\nWin Rate: {win_rate:.2f}%")

    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    print(f"Volatility: {volatility:.4f}")

    print(f"Max Drawdown: {max_drawdown:.2f}%")

    print("\n============================================\n")


if __name__ == "__main__":
    backtest()