import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Quant Dashboard",
    layout="wide"
)

st.title("📈 AI Quant Research Platform")


# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("data/processed/btc_features.csv")

# Create target
df['target'] = (
    df['close'].shift(-1) > df['close']
).astype(int)

df.dropna(inplace=True)


# =====================================
# FEATURES
# =====================================

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


# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


# =====================================
# MODEL
# =====================================

model = XGBClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# =====================================
# BACKTEST
# =====================================

backtest_df = df.iloc[-len(predictions):].copy()

backtest_df['prediction'] = predictions

backtest_df['strategy_return'] = (
    backtest_df['returns'] *
    backtest_df['prediction']
)

backtest_df['cumulative_market'] = (
    1 + backtest_df['returns']
).cumprod()

backtest_df['cumulative_strategy'] = (
    1 + backtest_df['strategy_return']
).cumprod()


# =====================================
# METRICS
# =====================================

market_return = (
    backtest_df['cumulative_market'].iloc[-1] - 1
) * 100

strategy_return = (
    backtest_df['cumulative_strategy'].iloc[-1] - 1
) * 100

win_rate = (
    len(
        backtest_df[
            backtest_df['strategy_return'] > 0
        ]
    ) / len(backtest_df)
) * 100


# =====================================
# METRIC CARDS
# =====================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "📊 Market Return",
    f"{market_return:.2f}%"
)

col2.metric(
    "🤖 Strategy Return",
    f"{strategy_return:.2f}%"
)

col3.metric(
    "🎯 Win Rate",
    f"{win_rate:.2f}%"
)


# =====================================
# CHART
# =====================================

st.subheader("📈 Strategy vs Market Performance")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    backtest_df['cumulative_market'],
    label="Market Return"
)

ax.plot(
    backtest_df['cumulative_strategy'],
    label="AI Strategy Return"
)

ax.legend()

st.pyplot(fig)


# =====================================
# DATA PREVIEW
# =====================================

st.subheader("📋 Latest Dataset")

st.dataframe(
    backtest_df.tail(20)
)