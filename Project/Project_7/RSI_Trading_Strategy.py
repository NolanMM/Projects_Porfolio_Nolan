import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from datetime import datetime

class RSIAnalyzer:
    def __init__(self, asset):
        self.asset = asset
        self.df = self.RSIcalc()

    def RSIcalc(self):
        end_date = datetime.now().strftime('%Y-%m-%d')
        df = yf.download(self.asset, start='2010-01-01', end=end_date)
        df['MA200'] = df['Adj Close'].rolling(window=200).mean()
        df['price change'] = df['Adj Close'].pct_change()
        df['Upmove'] = df['price change'].apply(lambda x: x if x > 0 else 0)
        df['Downmove'] = df['price change'].apply(lambda x: abs(x) if x < 0 else 0)
        df['avg Up'] = df['Upmove'].ewm(span=20).mean()
        df['avg Down'] = df['Downmove'].ewm(span=20).mean()
        df = df.dropna()
        df['RS'] = df['avg Up'] / df['avg Down']
        df['RSI'] = df['RS'].apply(lambda x: 100 - (100 / (x + 1)))
        df.loc[(df['Adj Close'] > df['MA200']) & (df['RSI'] < 30), 'Buy'] = 'Yes'
        df.loc[(df['Adj Close'] < df['MA200']) | (df['RSI'] > 30), 'Buy'] = 'No'
        return df

    def get_signals(self):
        Buying_dates = []
        Selling_dates = []
        df = self.df
        for i in range(len(df) - 11):
            if "Yes" in df['Buy'].iloc[i]:
                Buying_dates.append(df.iloc[i + 1].name)
                for j in range(1, 11):
                    if df['RSI'].iloc[i + j] > 40:
                        Selling_dates.append(df.iloc[i + j + 1].name)
                        break
                    elif j == 10:
                        Selling_dates.append(df.iloc[i + j + 1].name)
        return Buying_dates, Selling_dates

    def visualize_data(self):
        buy, sell = self.get_signals()
        df = self.df

        buy_signals = df.loc[buy]
        sell_signals = df.loc[sell]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Adj Close'))
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Adj Close'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', color='green', size=10)))
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Adj Close'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', color='red', size=10)))

        fig.update_layout(title=f"{self.asset} RSI Signals",
                          xaxis_title='Date',
                          yaxis_title='Price',
                          legend_title='Legend')

        st.plotly_chart(fig)

# ticker = 'AAPL'
# analyzer = RSIAnalyzer(ticker)
# analyzer.visualize_data()
