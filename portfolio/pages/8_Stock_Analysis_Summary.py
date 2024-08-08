import streamlit as st
import pandas as pd
import numpy as np
import re
from dotenv import load_dotenv
import os
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

load_dotenv(override=True)
alpha_vantage_api = os.getenv("ALPHAVANTAGE_API")

st.set_page_config(
    page_title="Stock Analysis and Visualization - NolanM",
    page_icon=":shark:",
)
css_file = "./styles/main.css"
st.sidebar.header("Project NolanM")
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

st.markdown(
    """
        <style>
            .st-emotion-cache-13ln4jf.ea3mdgi5 {
                max-width: 1200px;
            }
        </style>
    """, unsafe_allow_html=True)

st.title("Comprehensive Stock Analysis and Visualization - NolanM")

st.write("")

st.markdown("""
    ## **Objective**
    The objective of this application is to provide users with a comprehensive platform for analyzing and visualizing stock market data. By integrating various data sources and visualization tools, the app aims to help users make informed investment decisions. Including:
    - Alpha Vantage API: A service providing financial data such as stock prices, forex rates, and fundamental data through a free API.
    - StockNews API: A service providing financial news data through an API.
    - yfinance: A library that allows easy access to historical market data from Yahoo Finance.
""")

if "ticket_input" not in st.session_state:
    st.session_state.ticket_input = ""

if "startdate" not in st.session_state:
    st.session_state.startdate = ""

if "enddate" not in st.session_state:
    st.session_state.enddate = ""

text_input = st.text_input("Ticker")

pattern = re.compile(r"^[A-Z]{1,5}$")

startdate = st.date_input("Select start date")
enddate = st.date_input("Select end date")

if st.button("Analysis"):
    if re.match(pattern, text_input):
        st.session_state.ticket_input = text_input
        st.session_state.startdate = startdate
        st.session_state.enddate = enddate
        
        data = yf.download(text_input, start=st.session_state.startdate, end=st.session_state.enddate)

        summary_stats, fundamentals_data, news, reddit_news = st.tabs(["Summary Statistic","Fundamentals", "Top News", "Reddit News"])

        df = pd.DataFrame(data)
        df = df.rename_axis("Date").reset_index()

        with summary_stats:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Adj Close Price Over Time")
                fig = px.line(df, x=df.Date, y=df["Adj Close"], hover_data={"Date": "|%B %d, %Y"}, title=text_input)
                fig.update_xaxes(dtick="M1", tickformat="%b\n%Y")
                st.plotly_chart(fig)
            
            with col2:
                st.subheader("Volume Over Time")
                fig = px.line(df, x=df.Date, y=df["Volume"], hover_data={"Date": "|%B %d, %Y"}, title=text_input)
                fig.update_xaxes(dtick="M1", tickformat="%b\n%Y")
                st.plotly_chart(fig)
            

            st.header("Price Movement")
            data_ = data.copy()
            
            data_["% Change"] = data_["Adj Close"] / data_["Adj Close"].shift(1) - 1
            data_.dropna(inplace=True)
            
            sec_2_col1, sec_2_col2 = st.columns(2)

            _data_ = data.copy()
            df_reset = _data_.reset_index()
            df_drop = df_reset.drop(['Date','Adj Close'], axis = 1)

            with sec_2_col1:
                st.subheader("Moving Average Chart Over 100 Days")
                ma100 = df_drop.Close.rolling(100).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_drop.index, y=df_drop['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=df_drop.index, y=ma100, mode='lines', name='100-Day Moving Average', line=dict(color='red')))
                fig.update_layout(
                    title='Stock Price and 100-Day Moving Average',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    legend_title='Legend',
                    template='plotly_white'
                )
                st.plotly_chart(fig)

                ma100_date = data.Close.rolling(100).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=data.index, y=ma100_date, mode='lines', name='100-Day Moving Average', line=dict(color='red')))
                fig.update_layout(
                    title='Stock Price and 100-Day Moving Average - Date',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    legend_title='Legend',
                    template='plotly_white'
                )
                st.plotly_chart(fig)


            with sec_2_col2:
                st.subheader("Moving Average Chart Over 200 Days")
                ma200 = df_drop.Close.rolling(200).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_drop.index, y=df_drop['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=df_drop.index, y=ma200, mode='lines', name='200-Day Moving Average', line=dict(color='red')))
                fig.update_layout(
                    title='Stock Price and 200-Day Moving Average',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    legend_title='Legend',
                    template='plotly_white'
                )
                st.plotly_chart(fig)

                ma200_date = data.Close.rolling(200).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=data.index, y=ma100_date, mode='lines', name='200-Day Moving Average', line=dict(color='red')))
                fig.update_layout(
                    title='Stock Price and 200-Day Moving Average - Date',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    legend_title='Legend',
                    template='plotly_white'
                )
                st.plotly_chart(fig)

            annual_returns = data_["% Change"].mean() * 252 * 100
            st.write(f"Annual Returns: {annual_returns:.2f}%")
            std_ = np.std(data_["% Change"]) * np.sqrt(252) * 100
            st.write(f"Standard Deviation: {std_:.2f} %")
            st.write("Risk Adjusted Returns: ", annual_returns / std_)
        
        # with fundamentals_data:
        #     fd = FundamentalData(key=ALPHAVANTAGE_API, output_format="pandas")
        #     st.subheader("Balance Sheet")
        #     balance_sheet = fd.get_balance_sheet_annual(text_input)[0]
        #     balance_sheet_ = balance_sheet.T[2:]
        #     balance_sheet_.columns = list(balance_sheet_.iloc[0])
        #     st.write(balance_sheet_)
        #     # save balance sheet to csv
        #     balance_sheet_.to_csv(f"./data/Project_8/{text_input}_balance_sheet.csv", index=False)

        #     st.subheader("Income Statement")
        #     income_statement = fd.get_income_statement_annual(text_input)[0]
        #     income_statement_ = income_statement.T[2:]
        #     income_statement_.columns = list(income_statement_.iloc[0])
        #     st.write(income_statement_)
        #     # save income statement to csv
        #     income_statement_.to_csv(f"./data/Project_8/{text_input}_income_statement.csv", index=False)

        #     st.subheader("Cash Flow")
        #     cash_flow = fd.get_cash_flow_annual(text_input)[0]
        #     cash_flow_ = cash_flow.T[2:]
        #     cash_flow_.columns = list(cash_flow_.iloc[0])
        #     st.write(cash_flow_)
        #     # save cash flow to csv
        #     cash_flow_.to_csv(f"./data/Project_8/{text_input}_cash_flow.csv", index=False)


        with news:
            st.header("Top News")
            sn = StockNews(text_input, save_news=False)
            df_news = sn.read_rss()
            for i in range(10):
                st.subheader(df_news["title"][i])
                st.write(df_news["published"][i])
                st.write(df_news["summary"][i])
                title_sentiment = df_news["sentiment_title"][i]
                st.write(f"Title Sentiment: {title_sentiment}")
                news_sentiment = df_news["sentiment_summary"][i]
                st.write(f"News Sentiment: {news_sentiment}")
        
        with reddit_news:
            st.header("Reddit News")

    else:
        st.error("Text input does not match the required format (AAPL)")

