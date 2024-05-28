import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
import streamlit as st
from Project.Project_7.RSI_Trading_Strategy import RSIAnalyzer

st.set_page_config(page_title="RSI Trading Strategy and Backtest Implementation - NolanM", page_icon=":chart_with_upwards_trend:")
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

st.header("RSI Trading Strategy and Backtest Implementation - NolanM")

st.write("")

st.markdown("""
    ## The Strategy

    ___
    ### I. General Condition: The stock price is above its 200-day Moving average

    - #### Buy decision if:
        - ##### 10 period RSI of the stock is below 30
        - ##### -> Buy on the next's day open

    - #### Sell decision if:
        - ##### 10 period RSI is above 40 OR after 10 trading days
        - ##### -> Sell on the next day's open



    ___
    ### II. RSI Calculation

    - #### Step 1: Calculating Up and Down Moves
        - ##### Up moves:
            - ##### -> Take the daily return if return is positive
            - ##### -> Take 0 if return is negative or zero
        
        - ##### Down moves:
            - ##### -> Absolute value of daily return if return is negative
            - ##### -> Zero if return is positive or zeros
        
    - #### Step 2: Averaging Up and Downmoves
        - #### 1.Pick an average mothod:
            - ##### -> Using Simple Moving Average or Exponential moving average
        
        - #### 2. Using Wilder's smoothing Method which is the same as an exponential moving average but a different smoothing factor
            - ##### The Smoothing factor in exponential moving average: a = 2 / ( N + 1 )
            - ##### The Smoothing factor in WSM: a = 1 / N
            - ##### Because of there is no WSM function in Python to get the WSM alpha of 1/10 we would have to use a N of 19 in exponential moving average formula.

    - #### Step 3: RS and RSI calculation
        - ##### 1. RS -> Average Up Move / Average Down Move      
        - ##### 2. RSI -> 100 - 100 / ( 1 + RS )
""")

st.write("")

st.markdown("""
    ### I. Data Preparation
    ___

    - #### Step 1: Download the stock data
        - ##### -> Using wikipedia to get the list of S&P 500 companies
        - ##### -> Using yfinance library to download the stock data of the desired stock
        - ##### -> The stock data is downloaded from 2010 to 2021
""")
if "stock_data" not in st.session_state:
    st.session_state.stock_data = None
st.subheader("Using Wikipedia to get the list of S&P 500 companies")
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = tickers.Symbol.to_list()
tickers = [i.replace('.','-') for i in tickers]

code_wikipedia = """
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = tickers.Symbol.to_list()
    tickers = [i.replace('.','-') for i in tickers]
"""

st.code(code_wikipedia, language="python")
st.write(tickers)

the_stock = st.selectbox("Select the stock you want to analyze", tickers)
btn_prepare = st.button("Analysis the stock data...")

if btn_prepare:
    st.session_state.stock_data = the_stock
    st.write(f"Start Analyzing data for {st.session_state.stock_data}...")

    stock = RSIAnalyzer(the_stock)

    st.write("The stock data is downloaded and enriching successfully !!!")

    stock_data = stock.df
    st.write(stock_data)

    st.markdown("""
    ### Visualizing the RSI Signals
    ___
    """)

    stock.visualize_data()

    Buying_dates, Selling_dates = stock.get_signals()

    Profits = (stock_data.loc[Selling_dates].Open.values - stock_data.loc[Buying_dates].Open.values) / stock_data.loc[Buying_dates].Open.values


    wins_time = [i for i in Profits if i > 0]
    win_rate = len(wins_time)/len(Profits)

    st.write("The win rate of the strategy is: ", win_rate)
    #st.write("The profits of the strategy is: ", Profits)
    st.write("The average profits of the strategy is: ", Profits.mean())
    st.write("The total profits of the strategy is: ", Profits.sum())
    st.write("The total number of trades is: ", len(Profits))

    Buy_date_formatted = [ts.strftime('%d-%m-%Y') for ts in Buying_dates]
    Sell_date_formatted = [ts.strftime('%d-%m-%Y') for ts in Selling_dates]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("The buying dates: ")
        for Date in Buy_date_formatted:
            st.write(Date)
    with col2:
        st.subheader("The selling dates: ")
        for Date in Sell_date_formatted:
            st.write(Date)
        #st.table("The selling dates: ", Sell_date_formatted)


    






