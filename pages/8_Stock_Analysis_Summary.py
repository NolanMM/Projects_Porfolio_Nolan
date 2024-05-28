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
        st.write("Valid Ticker")
    else:
        st.error("Text input does not match the required format (AAPL)")
