import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from googleapiclient.discovery import build
import matplotlib.ticker as ticker
from textblob import TextBlob
from wordcloud import WordCloud
import streamlit as st
from dotenv import load_dotenv
import os
from pathlib import Path
import re
import requests
from multiprocessing.pool import ThreadPool
from Project.Project_4.Youtube_Analysis_Services import data_extraction_enriching_process
import time
import plotly.express as px

load_dotenv("../.env",override=True)
pool = ThreadPool(processes=1)

st.set_page_config(page_title="YouTube Data Sentiment Analysis", page_icon=":bar_chart:")

current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = "./styles/main.css"
PROCESS_TIME = 15

with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

# Hero Section
st.title("YouTube Data Sentiment Analysis")
st.markdown(
    """
    **Objective**: The goal of this project is to integrate big data workflows, including data extraction, analysis, and visualization, using YouTube as a data source. You will explore various aspects of YouTube analytics, such as view trends, content topics, and viewer sentiment, to gain insights into content strategy and audience engagement.
    """
)

if "youtube_channel" not in st.session_state or "df" not in st.session_state:
    st.session_state.youtube_channel = None
    st.session_state.df = None

input_channel = st.text_input("Please enter the YouTube channel name: ")
submit_youtube_analysis = st.button("Start Analysis YouTube Channel ...")

if submit_youtube_analysis:
    st.session_state.youtube_channel = "@" + input_channel
    st.write(f"Analyzing data for {st.session_state.youtube_channel}...")
    # Make another thread to process the data and collect the results
    async_result = pool.apply_async(data_extraction_enriching_process,args = (st.session_state.youtube_channel,))
    bar = st.progress(0)
    per = PROCESS_TIME / 100
    for i in range(100):
        time.sleep(per)
        bar.progress(i + 1)
    df = async_result.get()

    st.write(df)
    st.write(df.description())

    # EDA SECTION
    st.markdown(
        """
        ## EDA Process (Exploring Data Analysis) 
        - **View Trend Analysis**
        - **Publishing Frequency Analysis**

        **a)** Using Time Series Analysis method and visualize the data to analyze the trend of view over the time

        **b)** Plot the number of videos published each month to analyze content frequency to helps
        understand the channel's consistency and its potential impact on subscriber engagement and
        channel growth.

        **c)** Deliverable: A time-series plot and analysis of publishing patterns.
    """, unsafe_allow_html=True)

    df_sort_by_time = df.sort_values(by='published')
    fig = px.line(df, x=df_sort_by_time.published, y=df_sort_by_time.views_count, hover_data={"published": "|%B %d, %Y"}, title=st.session_state.youtube_channel)
    fig.update_xaxes(dtick="M1", tickformat="%b\n%Y")
    st.plotly_chart(fig)
    




    
