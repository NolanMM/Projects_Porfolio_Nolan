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
import threading
from Project.Project_4.Youtube_Analysis_Services import data_extraction_enriching_process
import time

load_dotenv("../.env",override=True)

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

if "youtube_channel" not in st.session_state:
    st.session_state.youtube_channel = None

input_channel = st.text_input("Please enter the YouTube channel name: ")
submit_youtube_analysis = st.button("Start Analysis YouTube Channel ...")

if submit_youtube_analysis:
    st.session_state.youtube_channel = "@" + input_channel
    st.write(f"Analyzing data for {st.session_state.youtube_channel}...")
    # Make another thread to process the data and collect the results
    df = pd.DataFrame()
    thread = threading.Thread(target=data_extraction_enriching_process, args=(st.session_state.youtube_channel, df, ))
    thread.start()
    bar = st.progress(0)
    per = PROCESS_TIME / 100
    for i in range(100):
        time.sleep(per)
        bar.progress(i + 1)
    
    thread.join()
    st.write("Data analysis completed!")
    st.write(df)

    
