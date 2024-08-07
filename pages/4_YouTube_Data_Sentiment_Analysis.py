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
from Project.Project_4.EDA_Process import exploring_data_analysis_process
from Project.Project_4.Sentiment_Analysis_Process import SentimentAnalysis
from Project.Project_4.In_depth_Sentiment_Analysis_Comments import YouTubeIndepthSentimentAnalysis
import time
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


load_dotenv("../.env",override=True)
pool = ThreadPool(processes=1)

st.set_page_config(page_title="YouTube Data Sentiment Analysis", page_icon=":bar_chart:")
st.sidebar.header("Project NolanM")
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = "./styles/main.css"
PROCESS_TIME = 15
PROCESS_TIME_FIVE_SEC = 5

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

# Hero Section
st.title("YouTube Data Sentiment Analysis - NolanM")
st.markdown(
    """
    **Objective**: The goal of this project is to integrate big data workflows, including data extraction, analysis, and visualization, using YouTube as a data source. You will explore various aspects of YouTube analytics, such as view trends, content topics, and viewer sentiment, to gain insights into content strategy and audience engagement.
    """
)

if "youtube_channel" not in st.session_state or "df" not in st.session_state or "number_of_indept_video" not in st.session_state:
    st.session_state.youtube_channel = None
    st.session_state.df = None
    st.number_of_indept_video = 5

input_channel = st.text_input("Please enter the YouTube channel name: ")
submit_youtube_analysis = st.button("Start Analysis YouTube Channel ...")

if submit_youtube_analysis:
    # Store the channel name in the session state
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
    #st.write(df.description())

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
    # st.write(df.columns)
    df_sort_by_time = df.copy()
    df_sort_by_time['published'] = pd.to_datetime(df_sort_by_time['published'], utc=True)
    df_sort_by_time = df_sort_by_time.sort_values(by='published')
    st.write("---")
    views_trend = df_sort_by_time.resample('ME', on='published').sum()['views_count']
    fig_views_trend = px.line(views_trend, 
                          title='Trend of Views Over Time',
                          labels={'value': 'Total Views', 'published': 'Time'})
    st.plotly_chart(fig_views_trend)

    publish_freq = df.resample('ME', on='published').size().reset_index(name="count")
    fig_publish_freq = px.bar(publish_freq, 
                            x='published', y='count',
                            title='Number of Videos Published Each Month',
                            labels={'count': 'Number of Videos', 'published': 'Time'})
    st.plotly_chart(fig_publish_freq)
    st.write("---")
    exploring_data_analysis_process(df, df_sort_by_time)

    # Sentiment Analysis
    st.markdown("""
        ## Sentiment Analysis of Video Titles
        **a)** Perform sentiment analysis on video titles using NLP TextBlob Python Library to categorize
        them into Positive, Neutral, and Negative.

        **b)** Deliverable: A pie chart showing the sentiment distribution and an analysis of content strategy
        or viewer engagement.
    """)

    sentiment_analysis = SentimentAnalysis(df)
    async_result_sentiment = pool.apply_async(sentiment_analysis.data_aggregation_and_summary_statistics, args=(df,))
    bar = st.progress(0)
    per = PROCESS_TIME_FIVE_SEC / 100
    for i in range(100):
        time.sleep(per)
        bar.progress(i + 1)
    df_sentiment = async_result_sentiment.get()
    st.write(df_sentiment.head(10))
    sentiment_analysis.visualize_statistic_distribution_data()

    # In-depth Sentiment Analysis of Comments
    st.markdown("""
        ## In-depth Sentiment Analysis of Comments
        **a)** Identify the top 5 videos with the maximum comment count and display them.

        **b)** Select the video with the maximum comment count and perform detailed sentiment analysis
        on its comments to understand public opinion.

        **c)** Fetch the comments, analyze the sentiments, and categorize them into Positive, Negative, and
        Neutral. Create and display the pie chart for sentiment distribution.

        **d)** Create and display a word cloud for each sentiment category.

        **e)** Deliverable
        - Pie chart for sentiment distribution.
        - word cloud for each sentiment category.
        - comprehensive insights for each analysis step.
    """)

    #number_of_indept_video = st.number_input("Enter the number of videos to display", min_value=5, step=1, value="min")
    st.session_state.number_of_indept_video = 5
    st.write(f"Analyzing the top {st.session_state.number_of_indept_video} videos with the maximum comment count...")
    Indepth_sentiment_analysis = YouTubeIndepthSentimentAnalysis(df_sentiment,st.session_state.number_of_indept_video)
    async_result_sentiment_indepth = pool.apply_async(Indepth_sentiment_analysis.analyze_sentiment_video_with_maximum_comment_count,)
    bar = st.progress(0)
    per = PROCESS_TIME_FIVE_SEC / 100
    for i in range(100):
        time.sleep(per)
        bar.progress(i + 1)
    df_sentiment_indepth = async_result_sentiment_indepth.get()
    Indepth_sentiment_analysis.chart_sentiment_category_distribution()
    Indepth_sentiment_analysis.display_word_cloud_by_sentiment_category()
    




    
