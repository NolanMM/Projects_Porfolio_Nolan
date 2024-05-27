import os
import pandas as pd
from googleapiclient.discovery import build
from textblob import TextBlob
import plotly.express as px
from wordcloud import WordCloud
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

API_KEY___ = os.getenv("API_KEY___")

class YouTubeIndepthSentimentAnalysis:
    def __init__(self, videos_df_, count=5):
        self.service = build('youtube', 'v3', developerKey=API_KEY___)
        self.videos_df = videos_df_
        self.top_videos_by_count = self.identify_top_5_videos_by_count(count)
        self.comments_df = pd.DataFrame()
        self.video_id = None

    def identify_top_5_videos_by_count(self, count):
        # Sort the DataFrame based on the 'comment_count' column in descending order
        videos_df_sorted = self.videos_df.sort_values(by='comments_count', ascending=False)
        # Select the top 5 videos with the maximum comment count
        # Check if head is int and not empty
        top_videos = videos_df_sorted.head(count)
        self.top_videos_by_count = top_videos
        return top_videos

    def get_video_comments(self, **kwargs):
        comments = []
        results = self.service.commentThreads().list(**kwargs).execute()

        while results:
            for item in results['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            # Check if there are more pages
            if 'nextPageToken' in results:
                kwargs['pageToken'] = results['nextPageToken']
                results = self.service.commentThreads().list(**kwargs).execute()
            else:
                break

        return comments

    def get_sentiment(self, text):
        # Create a TextBlob object
        blob = TextBlob(text)
        # Return the polarity
        return blob.sentiment.polarity

    def categorize_sentiment(self, polarity):
        if polarity > 0:
            return 'Positive'
        elif polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'

    def analyze_sentiment_video_with_maximum_comment_count(self):
        # Fetch comments for a specific video
        video_id = self.top_videos_by_count.iloc[0]['video_id']
        comments = self.get_video_comments(part='snippet', videoId=video_id, textFormat='plainText')

        # Add sentiment analysis to the comments
        comments_sentiment = [{'comment': comment, 'sentiment_polarity': self.get_sentiment(comment)} for comment in comments]

        # Convert to DataFrame
        comments_df = pd.DataFrame(comments_sentiment)

        # Apply the categorization function
        comments_df['sentiment_category'] = comments_df['sentiment_polarity'].apply(self.categorize_sentiment)

        # Display the first few rows of the DataFrame
        self.comments_df = comments_df.copy()
        self.video_id = video_id
        return comments_df

    def chart_sentiment_category_distribution(self):
        sentiment_counts = self.comments_df['sentiment_category'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        # Pie chart
        fig_pie = px.pie(sentiment_counts, values='Count', names='Sentiment', title='Sentiment Category Distribution on Video Comments')
        st.plotly_chart(fig_pie)

        # Bar chart
        fig_bar = px.bar(sentiment_counts, x='Sentiment', y='Count', title='Sentiment Category Distribution on Video Comments', 
                        color='Sentiment', text='Count')
        st.plotly_chart(fig_bar)
    
    def display_word_cloud_by_sentiment_category(self):
        positive_comments = ' '.join(self.comments_df[self.comments_df['sentiment_category'] == 'Positive']['comment'])
        negative_comments = ' '.join(self.comments_df[self.comments_df['sentiment_category'] == 'Negative']['comment'])
        neutral_comments = ' '.join(self.comments_df[self.comments_df['sentiment_category'] == 'Neutral']['comment'])

        wordclouds = {
            'Positive': WordCloud(width=800, height=400, background_color='white').generate(positive_comments),
            'Negative': WordCloud(width=800, height=400, background_color='white').generate(negative_comments),
            'Neutral': WordCloud(width=800, height=400, background_color='white').generate(neutral_comments)
        }

        st.write("### Word Clouds by Sentiment Category")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(wordclouds['Positive'].to_array(), use_column_width=True, caption='Positive Comments')

        with col2:
            st.image(wordclouds['Neutral'].to_array(), use_column_width=True, caption='Neutral Comments')

        with col3:
            st.image(wordclouds['Negative'].to_array(), use_column_width=True, caption='Negative Comments')

def chart_sentiment_category_distribution_for_cache(comments_df):
    sentiment_counts = comments_df['sentiment_category'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    # Pie chart
    fig_pie = px.pie(sentiment_counts, values='Count', names='Sentiment', title='Sentiment Category Distribution on Video Comments')
    st.plotly_chart(fig_pie)

    # Bar chart
    fig_bar = px.bar(sentiment_counts, x='Sentiment', y='Count', title='Sentiment Category Distribution on Video Comments', 
                    color='Sentiment', text='Count')
    st.plotly_chart(fig_bar)

def display_word_cloud_by_sentiment_category_for_cache(comments_df):
    positive_comments = ' '.join(comments_df[comments_df['sentiment_category'] == 'Positive']['comment'])
    negative_comments = ' '.join(comments_df[comments_df['sentiment_category'] == 'Negative']['comment'])
    neutral_comments = ' '.join(comments_df[comments_df['sentiment_category'] == 'Neutral']['comment'])

    wordclouds = {
        'Positive': WordCloud(width=800, height=400, background_color='white').generate(positive_comments),
        'Negative': WordCloud(width=800, height=400, background_color='white').generate(negative_comments),
        'Neutral': WordCloud(width=800, height=400, background_color='white').generate(neutral_comments)
    }

    st.write("### Word Clouds by Sentiment Category")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(wordclouds['Positive'].to_array(), use_column_width=True, caption='Positive Comments')

    with col2:
        st.image(wordclouds['Neutral'].to_array(), use_column_width=True, caption='Neutral Comments')

    with col3:
        st.image(wordclouds['Negative'].to_array(), use_column_width=True, caption='Negative Comments')
# analysis = YouTubeSentimentAnalysis(youtube_service)
# top_videos_df = analysis.identify_top_5_videos_by_count(videos_df)
# comments_df, video_id = analysis.analyze_sentiment_video_with_maximum_comment_count(top_videos_df)
# analysis.chart_sentiment_category_distribution(comments_df)
