import pandas as pd
from textblob import TextBlob
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

class SentimentAnalysis:
    def __init__(self, df):
        self.df = df

    def get_sentiment(self, text):
        # Create a TextBlob object
        blob = TextBlob(text)
        # Get the sentiment polarity
        sentiment_polarity = blob.sentiment.polarity
        # Determine sentiment category based on polarity
        sentiment_category = 'Positive' if sentiment_polarity > 0 else 'Neutral' if sentiment_polarity == 0 else 'Negative'
        return sentiment_category, sentiment_polarity

    def data_aggregation_and_summary_statistics(self, df):
        # Apply the function to each video title or comment text
        df['sentiment_category'], df['sentiment_polarity'] = zip(*df['title'].apply(self.get_sentiment))

        # Calculate the proportion of positive, negative, and neutral titles
        sentiment_proportions = df['sentiment_category'].value_counts(normalize=True)

        # Display the sentiment proportions
        print("Sentiment Proportions:")
        print(sentiment_proportions)

        # Extract just the date part for easier correlation analysis
        df['publish_date'] = df['published'].dt.date

        # Calculate the mean view count for each sentiment category
        mean_view_counts = df.groupby('sentiment_category')['views_count'].mean()

        # Display the mean view counts by sentiment
        print("\nMean View Counts by Sentiment:")
        print(mean_view_counts)

        videos_df = df

        # Convert sentiment to a numerical scale for correlation analysis
        videos_df['sentiment_score'] = videos_df['sentiment_polarity'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

        # Calculate correlation of sentiment with view count and publish time
        sentiment_view_correlation = videos_df['sentiment_score'].corr(videos_df['views_count'])
        sentiment_time_correlation = videos_df['sentiment_score'].corr(videos_df['published'].astype('int64'))

        # Display the correlation results
        print("\nCorrelation between Sentiment and View Count:")
        print(sentiment_view_correlation)
        print("\nCorrelation between Sentiment and Publish Time:")
        print(sentiment_time_correlation)

        self.df = videos_df
        return videos_df

    def visualize_statistic_distribution_data(self):
        st.markdown("## Video Titles Sentiment Analysis")

        # Sentiment distribution pie chart
        sentiment_counts = self.df['sentiment_category'].value_counts()
        fig_pie = px.pie(values=sentiment_counts, names=sentiment_counts.index, title='Sentiment Distribution of Video Titles')
        st.plotly_chart(fig_pie)

        # Time series plot of sentiment over time
        self.df['publish_date'] = pd.to_datetime(self.df['publish_date'])
        self.df.sort_values('publish_date', inplace=True)
        self.df['numeric_sentiment'] = self.df['sentiment_polarity'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        sentiment_trend = self.df.groupby(['publish_year', 'publish_date'])['numeric_sentiment'].mean().reset_index()
        fig_time_series = px.line(sentiment_trend, x='publish_date', y='numeric_sentiment', title='Sentiment Trend Over Time')
        st.plotly_chart(fig_time_series)

        # Bar graph of sentiment categories
        fig_bar = px.bar(x=sentiment_counts.index, y=sentiment_counts, title='Number of Videos per Title Sentiment Category',
                         labels={'x': 'Sentiment Category', 'y': 'Number of Videos'})
        st.plotly_chart(fig_bar)

        # Word clouds for each sentiment category
        st.subheader("Word Clouds of Video Titles by Sentiment Category")
        categories = self.df['sentiment_category'].unique()
        categories_col1, categories_col2, categories_col3 = st.columns(3) # Neutral, Positive, Negative
        category_to_col = {
            'Neutral': categories_col1,
            'Positive': categories_col2,
            'Negative': categories_col3
        }

        # Generate and display word clouds in the appropriate columns
        for category in categories:
            titles = ' '.join(self.df[self.df['sentiment_category'] == category]['title'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(titles)
            fig, ax = plt.subplots(figsize=(15, 7.5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'Word Cloud of {category} Video Titles', fontsize=40)
            ax.axis('off')

            # Display in the appropriate column
            category_to_col[category].pyplot(fig)
        
        st.write("---")

def visualize_statistic_distribution_data_for_cache(df):
    st.markdown("## Video Titles Sentiment Analysis")

    # Sentiment distribution pie chart
    sentiment_counts = df['sentiment_category'].value_counts()
    fig_pie = px.pie(values=sentiment_counts, names=sentiment_counts.index, title='Sentiment Distribution of Video Titles')
    st.plotly_chart(fig_pie)

    # Time series plot of sentiment over time
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df.sort_values('publish_date', inplace=True)
    df['numeric_sentiment'] = df['sentiment_polarity'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    sentiment_trend = df.groupby(['publish_year', 'publish_date'])['numeric_sentiment'].mean().reset_index()
    fig_time_series = px.line(sentiment_trend, x='publish_date', y='numeric_sentiment', title='Sentiment Trend Over Time')
    st.plotly_chart(fig_time_series)

    # Bar graph of sentiment categories
    fig_bar = px.bar(x=sentiment_counts.index, y=sentiment_counts, title='Number of Videos per Title Sentiment Category',
                        labels={'x': 'Sentiment Category', 'y': 'Number of Videos'})
    st.plotly_chart(fig_bar)

    # Word clouds for each sentiment category
    st.subheader("Word Clouds of Video Titles by Sentiment Category")
    categories = df['sentiment_category'].unique()
    categories_col1, categories_col2, categories_col3 = st.columns(3) # Neutral, Positive, Negative
    category_to_col = {
        'Neutral': categories_col1,
        'Positive': categories_col2,
        'Negative': categories_col3
    }

    # Generate and display word clouds in the appropriate columns
    for category in categories:
        titles = ' '.join(df[df['sentiment_category'] == category]['title'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(titles)
        fig, ax = plt.subplots(figsize=(15, 7.5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f'Word Cloud of {category} Video Titles', fontsize=40)
        ax.axis('off')

        # Display in the appropriate column
        category_to_col[category].pyplot(fig)
    
    st.write("---")