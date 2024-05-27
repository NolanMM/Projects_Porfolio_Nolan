from datetime import datetime, timedelta
import re
import requests
import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv(override=True)

url_sample = os.getenv("URL_SAMPLE_YOUTUBE")
API_KEY___ = os.getenv("API_KEY___")

@st.cache_data
def data_extraction_enriching_process(channel_name, API_key = API_KEY___):
    """
    Performs the process of extracting data from a YouTube channel, creates a DataFrame, and processes the data.

    Parameters:
        youtube (googleapiclient.discovery.Resource, optional): An authorized resource object for interacting with the YouTube Data API.
            Defaults to None.
        channel (str, optional): The unique identifier of the YouTube channel.
            Defaults to None.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted video data and processed metrics.
    """
    # if youtube is None or channel is None:
    #     raise ValueError("'youtube' and 'channel' parameters are required.")

    youtube_process_module = Youtube_Process_Module(channel_name, API_key)
    channel_id = youtube_process_module.get_channel_id()
    channel_stats = youtube_process_module.get_channel_stats()
    playlist_ID_List = channel_stats[0]['contentDetails']['relatedPlaylists']['uploads']
    video_list = youtube_process_module.get_video_list(playlist_ID_List)
    video_data = youtube_process_module.get_video_details(video_list)
    
    # Convert to pandas dataframe and Data enrichment
    df = pd.DataFrame(video_data)
    df['title_length'] = df['title'].str.len()
    df['views_count'] = pd.to_numeric(df['views_count'])
    df['published'] = pd.to_datetime(df['published'])
    df['likes_count'] = pd.to_numeric(df['likes_count'])
    df['dislikes_count'] = pd.to_numeric(df['dislikes_count'])
    df['comments_count'] = pd.to_numeric(df['comments_count'])
    df['reactions'] = df['likes_count'] + df['dislikes_count'] + df['comments_count']
    df['publish_month'] = df['published'].dt.month
    df['publish_day'] = df['published'].dt.day
    df['publish_year'] = df['published'].dt.year
    df['publish_hour'] = df['published'].dt.hour
    df['publish_period'] = df['publish_hour'].apply(lambda x: 'AM' if x < 12 else 'PM')
    return df

class Youtube_Process_Module:
    chanel_id = ""
    channel_name = ""
    number_of_videos = 0

    def __init__(self, channel_name, _API_KEY):
        self.youtube = build('youtube', 'v3', developerKey=_API_KEY)
        self.channel_name = channel_name

    def get_channel_id(self):
        url_request = url_sample + self.channel_name
        response = requests.get(url_request)
        if response.status_code == 200:
            data = response.text
            match = re.search(r'"key":"browse_id","value":"([^"]+)"', data)
            if match:
                value = match.group(1)
                self.channel_id = value
            else:
                self.channel_id = None
        return self.channel_id

    def get_channel_stats(self):
        """
        Retrieves detailed statistics and metadata of a YouTube channel using the YouTube Data API.

        Parameters:
            youtube (googleapiclient.discovery.Resource): An authorized resource object for interacting with the YouTube Data API.
            channel_id (str): The unique identifier of the YouTube channel.

        Returns:
            dict: A dictionary containing comprehensive statistics and metadata of the specified YouTube channel.

            # Initialize the YouTube Data API client
            youtube = build('youtube', 'v3', developerKey='YOUR_API_KEY')

            # Retrieve statistics for a YouTube channel with the specified channel ID
            channel_statistics = get_channel_stats(youtube, 'UC_x5XG1OV2P6uZZ5FSM9Ttw')

            # Output the retrieved statistics
            print(channel_statistics)
        """
        request = self.youtube.channels().list(part='snippet,contentDetails,statistics', id=self.channel_id)
        response = request.execute()
        return response['items']

    def get_video_list(self,  playlist_ID):
        """
        Retrieves a list of video IDs from a YouTube playlist using the YouTube Data API.

        Parameters:
            youtube (googleapiclient.discovery.Resource): An authorized resource object for interacting with the YouTube Data API.
            playlist_ID (str): The unique identifier of the YouTube playlist.

        Returns:
            list: A list of video IDs contained within the specified playlist.

            # Initialize the YouTube Data API client
            youtube = build('youtube', 'v3', developerKey='YOUR_API_KEY')

            # Retrieve the list of video IDs from a YouTube playlist with the specified playlist ID
            video_list = get_video_list(youtube, 'PLwJS-GTzxiESBmJ4gJj-1wOuP5JoE41h4')

            # Output the retrieved list of video IDs
            print(video_list)
        """
        video_list = []
        request = self.youtube.playlistItems().list(
            part = "snippet,contentDetails",
            playlistId = playlist_ID,
            maxResults = 50
        )
        next_page = True
        while next_page:
            response = request.execute()
            data = response['items']

            for video in data:
                video_id = video['contentDetails']['videoId']
                if video_id not in video_list:
                    video_list.append(video_id)

            if 'nextPageToken' in response:
                next_page = True
                request = self.youtube.playlistItems().list(
                    part = "snippet,contentDetails",
                    playlistId = playlist_ID,
                    maxResults = 50,
                    pageToken = response['nextPageToken']
                )
            else:
                next_page = False
        
        return video_list

    def get_video_details(self, video_list):
        """
        Retrieves detailed statistics and metadata of YouTube videos using the YouTube Data API.

        Parameters:
            youtube (googleapiclient.discovery.Resource): An authorized resource object for interacting with the YouTube Data API.
            video_list (list): A list of video IDs whose details need to be fetched.

        Returns:
            list: A list of dictionaries, each containing comprehensive statistics and metadata of a YouTube video.

            # Initialize the YouTube Data API client
            youtube = build('youtube', 'v3', developerKey='YOUR_API_KEY')

            # Retrieve statistics for a list of YouTube videos with the specified video IDs
            video_data = get_video_details(youtube, ['VIDEO_ID_1', 'VIDEO_ID_2'])

            # Output the retrieved video statistics
            for video in video_data:
                print(video)
        """
        stats_list = []

        for i in range(0, len(video_list), 50):
            request = self.youtube.videos().list(
                part = "snippet,contentDetails,statistics",
                id = video_list[i:i+50]
            )
            response = request.execute()
            
            for video in response['items']:
                video_id = video['id']
                title = video['snippet']['title']
                published = video['snippet']['publishedAt']
                description = video['snippet']['description']
                tag_count = len(video['snippet'].get('tags',[]))
                views_count = video['statistics'].get('viewCount',0)
                dislikes_count = video['statistics'].get('dislikeCount',0)
                likes_count = video['statistics'].get('likeCount',0)
                comments_count = video['statistics'].get('commentCount',0)

                stats_dictionary = dict(
                    video_id= video_id,
                    title=title,
                    published=published,
                    description=description,
                    tag_count=tag_count,
                    views_count=views_count,
                    dislikes_count=dislikes_count,
                    likes_count=likes_count,
                    comments_count=comments_count
                )

                stats_list.append(stats_dictionary)

        return stats_list


# def main():
#     df = data_extraction_enriching_process("@Optimus96", API_KEY___)
#     print(df.head())
#     print(df.info())


# if __name__ == "__main__":
#     main()

