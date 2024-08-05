from io import StringIO
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Project.Project_13 import project_tests as t
import pickle
from pathlib import Path
import streamlit as st
import plotly.graph_objs as go

st.set_page_config(
    page_title="Design a Recommendation Engine with IBM Watson", page_icon=":bar_chart:")
st.sidebar.header("Project NolanM")
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = "./styles/main.css"

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

# Title and Project Information
st.title("Design a Recommendation Engine with IBM Watson")
st.subheader(
    "Project 3 (Experimental Design & Recommendations) in the Udacity Data Scientist Nanodegree (Summer 2023)")
st.markdown(
    "[Data Scientist Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025)")

st.markdown("""
## Project Description:
IBM has an online data science community where members can post tutorials, notebooks, articles, and datasets. In this project, you will build a recommendation engine, based on user behavior and social network in IBM Watson Studio’s data platform, to surface content most likely to be relevant to a user.

For this project, we will analyze the interactions that users have with articles on the IBM Watson Studio platform, and make recommendations to them about new articles we think they will like. Below is an example of what the dashboard could look like displaying articles on the IBM Watson Platform.
""")
st.image("https://user-images.githubusercontent.com/59873708/122260801-35b60d00-cec3-11eb-95b8-847480bf879d.png",
         caption="IBM Watson Studio Dashboard")

# Read data
user_df = pd.read_csv("./data/Project_13/user-item-interactions.csv")
articles_df = pd.read_csv("./data/Project_13/articles_community.csv")
del user_df['Unnamed: 0']
del articles_df['Unnamed: 0']

# Content of the Python Notebook Section
st.markdown("""
## Content of the Python Notebook
The project is organized into the following tasks, each addressing specific aspects of building a recommendation system:

### I. Exploratory Data Analysis
In this initial phase, we perform a comprehensive analysis of the dataset to uncover patterns and insights. This involves:
- Loading the dataset and inspecting its structure, including the shape, data types, and missing values.
- Visualizing distributions of key features such as user interactions, article views, and article metadata.
- Conducting summary statistics to understand central tendencies and variances.
- Identifying trends and correlations that might influence the recommendation engine.""", unsafe_allow_html=True)

st.markdown("""#### Review Input Data""", unsafe_allow_html=True)
st.markdown("""##### 1. User Item Interactions Dataset""",
            unsafe_allow_html=True)
st.success("User Item Interactions Dataset Shape: {}".format(user_df.shape))
st.write(user_df.head(5))
st.markdown("""##### 2. Articles Community Dataset""",
            unsafe_allow_html=True)
st.success("Articles Community Dataset Shape: {}".format(articles_df.shape))
st.write(articles_df.head(5))

st.markdown("""#### Implemented Exploratory Data Analysis
**1. Examined the distribution of the number of articles each user interacts with in the dataset**""", unsafe_allow_html=True)

# Calculate the number of interactions per user:
interactions_per_user = user_df.groupby('email')['article_id'].count()
# Calculate the median and maximum number of user_article interactios
median_val = interactions_per_user.median()
max_views_by_user = interactions_per_user.max()


def plot_interactions(articles_by_user):
    trace0 = go.Histogram(
        x=articles_by_user,
        xbins={
            "start": np.min(articles_by_user),
            "end": np.max(articles_by_user),
            "size": 1
        }
    )
    data = [trace0]
    layout = go.Layout(
        title="DISTRIBUTION OF HOW MANY ARTICLES <br> A USER INTERACTS WITH ",
        xaxis={
            "title": "Number of articles",
            "automargin": True,
            "showgrid": True,
            "tick0": 0,
            "dtick": 25,
            "zeroline": False
        },

        yaxis={
            "title": "Number of users",
            "automargin": True,
            "showgrid": True,
        }

    )
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)


plot_interactions(interactions_per_user.values)

st.write(f"Median: {median_val}")
st.write(f"Maximum views by a user: {max_views_by_user}")
st.success(
    f"50% of individuals interact with {median_val} number of articles or fewer.")
st.success(
    f"The maximum number of user-article interactions by any single user is {max_views_by_user}.")

st.markdown("""**2. Explore and remove duplicate articles from the df_content dataframe.**""",
            unsafe_allow_html=True)
duplicate_articles = user_df['article_id'].duplicated()
# Remove any rows that have the same article_id - only keep the first
df_unique = user_df.drop_duplicates(
    subset='article_id', keep='first')
df_unique = df_unique.reset_index(drop=True)

# convert article_id type from int/float to string
articles_df.article_id = articles_df.article_id.astype(float).astype(str)
user_df.article_id = user_df.article_id.astype(str)
df_article_ids = set(user_df.article_id.tolist())
df_content_article_ids = set(articles_df.article_id.tolist())
st.write(df_unique)
st.warning(f"Number of duplicate articles: {duplicate_articles.sum()}")
st.success(f"Number of unique articles: {len(df_unique)}")

# The number of unique articles that have at least one interaction
unique_articles = (user_df.groupby('article_id').count()['email'] > 0).count()
# The number of unique articles on the IBM platform
total_articles = articles_df['article_id'].nunique()
unique_users = user_df.email.nunique(dropna=True)  # The number of unique users
# The number of user-article interactions
user_article_interactions = user_df.count()['article_id']
# The number of unique articles that have at least one interaction
number_of_unique_articles = len(user_df["article_id"].unique())
# The number of unique articles on the IBM platform
number_of_total_articles = articles_df["article_id"].nunique()
# The number of unique users
number_of_unique_users_excluding_null = len(user_df["email"].dropna().unique())
# The number of user-article interactions
num_interactions = user_df.shape[0]
###############################################################

st.markdown("""**3. Find Statistics Data for the list:**""",
            unsafe_allow_html=True)
st.markdown("""       
- a. Total the number of unique articles that have an interaction with a user.
```python
    number_of_unique_articles = len(user_df["article_id"].unique())
```
""", unsafe_allow_html=True)
st.success(f"Number of unique articles: {number_of_unique_articles}")

st.markdown("""
- b. Total the number of unique articles in the dataset (whether they have any interactions or not).
```python
    number_of_total_articles = articles_df["article_id"].nunique()
```
""", unsafe_allow_html=True)
st.success(f"Number of total articles: {number_of_total_articles}")

st.markdown("""
- c. Total the number of unique users in the dataset. (excluding null values)
```python
    number_of_unique_users_excluding_null = len(user_df["email"].dropna().unique())
```
""", unsafe_allow_html=True)
st.success(
    f"Number of unique users excluding null: {number_of_unique_users_excluding_null}")

st.markdown("""
- d. Total the number of user-article interactions in the dataset.
```python
    num_interactions = user_df.shape[0]
```
""", unsafe_allow_html=True)
st.success(f"Number of User-Article Interactions:{num_interactions}")

########################################################################
st.markdown("""**4. Analysis of Most Viewed Article and User Mapping Using Email Mapper Function**
            
Identify the most viewed article_id and the frequency of its views. Due to the discussions with company leaders, the email_mapper function was confirmed as a reasonable method for mapping users to ids. Although there were a small number of null values, it was determined that these null values likely belonged to a single user        
""", unsafe_allow_html=True)

most_viewed_article_id = user_df['article_id'].value_counts().idxmax()
max_view = user_df['article_id'].value_counts().max()

st.markdown("""
- 4.1. Identify the most viewed article_id and the frequency of its views.
```python
most_viewed_article_id = user_df['article_id'].value_counts().idxmax()
max_view = user_df['article_id'].value_counts().max()
```
""", unsafe_allow_html=True)
st.success(f"Most Viewed Article ID: {most_viewed_article_id}")
st.success(f"Most Viewed Article Frequency: {max_view}")

st.markdown("""
- 4.2. Email Mapper Function
```python
def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded
```
""", unsafe_allow_html=True)


def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []

    for val in user_df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter += 1

        email_encoded.append(coded_dict[val])
    return email_encoded


email_encoded = email_mapper()
del user_df['email']
user_df['user_id'] = email_encoded
st.write("Dataframe after mapping email to user_id:")
st.write(user_df.head(5))

st.markdown("""**5. Check Result of the Exploratory Data Analysis**""",
            unsafe_allow_html=True)
st.markdown("""
```python
sol_1_dict = {
    '`50% of individuals have _____ or fewer interactions.`': median_val,
    '`The total number of user-article interactions in the dataset is ______.`': user_article_interactions,
    '`The maximum number of user-article interactions by any 1 user is ______.`': max_views_by_user,
    '`The most viewed article in the dataset was viewed _____ times.`': max_views,
    '`The article_id of the most viewed article is ______.`': most_viewed_article_id,
    '`The number of unique articles that have at least 1 rating ______.`': unique_articles,
    '`The number of unique users in the dataset is ______`': unique_users,
    '`The number of unique articles on the IBM platform`': total_articles
}

# Test your dictionary against the solution
t.sol_1_test(sol_1_dict)
```
""", unsafe_allow_html=True)

sol_1_dict = {
    '`50% of individuals have _____ or fewer interactions.`': median_val,
    '`The total number of user-article interactions in the dataset is ______.`': user_article_interactions,
    '`The maximum number of user-article interactions by any 1 user is ______.`': max_views_by_user,
    '`The most viewed article in the dataset was viewed _____ times.`': max_view,
    '`The article_id of the most viewed article is ______.`': most_viewed_article_id,
    '`The number of unique articles that have at least 1 rating ______.`': unique_articles,
    '`The number of unique users in the dataset is ______`': unique_users,
    '`The number of unique articles on the IBM platform`': total_articles
}
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
# Test your dictionary against the solution
t.sol_1_test(sol_1_dict)
sys.stdout = old_stdout
st.text(mystdout.getvalue())


st.markdown("""
### II. Rank-Based Recommendations
This task focuses on identifying the most popular articles based on the total number of user interactions. Steps include:
- Aggregating interaction counts for each article to determine popularity.
- Sorting articles by interaction count to generate a ranked list.
- Creating a function to recommend the top-5, top-10, top-20 popular articles to new users or users with no prior interactions.
- Implementing this ranking logic in Python and testing its output with sample data.""", unsafe_allow_html=True)

st.markdown("""
#### II.1. Get Title Top Article Funtions
```python
def get_top_articles(n, df=df):
    top_articles = df['title'].value_counts().head(n).index.tolist()
    return top_articles
```
""", unsafe_allow_html=True)
st.markdown("""
#### II.2. Get Idx of Top Article Funtions
```python
def get_top_article_ids(n, df=df):
    top_articles_idx = df['article_id'].value_counts().head(n).index.tolist()
    return top_articles_idx
```
""", unsafe_allow_html=True)


def get_top_articles(n, df=user_df):
    top_articles = df['title'].value_counts().head(n).index.tolist()
    return top_articles


def get_top_article_ids(n, df=user_df):
    top_articles_idx = df['article_id'].value_counts().head(n).index.tolist()
    return top_articles_idx


top_5 = get_top_articles(5)
top_10 = get_top_articles(10)
top_20 = get_top_articles(20)

st.markdown("""**Top 5 Articles:**""")
st.write(top_5)
st.markdown("""**Top 10 Articles:**""")
st.write(top_10)
st.markdown("""**Top 20 Articles:**""")
st.write(top_20)

st.markdown("""**II.3 Check Result of the Rank-Based Recommendations**""",
            unsafe_allow_html=True)
st.markdown("""
```python
t.sol_2_test(get_top_articles)
```
""", unsafe_allow_html=True)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
t.sol_2_test(get_top_articles)
sys.stdout = old_stdout
st.text(mystdout.getvalue())
########################################################################
st.markdown("""
### III. User-User Based Collaborative Filtering
For personalized recommendations, we use collaborative filtering techniques, specifically:
- Calculating similarity scores between users based on their interaction histories using metrics such as cosine similarity or Pearson correlation.
- Using the similarity matrix to predict user preferences for unseen articles by averaging the ratings of similar users.""", unsafe_allow_html=True)

st.markdown("""**1. Pivot user dataframe and article dataframe**""",
            unsafe_allow_html=True)
st.markdown("""
```python
def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    user_item = df.groupby(['user_id', 'article_id']).size().unstack().fillna(0)
    user_item[user_item > 0] = 1
    return user_item

user_item = create_user_item_matrix(df)
```
""", unsafe_allow_html=True)


def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns

    OUTPUT:
    user_item - user item matrix 

    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    user_item = df.groupby(['user_id', 'article_id']
                           ).size().unstack().fillna(0)
    user_item[user_item > 0] = 1
    return user_item


user_item = create_user_item_matrix(user_df)
st.write(user_item.head(20))

st.markdown("""**2.QuickCheck Pivot Result**""",
            unsafe_allow_html=True)
st.markdown("""
```python
assert user_item.shape[0] == 5149, "Oops!  The number of users in the user-article matrix doesn't look right."
assert user_item.shape[1] == 714, "Oops!  The number of articles in the user-article matrix doesn't look right."
assert user_item.sum(axis=1)[1] == 36, "Oops!  The number of articles seen by user 1 doesn't look right."
print("Passed quick tests! Thank you!")
```
""", unsafe_allow_html=True)
assert user_item.shape[0] == 5149, "Oops!  The number of users in the user-article matrix doesn't look right."
assert user_item.shape[1] == 714, "Oops!  The number of articles in the user-article matrix doesn't look right."
assert user_item.sum(axis=1)[
    1] == 36, "Oops!  The number of articles seen by user 1 doesn't look right."
st.success("Passed quick tests! Thank you!")

st.markdown("""**3.Calculate Similarity Scores**""",
            unsafe_allow_html=True)
st.markdown("""
```python
def find_similar_users(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest Kendall's Tau coefficient users)
                    are listed first
    
    Description:
    Computes the similarity of every pair of users based on Kendall's Tau coefficient
    Returns an ordered list of similar users
    '''
    # Compute the Kendall's Tau coefficient between the given user and all other users
    similarity_scores = (user_item.loc[user_id] * user_item).sum(axis=1)
    
    # Sort the similarity scores in descending order
    similarity_scores = similarity_scores.sort_values(ascending=False)
    
    # Remove the given user_id from the list
    similarity_scores = similarity_scores.drop(user_id)
    
    # Get the list of similar users in order from most to least similar
    similar_users = similarity_scores.index.tolist()
    
    return similar_users
```
""", unsafe_allow_html=True)


def find_similar_users(user_id, user_item_=user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise

    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest Kendall's Tau coefficient users)
                    are listed first

    Description:
    Computes the similarity of every pair of users based on Kendall's Tau coefficient
    Returns an ordered list of similar users
    '''
    # Compute the Kendall's Tau coefficient between the given user and all other users
    similarity_scores = (user_item_.loc[user_id] * user_item_).sum(axis=1)

    # Sort the similarity scores in descending order
    similarity_scores = similarity_scores.sort_values(ascending=False)

    # Remove the given user_id from the list
    similarity_scores = similarity_scores.drop(user_id)

    # Get the list of similar users in order from most to least similar
    similar_users = similarity_scores.index.tolist()

    return similar_users


st.write("Test the function with user 1, 3933, and 46:")
st.success("The 10 most similar users to user 1 are: {}".format(
    find_similar_users(1)[:10]))
st.success("The 5 most similar users to user 3933 are: {}".format(
    find_similar_users(3933)[:5]))
st.success("The 3 most similar users to user 46 are: {}".format(
    find_similar_users(46)[:3]))

st.markdown("""**4. Building Recommendation For Each User Function**""",
            unsafe_allow_html=True)
st.markdown("""
```python
def get_article_names(article_ids, df=df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    article_names = []
    for i in article_ids:
        try:
            name = df.groupby(['article_id', 'title']).count().reset_index('title').loc[i, 'title']
            article_names.append(name)
        except:
            continue
    return article_names


def get_user_articles(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the doc_full_name column in df_content)
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    article_ids = user_item.loc[user_id].index[user_item.loc[user_id] == 1].tolist()
    article_names = get_article_names(article_ids)
    return article_ids, article_names


def user_user_recs(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    Users who are the same closeness are chosen arbitrarily as the 'next' user
    
    For the user where the number of recommended articles starts below m 
    and ends exceeding m, the last items are chosen arbitrarily
    
    '''
    # Get the list of similar users ordered by Kendall's Tau coefficient
    similar_users = find_similar_users(user_id)[:m]
    
    recs = []
    # Set up flag technique
    flag = get_user_articles(user_id)
    
    for user in similar_users:
        # Get the articles seen by the similar user
        article_ids, _ = get_user_articles(user)
        
        # Find articles that the input user hasn't seen yet
        for i in article_ids:
            if i not in flag and i not in recs and len(recs) < m:
                recs.append(i)
            else:
                break
        if len(recs) >= m:
            break
    return recs
```
""", unsafe_allow_html=True)


def get_article_names(article_ids, df=user_df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    article_names = []
    for i in article_ids:
        try:
            name = df.groupby(['article_id', 'title']).count(
            ).reset_index('title').loc[i, 'title']
            article_names.append(name)
        except Exception as e:
            st.error(e)
    return article_names


def get_user_articles(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise

    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the doc_full_name column in df_content)

    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    article_ids = user_item.loc[user_id].index[user_item.loc[user_id] == 1].tolist(
    )
    article_names = get_article_names(article_ids)
    return article_ids, article_names


def user_user_recs(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user

    OUTPUT:
    recs - (list) a list of recommendations for the user

    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found

    Notes:
    Users who are the same closeness are chosen arbitrarily as the 'next' user

    For the user where the number of recommended articles starts below m 
    and ends exceeding m, the last items are chosen arbitrarily

    '''
    # Get the list of similar users ordered by Kendall's Tau coefficient
    similar_users = find_similar_users(user_id)[:m]

    recs = []
    # Set up flag technique
    flag = get_user_articles(user_id)

    for user in similar_users:
        # Get the articles seen by the similar user
        article_ids, _ = get_user_articles(user)

        # Find articles that the input user hasn't seen yet
        for i in article_ids:
            if i not in flag and i not in recs and len(recs) < m:
                recs.append(i)
            else:
                break
        if len(recs) >= m:
            break
    return recs


st.write("Test the function with user 1 with 10 recommendations")
recommend_test_section_1, recommend_test_section_2 = st.columns(2)
with recommend_test_section_1:
    st.write("Recommendations for User 1 Articles ID:")
    st.write(user_user_recs(1, 10))
with recommend_test_section_2:
    st.write("Recommendations for User 1 Articles:")
    st.write(get_article_names(user_user_recs(1, 10)))

st.markdown("""**5. Check Result of the User-Articles Recommendations**""",
            unsafe_allow_html=True)

st.markdown("""
```python
assert set(get_article_names(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model', 'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis']), "Oops! Your the get_article_names function doesn't work quite how we expect."
assert set(get_article_names(['1320.0', '232.0', '844.0'])) == set(['housing (2015): united states demographic measures','self-service data preparation with ibm data refinery','use the cloudant-spark connector in python notebook']), "Oops! Your the get_article_names function doesn't work quite how we expect."
assert set(get_user_articles(20)[0]) == set(['1320.0', '232.0', '844.0'])
assert set(get_user_articles(20)[1]) == set(['housing (2015): united states demographic measures', 'self-service data preparation with ibm data refinery','use the cloudant-spark connector in python notebook'])
assert set(get_user_articles(2)[0]) == set(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])
assert set(get_user_articles(2)[1]) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model', 'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis'])
print("Passed all of the tests! Thank you!")
```
""", unsafe_allow_html=True)

assert set(get_article_names(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model',
                                                                                                    'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis']), "Oops! Your the get_article_names function doesn't work quite how we expect."
assert set(get_article_names(['1320.0', '232.0', '844.0'])) == set(['housing (2015): united states demographic measures', 'self-service data preparation with ibm data refinery',
                                                                    'use the cloudant-spark connector in python notebook']), "Oops! Your the get_article_names function doesn't work quite how we expect."
assert set(get_user_articles(20)[0]) == set(['1320.0', '232.0', '844.0'])
assert set(get_user_articles(20)[1]) == set(['housing (2015): united states demographic measures',
                                             'self-service data preparation with ibm data refinery', 'use the cloudant-spark connector in python notebook'])
assert set(get_user_articles(2)[0]) == set(
    ['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])
assert set(get_user_articles(2)[1]) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model',
                                            'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis'])
st.success("Passed all of the tests! Thank you!")

st.markdown("""**6. Improve the User-Articles Recommendations**
- Instead of arbitrarily choosing when we obtain users who are all the same closeness to a given user - choose the users that have the most total article interactions before choosing those with fewer article interactions.
- Instead of arbitrarily choosing articles from the user where the number of recommended articles starts below m and ends exceeding m, choose articles with the articles with the most total interactions before choosing those with fewer total interactions. This ranking should be what would be obtained from the top_articles function you wrote earlier.""", unsafe_allow_html=True)

st.markdown("""
```python
def get_top_sorted_users(user_id, df=df, user_item=user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise
    
            
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''
    user_similarities = (user_item.loc[user_id] * user_item).sum(axis=1).sort_values(ascending=False).drop(user_id).to_frame("similarity")
    user_interactions = df.groupby('user_id').count()['article_id'].sort_values(ascending=False).drop(user_id).to_frame("num_interactions")
    neighbors_df = user_similarities.join(user_interactions, on='user_id', how='left').sort_values(by=['similarity', 'num_interactions'], ascending=False)
    neighbors_df.rename_axis("neighbor_id", inplace=True)
    
    return neighbors_df # Return the dataframe specified in the doc_string


def user_user_recs_part2(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    * Choose the users that have the most total article interactions 
    before choosing those with fewer article interactions.

    * Choose articles with the articles with the most total interactions 
    before choosing those with fewer total interactions. 
   
    '''
    closest_neighbors = get_top_sorted_users(user_id)[:m]
    
    # seen items of user_id
    seen_items = get_user_articles(user_id)
    
    # get top interacted article ids
    top_articles = get_top_article_ids(len(df), df=df)
    
    recs = []
    for u in closest_neighbors.iterrows():
        article_ids, _ = get_user_articles(u[0])
        
        for i in top_articles:
            if i in article_ids and i not in seen_items and i not in recs and len(recs) < m:
                recs.append(i)
            elif len(recs) >= m:
                break
        if len(recs) >= m:
            break
    rec_names = get_article_names(recs)
    return recs, rec_names
```
""", unsafe_allow_html=True)


def get_top_sorted_users(user_id, df=user_df, user_item=user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise


    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u

    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe

    '''
    user_similarities = (user_item.loc[user_id] * user_item).sum(
        axis=1).sort_values(ascending=False).drop(user_id).to_frame("similarity")
    user_interactions = df.groupby('user_id').count()['article_id'].sort_values(
        ascending=False).drop(user_id).to_frame("num_interactions")
    neighbors_df = user_similarities.join(user_interactions, on='user_id', how='left').sort_values(
        by=['similarity', 'num_interactions'], ascending=False)
    neighbors_df.rename_axis("neighbor_id", inplace=True)

    return neighbors_df


def user_user_recs_part2(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user

    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title

    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found

    Notes:
    * Choose the users that have the most total article interactions 
    before choosing those with fewer article interactions.

    * Choose articles with the articles with the most total interactions 
    before choosing those with fewer total interactions. 

    '''
    closest_neighbors = get_top_sorted_users(user_id)[:m]

    # seen items of user_id
    seen_items = get_user_articles(user_id)

    # get top interacted article ids
    top_articles = get_top_article_ids(len(user_df), df=user_df)

    recs = []
    for u in closest_neighbors.iterrows():
        article_ids, _ = get_user_articles(u[0])

        for i in top_articles:
            if i in article_ids and i not in seen_items and i not in recs and len(recs) < m:
                recs.append(i)
            elif len(recs) >= m:
                break
        if len(recs) >= m:
            break
    rec_names = get_article_names(recs)
    return recs, rec_names


st.markdown("""**7. Testing Improvement of the User-Articles Recommendations**""",
            unsafe_allow_html=True)
st.markdown("""
```python
# Testing with the top 10 recommendations for user 20
rec_ids, rec_names = user_user_recs_part2(20, 10)
print("The top 10 recommendations for user 20 are the following article ids:")
print(rec_ids)
print("The top 10 recommendations for user 20 are the following article names:")
print(rec_names)

# Tests with a dictionary of results
# Find the user that is most similar to user 1         
user1_most_sim = get_top_sorted_users(1)[:1].index[0]
# Find the 10th most similar user to user 131
user131_10th_sim = get_top_sorted_users(131)[9:10].index[0]

sol_5_dict = {
    'The user that is most similar to user 1.': user1_most_sim, 
    'The user that is the 10th most similar to user 131': user131_10th_sim,
}

t.sol_5_test(sol_5_dict)
```""", unsafe_allow_html=True)

rec_ids, rec_names = user_user_recs_part2(20, 10)
testing_section_1_improve, testing_section_2_improve = st.columns(2)
with testing_section_1_improve:
    st.write("The top 10 recommendations for user 20 are the following article ids:")
    st.write(rec_ids)
with testing_section_2_improve:
    st.write("The top 10 recommendations for user 20 are the following article names:")
    st.write(rec_names)

user1_most_sim = get_top_sorted_users(1)[:1].index[0]
user131_10th_sim = get_top_sorted_users(131)[9:10].index[0]

sol_5_dict = {
    'The user that is most similar to user 1.': user1_most_sim,
    'The user that is the 10th most similar to user 131': user131_10th_sim,
}

old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
t.sol_5_test(sol_5_dict)
sys.stdout = old_stdout
st.text(mystdout.getvalue())

st.markdown("""**8. Recommendation for New User (0.0)**
- We using the top 10 recommendations for a new user (0.0) because we don't have any interaction data for this user yet.            
""",
            unsafe_allow_html=True)
st.markdown("""
```python
new_user = '0.0'
new_user_recs = get_top_article_ids(10)

# Test the recommendations for user 0.0
assert set(new_user_recs) == set(['1314.0','1429.0','1293.0','1427.0','1162.0','1364.0','1304.0','1170.0','1431.0','1330.0']), "Oops!  It makes sense that in this case we would want to recommend the most popular articles, because we don't know anything about these users."
print("Passed!  Nice job!")
```""", unsafe_allow_html=True)

new_user = '0.0'
new_user_recs = get_top_article_ids(10)

assert set(new_user_recs) == set(['1314.0', '1429.0', '1293.0', '1427.0', '1162.0', '1364.0', '1304.0', '1170.0', '1431.0', '1330.0']
                                 ), "Oops!  It makes sense that in this case we would want to recommend the most popular articles, because we don't know anything about these users."

st.success("Passed! Nice job!")

st.markdown("""
### IV. Matrix Factorization
- Build use matrix factorization to make article recommendations to the users on the IBM Watson Studio platform.
- Prepare Dataset for Part V
- Calculate how the accuracy improves as we increase the number of latent features.
""", unsafe_allow_html=True)

st.markdown("""**1. Load the Data**""", unsafe_allow_html=True)
st.markdown("""
```python
def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns

    OUTPUT:
    user_item - user item matrix 

    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    user_item = df.groupby(['user_id', 'article_id']).size().unstack().fillna(0)
    user_item[user_item > 0] = 1
    return user_item

user_item = create_user_item_matrix(df)
```""", unsafe_allow_html=True)

user_item_matrix = pd.read_pickle('./Project/Project_13/user_item_matrix.p')
st.write("User-Item Matrix:")
st.write(user_item_matrix.head(5))

st.markdown("""**2. Using Singular Value Decomposition (SVD) and Find Number of Latent Features Use**
- More information:<a>https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.svd.html</a>

```python
u, s, vt = np.linalg.svd(user_item_matrix)

num_latent_feats = np.arange(10,700+10,20)
sum_errs = []

for k in num_latent_feats:
    # restructure with k latent features
    s_new, u_new, vt_new = np.diag(s[:k]), u[:, :k], vt[:k, :]
    
    # take dot product
    user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))
    
    # compute error for each prediction to actual value
    diffs = np.subtract(user_item_matrix, user_item_est)
    
    # total errors and keep track of them
    err = np.sum(np.sum(np.abs(diffs)))
    sum_errs.append(err)

# Calculate accuracy            
accuracy = 1 - np.array(sum_errs)/df.shape[0]

plt.figure(figsize=(10, 6))       
plt.plot(num_latent_feats, accuracy, marker='o')
plt.xlabel('Number of Latent Features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Latent Features')
```
""", unsafe_allow_html=True)
u, s, vt = np.linalg.svd(user_item_matrix)
num_latent_feats = np.arange(10, 700+10, 20)
sum_errs = []

for k in num_latent_feats:
    # restructure with k latent features
    s_new, u_new, vt_new = np.diag(s[:k]), u[:, :k], vt[:k, :]

    # take dot product
    user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))

    # compute error for each prediction to actual value
    diffs = np.subtract(user_item_matrix, user_item_est)

    # total errors and keep track of them
    err = np.sum(np.sum(np.abs(diffs)))
    sum_errs.append(err)

accuracy = 1 - np.array(sum_errs)/user_df.shape[0]

plt.figure(figsize=(10, 6))
plt.plot(num_latent_feats, accuracy, marker='o')
plt.xlabel('Number of Latent Features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Latent Features')
st.pyplot(plt)

st.warning("From the above, we can't really be sure how many features to use, because simply having a better way to predict the 1's and 0's of the matrix doesn't exactly give us an indication of if we are able to make good recommendations. Instead, we might split our dataset into a training and test set of data and use the list of question to figure out the impact on accuracy of the training and test sets of data with different numbers of latent features")

st.markdown("""**List of statements***:
1. How many users can we make predictions for in the test set?
2. How many users are we not able to make predictions for because of the cold start problem?
3. How many articles can we make predictions for in the test set?
4. How many articles are we not able to make predictions for because of the cold start problem?""", unsafe_allow_html=True)

st.markdown("""**3. Prepare Dataset**
```python
df_train = df.head(40000)
df_test = df.tail(5993)

def create_test_and_train_user_item(df_train, df_test):
    '''
    INPUT:
    df_train - training dataframe
    df_test - test dataframe
    
    OUTPUT:
    user_item_train - a user-item matrix of the training dataframe 
                      (unique users for each row and unique articles for each column)
    user_item_test - a user-item matrix of the testing dataframe 
                    (unique users for each row and unique articles for each column)
    test_idx - all of the test user ids
    test_arts - all of the test article ids
    
    '''
    user_item_train = create_user_item_matrix(df_train)
    user_item_test = create_user_item_matrix(df_test)
    
    test_idx = user_item_test.index.tolist()
    test_arts = user_item_test.columns.tolist()
    
    return user_item_train, user_item_test, test_idx, test_arts

user_item_train, user_item_test, test_idx, test_arts = create_test_and_train_user_item(df_train, df_test)

# The number of users can we make predictions for in the test set
len(test_idx) - user_item_test.shape[0]
# The number of articles can we make predictions for in the test set
len(test_arts)
```""", unsafe_allow_html=True)
df_train = user_df.head(40000)
df_test = user_df.tail(5993)


def create_test_and_train_user_item(df_train, df_test):
    '''
    INPUT:
    df_train - training dataframe
    df_test - test dataframe

    OUTPUT:
    user_item_train - a user-item matrix of the training dataframe 
                      (unique users for each row and unique articles for each column)
    user_item_test - a user-item matrix of the testing dataframe 
                    (unique users for each row and unique articles for each column)
    test_idx - all of the test user ids
    test_arts - all of the test article ids

    '''
    user_item_train = create_user_item_matrix(df_train)
    user_item_test = create_user_item_matrix(df_test)

    test_idx = user_item_test.index.tolist()
    test_arts = user_item_test.columns.tolist()

    return user_item_train, user_item_test, test_idx, test_arts


user_item_train, user_item_test, test_idx, test_arts = create_test_and_train_user_item(
    df_train, df_test)

create_data_result_section_1, create_data_result_section_2 = st.columns(2)

st.success("The number of users can we make predictions for in the test set: {}".format(
    len(test_idx) - user_item_test.shape[0]))
st.success("The number of articles can we make predictions for in the test set: {}".format(
    len(test_arts)))

with create_data_result_section_1:
    st.write("User-Item Matrix Train:")
    st.write(user_item_train.head(15))

with create_data_result_section_2:
    st.write("User-Item Matrix Test:")
    st.write(user_item_test.head(15))

st.markdown("""**4. Validated the Dataset**
```python
a = 662 # len(test_idx) - user_item_test.shape[0]
b = 574 # user_test_shape[1] or len(test_arts) because we can make predictions for all articles
c = 20 # user_item_test.shape[0]
d = 0 # len(test_arts) - user_item_test.shape[1]

sol_4_dict_1 = {
    'How many users can we make predictions for in the test set?': c, 
    'How many users in the test set are we not able to make predictions for because of the cold start problem?': a, 
    'How many movies can we make predictions for in the test set?': b,
    'How many movies in the test set are we not able to make predictions for because of the cold start problem?': d
}

t.sol_4_test(sol_4_dict)
```""", unsafe_allow_html=True)

a = 662  # len(test_idx) - user_item_test.shape[0]
# user_test_shape[1] or len(test_arts) because we can make predictions for all articles
b = 574
c = 20  # user_item_test.shape[0]
d = 0  # len(test_arts) - user_item_test.shape[1]

sol_4_dict = {
    'How many users can we make predictions for in the test set?': c,
    'How many users in the test set are we not able to make predictions for because of the cold start problem?': a,
    'How many movies can we make predictions for in the test set?': b,
    'How many movies in the test set are we not able to make predictions for because of the cold start problem?': d
}
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
t.sol_4_test(sol_4_dict)
sys.stdout = old_stdout
st.text(mystdout.getvalue())

st.markdown("""
### V. Improve the Recommendations Model
- Used the user_item_train dataset to find U, S, and V transpose using SVD.
- Find the subset of rows in the user_item_test dataset that can predict using this matrix decomposition with different numbers of latent features to see how many features makes sense to keep based on the accuracy on the test data
<br>
```python
# fit SVD on the user_item_train matrix
u_train, s_train, vt_train = np.linalg.svd(user_item_train)

# Change the dimensions of u, s, and vt as necessary to use for the 20 users in the test set that SVD can make predictions on.
# Update the shape of u_train and store in u_new

# users in test set that can be predicted (because also in training set)
# We filter for test_idx - all of the test user ids
u_new = u_train[user_item_train.index.isin(test_idx), :]

# update the shape of s and store in s_new
s_new = np.zeros((len(s), len(s)))
s_new[:len(s), :len(s)] = np.diag(s) 

# user_ids in test set that can be predicted because also in the training set
vt_new = vt_train[:, user_item_train.columns.isin(test_arts)]

# Use the training decomposition to predict on test data
num_latent_feat=np.arange(10,714+10,20)

sum_errs_train = []
sum_errs_test  = []
intersec_ids   = list(set(user_item_train.index).intersection(set(test_idx))) # list of users in both train and test sets

for k in num_latent_feat: 
    # restructure with k latent features
    u_train_lat, s_train_lat, vt_train_lat = u_train[:, :k], np.diag(s_train[:k]), vt_train[:k, :]
    u_test_lat, vt_test_lat = u_new[:, :k], vt_new[:k,:]
    
    # take dot product
    user_item_train_est = np.around(np.dot(np.dot(u_train_lat, s_train_lat), vt_train_lat))
    user_item_test_est  = np.around(np.dot(np.dot(u_test_lat, s_train_lat), vt_test_lat))
    
    # compute error for each prediction to actual value
    diffs_train = np.subtract(user_item_train, user_item_train_est)
    diffs_test = np.subtract(user_item_test.loc[intersec_ids, :], user_item_test_est)
    
    # total errors and keep track of them 
    train_err = np.sum(np.sum(np.abs(diffs_train))) 
    sum_errs_train.append(train_err)
    
    test_err = np.sum(np.sum(np.abs(diffs_test))) 
    sum_errs_test.append(test_err)
    
    all_classifications_train = user_item_train_est.shape[0]* user_item_train_est.shape[1]
    all_classifications_test  = user_item_test_est.shape[0] * user_item_test_est.shape[1]
    
plt.plot(num_latent_feat, 1 - (np.array(sum_errs_train) / all_classifications_train), label='Training data accuracy')
plt.plot(num_latent_feat, 1 - (np.array(sum_errs_test ) / all_classifications_test),  label='Test data accuracy')

plt.grid(linestyle='--')
plt.title('Accuracy vs. Number of Latent Features\n Standard SVD Recommendation Engine\n Data: IBM Watson Studio')
plt.xlabel('Number of Latent Features')
plt.ylabel('Accuracy')
plt.legend(loc='right')
plt.show();    
```""", unsafe_allow_html=True)

u_train, s_train, vt_train = np.linalg.svd(user_item_train)
st.warning(
    f"SVD on the user_item_train matrix (u,s,vt)_train: {u_train.shape}, {s_train.shape}, {vt_train.shape}")

# Users in test set that can be predicted (because also in training set)
# We filter for test_idx - all of the test user ids
u_new = u_train[user_item_train.index.isin(test_idx), :]

# update the shape of s and store in s_new
s_new = np.zeros((len(s), len(s)))
s_new[:len(s), :len(s)] = np.diag(s)

# user_ids in test set that can be predicted because also in the training set
vt_new = vt_train[:, user_item_train.columns.isin(test_arts)]

st.success("The shape of u improve: {}".format(u_new.shape))
st.success("The shape of s improve: {}".format(s_new.shape))
st.success("The shape of vt improve: {}".format(vt_new.shape))

num_latent_feat = np.arange(10, 714+10, 20)

sum_errs_train = []
sum_errs_test = []
intersec_ids = list(set(user_item_train.index).intersection(
    set(test_idx)))  # list of users in both train and test sets

for k in num_latent_feat:
    # restructure with k latent features
    u_train_lat, s_train_lat, vt_train_lat = u_train[:, :k], np.diag(
        s_train[:k]), vt_train[:k, :]
    u_test_lat, vt_test_lat = u_new[:, :k], vt_new[:k, :]

    # take dot product
    user_item_train_est = np.around(
        np.dot(np.dot(u_train_lat, s_train_lat), vt_train_lat))
    user_item_test_est = np.around(
        np.dot(np.dot(u_test_lat, s_train_lat), vt_test_lat))

    # compute error for each prediction to actual value
    diffs_train = np.subtract(user_item_train, user_item_train_est)
    diffs_test = np.subtract(
        user_item_test.loc[intersec_ids, :], user_item_test_est)

    # total errors and keep track of them
    train_err = np.sum(np.sum(np.abs(diffs_train)))
    sum_errs_train.append(train_err)

    test_err = np.sum(np.sum(np.abs(diffs_test)))
    sum_errs_test.append(test_err)

    all_classifications_train = user_item_train_est.shape[0] * \
        user_item_train_est.shape[1]
    all_classifications_test = user_item_test_est.shape[0] * \
        user_item_test_est.shape[1]

plt.figure(figsize=(10, 6))
plt.plot(num_latent_feat, 1 - (np.array(sum_errs_train) /
         all_classifications_train), label='Training data accuracy')
plt.plot(num_latent_feat, 1 - (np.array(sum_errs_test) /
         all_classifications_test),  label='Test data accuracy')
plt.grid(linestyle='--')
plt.title('Accuracy vs. Number of Latent Features\n Standard SVD Recommendation Engine\n Data: IBM Watson Studio')
plt.xlabel('Number of Latent Features')
plt.ylabel('Accuracy')
plt.legend(loc='right')
st.pyplot(plt)

st.markdown("""
### VI. Conclusion
We have performed a **Standard Singular Value Decomposition** (Standard SVD) matrix factorization technique to make article recommendations. We use real data from users on the IBM Watson Studio platform. We have used data on customers' interaction with articles on the platform to generate a User-Item Matrix (unique users for each row and unique articles for each column). However, the User-Item Matrix is a **sparse matrix** with 99.1 percent of elements being zero, indicating no interaction. The remaining .9 percent of elements are ones, indicating an interaction of a user with an article. We can perform Standard SVD because there are no missing values in User-Item Matrix. Even with just one missing value we cannot perform SVD. Alternatively, we could use Funk SVD to perform Matrix Factorization with missing values in the User-Item Matrix.

Next, we have divided the dataset into a training set and a test set to determine the optimal number of latent features for the SVD recommendation engine. **Based on the accuracy of the predictions on the training data and test data, we should choose about 100 latent features**. While the accuracy on the training data keeps improving from approx. 99.0 percent for 10 latent features with more latent features, the accuracy on the test data reaches its maximum of 97.5 percent with just 10 latent features and decreases with more latent features. However, we should take into consideration that the test data set is rather small, with only 20 users available for testing the model's predictions. Therefore, we should place more emphasis on the development of the accuracy of the training set, which starts to flatten above 100 latent features. 

However, **accuracy** is not a good metric to use here in this case, because it does not provide us with a fair assessment of the model's performance. Accuracy, the proportion of correct classifications among all classifications, may be **a poor measure for imbalanced data** like our sparse User-Item Matrix. Even without a model, we would achieve a high accuracy by simply predicting a zero value for every entry. Furthermore, the current assessment **framework is not robust enough to make conclusive results about the model**. The small number of only 20 users in our test set for whom we can make predictions is not sufficient.

Alternatively, we could use an online evaluation technique like **A/B testing** here. We could separate the user groups by userIDs, cookies, or IP addresses to get a 50:50 split. The users of the experimental group get article recommendations from our recommendation engine. The control group gets the current recommendations from IBM Watson. We would **track the number of articles the user interacts with during this experiment**. We could run the experiment until we get enough observations to achieve statistical power of 80 percent, 90 percent, or 95 percent. Statistical power is the probability of rejecting the null hypothesis (H0: recommendation engine has no impact on number of user-article interactions) given the true mean is different from the null (i.e., that the alternative hypothesis is true and that there is indeed a difference in the number of user-article interactions between the two recommendation engines). Increasing the number of observations will increase statistical power. Furthermore, we need to decide if we want to run a one-sided test (alternative hypothesis: our recommendation engine leads to higher user-article interaction) or two-sides test in case we want to check if our recommendation engine performs better or worse than the current one usede by IBM Watson.   
""", unsafe_allow_html=True)

st.markdown("""
## Installation:
You need python3 and the following libraries installed to run the project:
- pandas
- numpy
- matplotlib

Furthermore, this Python file needs to be stored in the working directory:
- top_5.p
- top_10.p
- top_20.p
- project_tests.py
""")

# Files Section
st.markdown("""
## Files:
1. **Python Notebook**
    - Recommendations_with_IBM_NolanM.ipynb

2. **Original Data**:
    - articles_community.csv
    - user-item-interactions.csv

3. **Python code file necessary to run the notebook (Source: Udacity)**
    - top_5.p
    - top_10.p
    - top_20.p
    - project_tests.py
""")


# Data Source Section
st.markdown("""
## Data Source:
IBM Watson
""")

# Licensing, Authors, and Acknowledgements Section
st.markdown("""
### Licensing, Authors, and Acknowledgements:
Thanks to Udacity for the starter code and IBM Watson for providing the anonymized data of its users and their interactions on the IBM Watson Studio’s data platform to be used in this project.
""")

######################
st.markdown("""
<center>
<h6> Created by Nolan Nguyen - Internship 2023</h6>
</center>
""", unsafe_allow_html=True)
