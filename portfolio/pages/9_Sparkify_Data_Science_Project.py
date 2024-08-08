import sys
from pyspark.ml.feature import StandardScaler
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
from pyspark.sql.functions import split, sum
import numpy as np
import pandas as pd
import datetime
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.functions import sum as Fsum
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from time import time
from pyspark.sql import Window
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lit
# from pyspark.sql.functions import col, sum as Fsum

import streamlit as st
from dotenv import load_dotenv
from io import StringIO
# spark = SparkSession.builder \
#     .appName("Sparkify_Project_Spark_Local") \
#     .getOrCreate()
spark = SparkSession.builder \
    .appName("Sparkify_Project_Spark_Local") \
    .getOrCreate()

load_dotenv(override=True)

st.set_page_config(
    page_title="Sparkify using Spark Project - NolanM",
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

st.title("Sparkify using Spark Project - NolanM")

st.write("")

st.markdown("""
    ## Objective

    The objective of this project is to analyze and model user behavior data from Sparkify, a fictional digital music service, using Apache Spark. The goal is to build and evaluate machine learning models to predict user churn. This involves the following steps:

    1. **Data Ingestion and Cleaning:**
    - Load user activity data into a Spark DataFrame.
    - Clean and preprocess the data to handle missing values, outliers, and data inconsistencies.

    2. **Feature Engineering:**
    - Create relevant features that capture user behavior, such as session length, number of songs played, and user demographics.

    3. **Modeling:**
    - Utilize Spark's MLlib to build classification models such as Logistic Regression, Random Forest, and Gradient-Boosted Trees.
    - Perform hyperparameter tuning and model selection using cross-validation.

    4. **Evaluation:**
    - Evaluate the performance of the models using appropriate metrics such as precision, recall, F1-score, and ROC-AUC.

    5. **Deployment:**
    - Develop a pipeline for deploying the chosen model to predict user churn on new data.

    The project aims to provide actionable insights into user behavior and identify key factors contributing to churn, enabling Sparkify to enhance user retention strategies.

""")

# Load data
data_path = "./data/Project_9/mini_sparkify_event_data.json"
st.markdown("## Implementation Steps")
st.markdown("### I. Data Ingestion and Cleaning")

df = spark.read.json(data_path)


code_ingestion = """
df = spark.read.json(data_path)
"""
st.write("#### 1. Load the dataset into a Spark DataFrame")
st.code(code_ingestion, language="python")
st.write("The dataset contains the following columns:")
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
df.printSchema()
sys.stdout = old_stdout
st.text(mystdout.getvalue())

st.write(df)

st.write("#### 2. Data Overview")

st.write("The dataset contains {} rows.".format(df.count()))

start_date = df.select(min(to_timestamp(col('ts')/1000)).alias('Start time'))
end_date = df.select(max(to_timestamp(col('ts')/1000)).alias('End time'))
st.write("The first day in the dataset = df.select(min(to_timestamp(col('ts')/1000)).alias('Start time'))")
st.write(start_date)
st.write(f"The last day in the dataset = df.select(max(to_timestamp(col('ts')/1000)).alias('End time'))")
st.write(end_date)

distinct_pages = df.select("page").distinct()

st.write("The dataset contains the following pages:")
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
distinct_pages.show(100, False)
sys.stdout = old_stdout
st.text(mystdout.getvalue())

st.write("")
st.write("##### Define Churn")
st.write("The churn is defined as the event `Cancellation Confirmation`")

cancellation_check_function = udf(
    lambda x: 1 if x == "Cancellation Confirmation" else 0, IntegerType())
df = df.withColumn("churn", cancellation_check_function("page"))

code_churn = """
    cancellation_check_function = udf(lambda x: 1 if x == "Cancellation Confirmation" else 0, IntegerType())
    df = df.withColumn("churn", cancellation_check_function("page"))
"""

st.code(code_churn, language="python")
st.write("#### 3. Df Add Churn Column")
st.write(df)
num_cancelled_users = df.filter(
    df.churn == 1).select("userId").distinct().count()
st.write(f"Number of users who churned: {num_cancelled_users}")

st.markdown("---")
st.write("#### 3.Exploratory Data Analysis")

# Define the window bounds to use Fsum to count for the churn
windowval = Window.partitionBy("userId").rangeBetween(
    Window.unboundedPreceding, Window.unboundedFollowing)

df = df.withColumn("churn", Fsum(col("churn")).over(windowval))

st.write("Missing values in the dataset")

code_missing = """
def missing_values(df, col):
    for col in df.columns:
        missing_count = df.filter((isnan(df[col])) | (df[col].isNull()) | (df[col] == "")).count()
        if missing_count > 0:
            print("{}: {}".format(col, missing_count))
"""

st.code(code_missing, language="python")

st.write("The dataset contains the following missing values:")
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
for col_ in df.columns:
    missing_count = df.filter((isnan(df[col_])) | (
        df[col_].isNull()) | (df[col_] == "")).count()
    if missing_count > 0:
        print("{}: {}".format(col_, missing_count))

sys.stdout = old_stdout
st.text(mystdout.getvalue())
st.markdown("---")
st.write("##### Numerical Exploration")

st.write("##### 3.1 Rate within female and males category")

code_Rate_male_female = """
    stat_df = spark.createDataFrame(df.dropDuplicates(['userId']).collect())
    stat_df_GC = stat_df[['gender', 'churn']]
    print('The avg churn rate of females is:', stat_df_GC.groupby(['gender']).mean().collect()[0][1]*100)
    print('The avg churn rate of males is:', stat_df_GC.groupby(['gender']).mean().collect()[1][1]*100)
"""

st.code(code_Rate_male_female, language="python")

stat_df = spark.createDataFrame(df.dropDuplicates(['userId']).collect())
stat_df_GC = stat_df[['gender', 'churn']]
avg_churn_rate_female = stat_df_GC.groupby(
    ['gender']).mean().collect()[0][1]*100
avg_churn_rate_male = stat_df_GC.groupby(['gender']).mean().collect()[1][1]*100

st.write("The avg churn rate of females is {:.2f}%".format(
    avg_churn_rate_female))
st.write("The avg churn rate of males is {:.2f}%".format(avg_churn_rate_male))

# 2
st.write("##### 3.2 Viewing top 5 cancellations by artist")

code_Top_5_Cancellations = """
    stat_df1 = stat_df[['artist', 'churn']]
    display(stat_df1.groupBy(['artist']).sum().orderBy('sum(churn)', ascending = False).collect()[:5])
"""

st.code(code_Top_5_Cancellations, language="python")
stat_df1 = stat_df[['artist', 'churn']]
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
print(stat_df1.groupBy(['artist']).sum().orderBy(
    'sum(churn)', ascending=False).collect()[:5])
sys.stdout = old_stdout
st.text(mystdout.getvalue())

# 3
st.write("##### 3.3 Rate for “Paid” and “Free” Category")

code_Rate_Paid_Free = """
stat_df1 = stat_df[['level', 'churn']]
print('Proportion of users that chruned from free subscirption', stat_df1.groupBy(['level']).mean().collect()[0][1]*100)
print('Proportion of users that chruned from paid subscirption',stat_df1.groupBy(['level']).mean().collect()[1][1]*100)
"""

st.code(code_Rate_Paid_Free, language="python")

stat_df1 = stat_df[['level', 'churn']]
st.write('Proportion of users that chruned from free subscirption',
         stat_df1.groupBy(['level']).mean().collect()[0][1]*100)
st.write('Proportion of users that chruned from paid subscirption',
         stat_df1.groupBy(['level']).mean().collect()[1][1]*100)

# 4
st.write("##### 3.4 Top 10 States Category")

code_state_category = """
    from pyspark.sql.functions import split, sum
    from pyspark.sql import functions as F

    stat_df1 = stat_df.select('location', 'churn')
    stat_df1 = stat_df1.withColumn('state', split(stat_df1['location'], ',').getItem(1))
    stat_df1 = stat_df1.drop('location')

    print('Viewing top 10 states with churn:\n')
    stat_df1.groupBy('state').agg(F.sum('churn').alias('total_churn')).filter('total_churn > 0').orderBy(F.desc('total_churn')).show(10)
"""

st.code(code_state_category, language="python")

stat_df1 = stat_df.select('location', 'churn')
stat_df1 = stat_df1.withColumn('state', split(
    stat_df1['location'], ',').getItem(1))
stat_df1 = stat_df1.drop('location')
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
print('Viewing top 10 states with churn:\n')
stat_df1.groupBy('state').agg(F.sum('churn').alias('total_churn')).filter(
    'total_churn > 0').orderBy(F.desc('total_churn')).show(10)
sys.stdout = old_stdout
st.text(mystdout.getvalue())

st.markdown("---")
st.write("##### Time Analysis")
st.write("##### Helper Function and Pre processing")

code_helper_function = """
    # Function to retrieve series based on col, value, with normalization
    from pyspark.sql.functions import udf
    from pyspark.sql.types import IntegerType
    import datetime
    def get_series(col, value, normalize):
        if normalize:
            total_count = dfp[col].sum()
            if total_count > 0:
                series = dfp[dfp[col] == value][col].value_counts(normalize=True) * 100
            else:
                series = pd.Series()
        else:
            series = dfp[dfp[col] == value][col].value_counts()
        return series
    
    get_hour = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).hour, IntegerType())
    get_day = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).day, IntegerType())
    get_month = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).month, IntegerType())
    get_weekday = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).strftime('%w'), IntegerType())

    data = df.withColumn('hour', get_hour(df.ts))
    data = data.withColumn('day', get_day(df.ts))
    data = data.withColumn('month', get_month(df.ts))
    data = data.withColumn('week_day', get_weekday(df.ts))
    dfp = data.toPandas()
"""

st.code(code_helper_function, language="python")


def get_series(col_, value, normalize):
    if normalize:
        total_count = dfp[col_].sum()
        if total_count > 0:
            series = dfp[dfp[col_] == value][col_].value_counts(
                normalize=True) * 100
        else:
            series = pd.Series()
    else:
        series = dfp[dfp[col_] == value][col_].value_counts()
    return series


get_hour = udf(lambda x: datetime.datetime.fromtimestamp(
    x / 1000.0).hour, IntegerType())
get_day = udf(lambda x: datetime.datetime.fromtimestamp(
    x / 1000.0).day, IntegerType())
get_month = udf(lambda x: datetime.datetime.fromtimestamp(
    x / 1000.0).month, IntegerType())
get_weekday = udf(lambda x: datetime.datetime.fromtimestamp(
    x / 1000.0).strftime('%w'), IntegerType())

data = df.withColumn('hour', get_hour(df.ts))
data = data.withColumn('day', get_day(df.ts))
data = data.withColumn('month', get_month(df.ts))
data = data.withColumn('week_day', get_weekday(df.ts))
dfp = data.toPandas()


def draw_time(col_, normalize=True, figsize=(16, 4), title=None, label_rotation=0):
    # Normalize to fit with 2 group of users
    df_time = pd.DataFrame({'Cancelled': get_series(col_, 1, normalize),
                            'Active users': get_series(col_, 0, normalize)})

    fig, ax = plt.subplots(figsize=figsize)
    df_time.plot(kind='bar', ax=ax)
    ax.set_ylabel('Percent of users')

    if title is None:
        title = col_

    ax.set_xticklabels(ax.get_xticklabels(), rotation=label_rotation)
    ax.set_title(f'Percent of users took action per {title}')

    st.pyplot(fig)


st.markdown("#### User Activity Analysis")
draw_time('hour', title='Hour of the Day')
############################################################################################
st.markdown("---")
st.write("#### 4. Feature Engineering")
st.markdown("##### Independent Variables")
st.markdown("##### 4.1 Time Since Registration")
code_time_since_register = """
```python
from pyspark.sql.functions import col

feature_1 = data.select('userId', 'registration', 'ts') \
    .withColumn('lifetime', (data.ts - data.registration)) \
    .groupBy('userId') \
    .agg({'lifetime': 'max'}) \
    .withColumnRenamed('max(lifetime)', 'lifetime') \
    .select('userId', (col('lifetime') / 1000 / 3600 / 24).alias('lifetime'))

feature_1.describe().show()
```
"""

st.markdown(code_time_since_register, unsafe_allow_html=True)
feature_1 = data.select('userId', 'registration', 'ts') \
    .withColumn('lifetime', (data.ts - data.registration)) \
    .groupBy('userId') \
    .agg({'lifetime': 'max'}) \
    .withColumnRenamed('max(lifetime)', 'lifetime') \
    .select('userId', (col('lifetime') / 1000 / 3600 / 24).alias('lifetime'))
st.write(feature_1.describe())

############################################################################################
st.markdown("##### 4.2 Total Songs Listened")
code_total_song_listened = """
```python
feature_2 = data \
    .select('userID','song') \
    .groupBy('userID') \
    .count() \
    .withColumnRenamed('count', 'total_songs')
feature_2.describe().show()
```
"""

st.markdown(code_total_song_listened, unsafe_allow_html=True)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
feature_2 = data \
    .select('userID', 'song') \
    .groupBy('userID') \
    .count() \
    .withColumnRenamed('count', 'total_songs')
feature_2.describe().show()
sys.stdout = old_stdout
st.text(mystdout.getvalue())
############################################################################################
st.markdown("##### 4.3 Like of user")
code_like_of_users = """
```python
feature_3 = data \
    .select('userID','page') \
    .where(data.page == 'Thumbs Up') \
    .groupBy('userID') \
    .count() \
    .withColumnRenamed('count', 'num_thumb_up')
feature_3.describe().show()
```
"""

st.markdown(code_like_of_users, unsafe_allow_html=True)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
feature_3 = data \
    .select('userID', 'page') \
    .where(data.page == 'Thumbs Up') \
    .groupBy('userID') \
    .count() \
    .withColumnRenamed('count', 'num_thumb_up')
feature_3.describe().show()
sys.stdout = old_stdout
st.text(mystdout.getvalue())
############################################################################################
st.markdown("##### 4.4 Dislike of user: ( Check for Thumbs Down Page)")
code_dislike_of_users = """
```python
feature_4 = data \
    .select('userID','page') \
    .where(data.page == 'Thumbs Down') \
    .groupBy('userID') \
    .count() \
    .withColumnRenamed('count', 'num_thumb_down')
feature_4.describe().show()
```
"""

st.markdown(code_dislike_of_users, unsafe_allow_html=True)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
feature_4 = data \
    .select('userID', 'page') \
    .where(data.page == 'Thumbs Down') \
    .groupBy('userID') \
    .count() \
    .withColumnRenamed('count', 'num_thumb_down')
feature_4.describe().show()
sys.stdout = old_stdout
st.text(mystdout.getvalue())
##############################################################################################
st.markdown("##### 4.5 Playlist length: (Check for the Add to Playlist Page)")
code_playlist_length = """
```python
feature_5 = data \
    .select('userID','page') \
    .where(data.page == 'Add to Playlist') \
    .groupBy('userID')\
    .count() \
    .withColumnRenamed('count', 'add_to_playlist')
feature_5.describe().show()
```
"""
st.markdown(code_playlist_length, unsafe_allow_html=True)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
feature_5 = data \
    .select('userID', 'page') \
    .where(data.page == 'Add to Playlist') \
    .groupBy('userID')\
    .count() \
    .withColumnRenamed('count', 'add_to_playlist')
feature_5.describe().show()
sys.stdout = old_stdout
st.text(mystdout.getvalue())
###############################################################################################
st.markdown("##### 4.6 Referring friends: (Check for the Add Friend Page)")
code_referring_friend = """
```python
feature_6 = data \
    .select('userID','page') \
    .where(data.page == 'Add Friend') \
    .groupBy('userID') \
    .count() \
    .withColumnRenamed('count', 'add_friend')
feature_6.describe().show()
```
"""
st.markdown(code_referring_friend, unsafe_allow_html=True)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
feature_6 = data \
    .select('userID', 'page') \
    .where(data.page == 'Add Friend') \
    .groupBy('userID') \
    .count() \
    .withColumnRenamed('count', 'add_friend')
feature_6.describe().show()
sys.stdout = old_stdout
st.text(mystdout.getvalue())
###############################################################################################
st.markdown(
    "##### 4.7 Listening Longevity: ( Check for the total listen time each user)")
code_listening_longevity = """
```python
feature_7 = data \
    .select('userID','length') \
    .groupBy('userID') \
    .sum() \
    .withColumnRenamed('sum(length)', 'listen_time')
feature_7.describe().show()
```
"""
st.markdown(code_listening_longevity, unsafe_allow_html=True)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
feature_7 = data \
    .select('userID', 'length') \
    .groupBy('userID') \
    .sum() \
    .withColumnRenamed('sum(length)', 'listen_time')
feature_7.describe().show()
sys.stdout = old_stdout
st.text(mystdout.getvalue())
###############################################################################################
st.markdown(
    "##### 4.8 Songs per Session")
st.markdown("###### Avange song played per Sessions Count the number user hit NextSong group by sessionId, userId and take avarange of the number of song be played corresponding to that SessionID")
code_song_per_session = """
```python
feature_8 = data.where('page == "NextSong"') \
    .groupby(['userId', 'sessionId']) \
    .count() \
    .groupby(['userId']) \
    .agg({'count':'avg'}) \
    .withColumnRenamed('avg(count)', 'avg_songs_played')
```
"""
st.markdown(code_song_per_session, unsafe_allow_html=True)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
feature_8 = data.where('page == "NextSong"') \
    .groupby(['userId', 'sessionId']) \
    .count() \
    .groupby(['userId']) \
    .agg({'count': 'avg'}) \
    .withColumnRenamed('avg(count)', 'avg_songs_played')
feature_8.describe().show()
sys.stdout = old_stdout
st.text(mystdout.getvalue())
###############################################################################################
st.markdown(
    "##### 4.9 Gender: (Replace M and F by 0 and 1)")
code_gender_numeric = """
```python
feature_9 = data \
    .select("userId", "gender") \
    .dropDuplicates() \
    .replace(['M', 'F'], ['0', '1'], 'gender') \
    .select('userId', col('gender').cast('int'))
```
"""
st.markdown(code_gender_numeric, unsafe_allow_html=True)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
feature_9 = data \
    .select("userId", "gender") \
    .dropDuplicates() \
    .replace(['M', 'F'], ['0', '1'], 'gender') \
    .select('userId', col('gender').cast('int'))
feature_9.describe().show()
sys.stdout = old_stdout
st.text(mystdout.getvalue())
###############################################################################################
st.markdown(
    "##### 4.10 Number of Artists Listened: ( Count the total of Artists that each user listen to)")
code_artist_listened = """
    feature_10 = data \
        .filter(data.page=="NextSong") \
        .select("userId", "artist") \
        .dropDuplicates() \
        .groupby("userId") \
        .count() \
        .withColumnRenamed("count", "artist_count")
"""
st.code(code_artist_listened, language="python")
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
feature_10 = data \
    .filter(data.page == "NextSong") \
    .select("userId", "artist") \
    .dropDuplicates() \
    .groupby("userId") \
    .count() \
    .withColumnRenamed("count", "artist_count")
feature_10.describe().show()
sys.stdout = old_stdout
st.text(mystdout.getvalue())
###############################################################################################
st.markdown("##### Independent Variables")
st.markdown("""
    - Create a dependent variable from churn column
    - Create numeric categories of churn label as 1 = churn and 0 = no churn.
""", unsafe_allow_html=True)
code_dependent_variable = """
```python
feature_10 = data \
    .filter(data.page=="NextSong") \
    .select("userId", "artist") \
    .dropDuplicates() \
    .groupby("userId") \
    .count() \
    .withColumnRenamed("count", "artist_count")
```
"""
st.markdown(code_dependent_variable, unsafe_allow_html=True)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
label_churn = data \
    .select('userId', col('churn').alias('label')) \
    .dropDuplicates()
label_churn.describe().show()
sys.stdout = old_stdout
st.text(mystdout.getvalue())
###############################################################################################
st.markdown("##### Construct the DataSet for Model")
st.markdown("""
    - Joint all the feature table and label table into dataframe with outer join based on userID and fillna with 0
""")
code_construct_dataset = """
```python
data = feature_1.join(feature_2,'userID','outer') \
    .join(feature_3,'userID','outer') \
    .join(feature_4,'userID','outer') \
    .join(feature_5,'userID','outer') \
    .join(feature_6,'userID','outer') \
    .join(feature_7,'userID','outer') \
    .join(feature_8,'userID','outer') \
    .join(feature_9,'userID','outer') \
    .join(feature_10,'userID','outer') \
    .join(label_churn,'userID','outer') \
    .drop('userID') \
    .fillna(0)
```
"""
st.markdown(code_construct_dataset, unsafe_allow_html=True)
data = feature_1.join(feature_2, 'userID', 'outer') \
    .join(feature_3, 'userID', 'outer') \
    .join(feature_4, 'userID', 'outer') \
    .join(feature_5, 'userID', 'outer') \
    .join(feature_6, 'userID', 'outer') \
    .join(feature_7, 'userID', 'outer') \
    .join(feature_8, 'userID', 'outer') \
    .join(feature_9, 'userID', 'outer') \
    .join(feature_10, 'userID', 'outer') \
    .join(label_churn, 'userID', 'outer') \
    .drop('userID') \
    .fillna(0)
st.write(data)
############################################################################################
st.markdown("---")
st.write("#### 5. Modeling Engineering")
st.markdown("""
    Split the full dataset into train, test, and validation sets. 
    Test out several of the machine learning methods you learned. 
    Evaluate the accuracy of the various models, tuning parameters as necessary. 
    Determine your winning model based on test accuracy and report results on the validation set. 
    Since the churned users are a fairly small subset, I suggest using F1 score as the metric to optimize.
""")
st.markdown("##### 5.1 Normalizing")
st.markdown("""
- Turn to normalization of our features that will assurance any independent variable alone or few of them would not influence the dependent variable 
    to an extent where other features become redundant.
""")

code_normalizing = """
    cols = ["lifetime", "total_songs", "num_thumb_up", 'num_thumb_down','add_to_playlist','add_friend','listen_time','avg_songs_played','gender','artist_count']
    assembler = VectorAssembler(inputCols=cols, outputCol="NumFeatures")
    data = assembler.transform(data)
"""
st.code(code_normalizing, language="python")
cols = ["lifetime", "total_songs", "num_thumb_up", 'num_thumb_down', 'add_to_playlist',
        'add_friend', 'listen_time', 'avg_songs_played', 'gender', 'artist_count']
assembler = VectorAssembler(inputCols=cols, outputCol="NumFeatures")
data = assembler.transform(data)
st.write(data)
############################################################################################
st.markdown("##### 5.2 Scaling")
st.markdown("""
- Substract the mean of each feature from every value of that feature and then divide it by standard deviation of each feature using withStd = True
""")

code_scaling = """
    from pyspark.ml.feature import StandardScaler
    scaler = StandardScaler(inputCol="NumFeatures",outputCol="features", withStd=True)
    scalerModel = scaler.fit(data)
    data = scalerModel.transform(data)
"""
st.code(code_scaling, language="python")
scaler = StandardScaler(inputCol="NumFeatures",
                        outputCol="features", withStd=True)
scalerModel = scaler.fit(data)
data = scalerModel.transform(data)
st.write(data)
############################################################################################
st.markdown("##### 5.3 Train - Validation — Test Dataset Split")
st.markdown("""
- Take 70% for TrainData
- Take 18% for Validation
- Take 12% for Test
**Note**:: The seed=42 parameter of randomSplit() ensures that same pseudorandom number is generated every-time by preserving the copy of first time generated pseudo number.
""", unsafe_allow_html=True)

code_scaling = """
    train, rest = data.randomSplit([0.7, 0.3], seed=42)
    validation, test = rest.randomSplit([0.6, 0.4], seed=42)
"""
train, rest = data.randomSplit([0.7, 0.3], seed=42)
validation, test = rest.randomSplit([0.6, 0.4], seed=42)
st.write("Train Data")
st.write(train)

############################################################################################
st.markdown("---")
st.markdown("#### 6. Model Model")
st.markdown("##### 6.1 Base Model")
st.markdown("""
##### using MulticlassClassificationEvaluator to mesures how 0 label and 1 label are performing
""")

code_base_model = """
```python
# Mesure label 0
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lit

# Take the test dataset and create prediction column and filled with all 0 to test with the evaluator
test_set_base_0 = test.withColumn('prediction', lit(0.0))
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

# Evaluate the dataset based all 0 for accuracy and f1 score
print('Accuracy: {}'.format(evaluator.evaluate(test_set_base_0, {evaluator.metricName: "accuracy"})))
print('F1 Score:{}'.format(evaluator.evaluate(test_set_base_0, {evaluator.metricName: "f1"})))

# Mesure label 1
# Take the test dataset and create prediction column and filled with all 0 to test with the evaluator
test_set_base_1 = test.withColumn('prediction', lit(1.0))
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

# Evaluate the dataset based all 1 for accuracy and f1 score
print('Accuracy: {}'.format(evaluator.evaluate(test_set_base_1, {evaluator.metricName: "accuracy"})))
print('F1 Score:{}'.format(evaluator.evaluate(test_set_base_1, {evaluator.metricName: "f1"})))
```
"""
st.markdown(code_base_model, unsafe_allow_html=True)

# Take the test dataset and create prediction column and filled with all 0 to test with the evaluator
test_set_base_0 = test.withColumn('prediction', lit(0.0))
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

# Evaluate the dataset based all 0 for accuracy and f1 score
st.write('Accuracy Mesure label 0: {}'.format(evaluator.evaluate(
    test_set_base_0, {evaluator.metricName: "accuracy"})))
st.write('F1 Score Mesure label 0: {}'.format(evaluator.evaluate(
    test_set_base_0, {evaluator.metricName: "f1"})))

# Mesure label 1
# Take the test dataset and create prediction column and filled with all 0 to test with the evaluator
test_set_base_1 = test.withColumn('prediction', lit(1.0))
evaluator_1 = MulticlassClassificationEvaluator(predictionCol="prediction")

# Evaluate the dataset based all 1 for accuracy and f1 score
st.write('Accuracy Mesure label 1: {}'.format(evaluator_1.evaluate(
    test_set_base_1, {evaluator_1.metricName: "accuracy"})))
st.write('F1 Score Mesure label 1:{}'.format(evaluator_1.evaluate(
    test_set_base_1, {evaluator_1.metricName: "f1"})))


st.markdown("##### 6.2 Logistic Regression Model")
st.markdown("""
##### Using Logistic Regression Model measure the performance
""")

code_logistic_model = """
```python
# initialize classifier
lr = LogisticRegression(maxIter=10)

# set evaluator
f1_evaluator = MulticlassClassificationEvaluator(metricName='f1')   # Based on f1 score

# build empty paramGrid
paramGrid = ParamGridBuilder().build()

# data will be split into three folds during cross-validation (numFolds=3)
crossval_lr = CrossValidator(estimator=lr, evaluator=f1_evaluator, estimatorParamMaps=paramGrid, numFolds=3) 

# Fit data to crossval_lr
FittedModel_lr = crossval_lr.fit(train)

# Take the List of result metrics 
FittedModel_lr.avgMetrics

# Transform the data and collect the result
results_lr = FittedModel_lr.transform(validation)

# Evaluate the data
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print('Logistic Regression Metrics Result')
print('Accuracy: {}'.format(evaluator.evaluate(results_lr, {evaluator.metricName: "accuracy"})))
print('F1 Score:{}'.format(evaluator.evaluate(results_lr, {evaluator.metricName: "f1"})))
```
"""
st.markdown(code_logistic_model, unsafe_allow_html=True)

# initialize classifier
lr = LogisticRegression(maxIter=10)

# set evaluator
f1_evaluator = MulticlassClassificationEvaluator(
    metricName='f1')   # Based on f1 score

# build empty paramGrid
paramGrid = ParamGridBuilder().build()
# data will be split into three folds during cross-validation (numFolds=3)
crossval_lr = CrossValidator(
    estimator=lr, evaluator=f1_evaluator, estimatorParamMaps=paramGrid, numFolds=3)

# Fit data to crossval_lr
FittedModel_lr = crossval_lr.fit(train)

# Take the List of result metrics
FittedModel_lr.avgMetrics

# Transform the data and collect the result
results_lr = FittedModel_lr.transform(validation)

# Evaluate the data
evaluator_logistic = MulticlassClassificationEvaluator(
    predictionCol="prediction")

st.write('Logistic Regression Metrics Result')
st.write('Accuracy Logistic Regression Metrics: {}'.format(evaluator_logistic.evaluate(
    results_lr, {evaluator_logistic.metricName: "accuracy"})))
st.write('F1 Score Logistic Regression Metrics:{}'.format(evaluator_logistic.evaluate(
    results_lr, {evaluator_logistic.metricName: "f1"})))

############################################################################################
st.markdown("##### 6.3 Gradient Boosted Tree")
st.markdown("""
##### Using Gradient Boosted Tree measure the performance
""")

code_gradient_model = """
```python
# Initialize classifier
GradBoostTree = GBTClassifier(maxIter=5,seed=42)

# Set evaluator (f1 score)
f1_evaluator = MulticlassClassificationEvaluator(metricName='f1') 

# Build paramGrid
paramGrid = ParamGridBuilder().build()
crossval_GradBoostTree = CrossValidator(estimator=GradBoostTree, estimatorParamMaps=paramGrid, evaluator=f1_evaluator, numFolds=5)

# Fit data to crossval_GradBoostTree
FittedModel_GradBoostTree = crossval_GradBoostTree.fit(train)

# Take the List of result metrics 
FittedModel_GradBoostTree.avgMetrics

# Transform the data and collect the result
results_GradBoostTree = FittedModel_GradBoostTree.transform(validation)

# Evaluate the data
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

print('Gradient Boosted Trees Metrics:')
print('Accuracy: {}'.format(evaluator.evaluate(results_GradBoostTree, {evaluator.metricName: "accuracy"})))
print('F1 Score:{}'.format(evaluator.evaluate(results_GradBoostTree, {evaluator.metricName: "f1"})))
```
"""
st.markdown(code_gradient_model, unsafe_allow_html=True)

# initialize classifier
GradBoostTree = GBTClassifier(maxIter=5, seed=42)
# set evaluator
f1_evaluator = MulticlassClassificationEvaluator(metricName='f1')  # f1 score
# build paramGrid
paramGrid = ParamGridBuilder().build()
crossval_GradBoostTree = CrossValidator(
    estimator=GradBoostTree, estimatorParamMaps=paramGrid, evaluator=f1_evaluator, numFolds=5)

# Fit data to crossval_GradBoostTree
FittedModel_GradBoostTree = crossval_GradBoostTree.fit(train)

# Take the List of result metrics
FittedModel_GradBoostTree.avgMetrics

# Transform the data and collect the result
results_GradBoostTree = FittedModel_GradBoostTree.transform(validation)

# Evaluate the data
evaluator_gradient = MulticlassClassificationEvaluator(
    predictionCol="prediction")

st.write('Gradient Boosted Trees Metrics:')
st.write('Accuracy Gradient Boosted Trees: {}'.format(evaluator_gradient.evaluate(
    results_GradBoostTree, {evaluator_gradient.metricName: "accuracy"})))
st.write('F1 Score Gradient Boosted Trees:{}'.format(evaluator_gradient.evaluate(
    results_GradBoostTree, {evaluator_gradient.metricName: "f1"})))

############################################################################################
st.markdown("---")
st.markdown("#### 7. Test Deployment & Hyperparameter Tuning")
st.markdown("""
##### using MulticlassClassificationEvaluator to mesures how 0 label and 1 label are performing
""")

code_hypertuning_model = """
```python
f1_evaluator = MulticlassClassificationEvaluator(metricName='f1')
lr = LogisticRegression()

paramGrid = ParamGridBuilder().addGrid(lr.maxIter, [10,12]).addGrid(lr.regParam, [0,0.1]).addGrid(lr.elasticNetParam, [0.001,0.01]).build()

lr_cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid,evaluator=f1_evaluator, numFolds=3)

lrModel = lr_cv.fit(train)
bestModel = lrModel.bestModel
results_final = bestModel.transform(test)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print('Test set metrics:') 
print('Accuracy: {}'.format(evaluator.evaluate(results_final, {evaluator.metricName: "accuracy"})))
print('F-1 Score:{}'.format(evaluator.evaluate(results_final, {evaluator.metricName: "f1"})))
```
"""
st.markdown(code_hypertuning_model, unsafe_allow_html=True)

f1_evaluator = MulticlassClassificationEvaluator(metricName='f1')
lr = LogisticRegression()

paramGrid = ParamGridBuilder().addGrid(lr.maxIter, [10, 12]).addGrid(
    lr.regParam, [0, 0.1]).addGrid(lr.elasticNetParam, [0.001, 0.01]).build()

lr_cv = CrossValidator(
    estimator=lr, estimatorParamMaps=paramGrid, evaluator=f1_evaluator, numFolds=3)

lrModel = lr_cv.fit(train)
bestModel = lrModel.bestModel
results_final = bestModel.transform(test)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
st.write('Test set metrics:')
st.write('Accuracy: {}'.format(evaluator.evaluate(
    results_final, {evaluator.metricName: "accuracy"})))
st.write('F-1 Score:{}'.format(evaluator.evaluate(results_final,
         {evaluator.metricName: "f1"})))

############################################################################################
st.markdown("---")
st.markdown("#### 8. Calculate Feature Importances")
st.markdown("""
- Calculate the feature importances of the best model
""")
code_feature_importances = """
```python
def calculate_feature_importances(coefficients, feature_names):
    feat_imp = [0 - x if x < 0 else x for x in coefficients]
    feature_importances = list(zip(feature_names, feat_imp))
    return feat_imp, feature_importances

feat_imp, feature_importances = calculate_feature_importances(bestModel.coefficients, cols)

# Plotting the vertical bar chart
plt.bar(cols, feat_imp, align='center')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Feature Importances')

plt.xticks(rotation=45) # Rotate to Vertical bar chart
plt.tight_layout()
plt.show()
```
"""
st.markdown(code_feature_importances, unsafe_allow_html=True)


def calculate_feature_importances(coefficients, feature_names):
    feat_imp = [0 - x if x < 0 else x for x in coefficients]
    feature_importances = list(zip(feature_names, feat_imp))
    fig, ax = plt.subplots()
    ax.bar(feature_names, feat_imp, align='center')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance Score')
    ax.set_title('Feature Importances')
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    plt.tight_layout()
    # Display the plot
    st.pyplot(fig)
    return feat_imp, feature_importances


feat_imp, feature_importances = calculate_feature_importances(
    bestModel.coefficients, cols)
