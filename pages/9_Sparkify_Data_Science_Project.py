import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType

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

import streamlit as st
from dotenv import load_dotenv
from io import StringIO
spark = SparkSession.builder \
    .appName("Sparkify_Project_Spark_Local") \
    .getOrCreate()

load_dotenv(override=True)

st.set_page_config(
    page_title="Sparkify using Spark Project - NolanM",
    page_icon=":shark:",
)
st.write("Spark Session created")

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

cancellation_check_function = udf(lambda x: 1 if x == "Cancellation Confirmation" else 0, IntegerType())
df = df.withColumn("churn", cancellation_check_function("page"))

st.write(df)