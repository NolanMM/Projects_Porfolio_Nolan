from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Data Modeling with Apache Cassandra Project", page_icon=":bar_chart:")
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

# Set the title of the application
st.title("PowerBI for Seven Sages Brewing Company - NolanM")
st.markdown("""
Project 1 of Udacity's [Data Analysis and Visualization with Microsoft Power BI Nanodegree Program](https://www.udacity.com/course/data-analysis-and-visualization-with-power-BI-nanodegree--nd331)
in **Introduction to Preparing and Modeling Data** course.
""")

# Project Description
st.header("Project Description")
st.markdown("""
The mission is to tame the datasets and create an efficient data model for a small brewing company that will help them better understand 
what products are popular and profitable so they can make smart decisions about what products to prioritize as the company continues to grow. 
The project demonstrates an understanding of core data modeling principles, including the ability to clean, organize and structure data in Power Query, 
to make a date table, to build a data model with the appropriate relationships and filters, and to create a simple report 
using common visualizations and DAX measures.
""")

pbix_file_path = './Project/Project_11/Project_5_Preparing_and_Modeling_Data_with_Power_BI_Nolan_Nguyen.pbix'
data_file_path = './Project/Project_11/PowerBI_for_Seven_Sages_Brewing_Company_Data.zip'
project_file_path = './Project/Project_11/PowerBI_for_Seven_Sages_Brewing_Company_Project_NolanM.zip'

######################


@st.cache_data
def read_binary_file(binary_file_path):
    with open(binary_file_path, 'rb') as file:
        binary_contents = file.read()
    return binary_contents


######################
download_powerbi_file_col, download_data_zip_col, download_project_files_col = st.columns(
    3)

with download_powerbi_file_col:
    st.download_button("Download PowerBI file (.pbix)",
                       read_binary_file(pbix_file_path), file_name="Seven_Sages_Brewing_Company.pbix")
with download_data_zip_col:
    st.download_button("Download Data Files (.zip)",
                       read_binary_file(data_file_path), file_name="Seven_Sages_Brewing_Company_Data.zip")
with download_project_files_col:
    st.download_button("Download Project Files (.zip)",
                       read_binary_file(project_file_path), file_name="Seven_Sages_Brewing_Company_Project_NolanM.zip")

######################
st.header("Project Steps")
st.subheader("Get Data")
st.markdown("""
Used files are `CFO Metrics Tracker.xlsx`, `Customer List (as of FY2021).txt`,
 `SSBC Product Offerings.pdf`, `USD-CAD Exchange Rates.csv`, 
 `Monthly Sales Logs/` downloaded from Udacity and can be found on `Source Files/` folder on this repo.
""")

######################
st.subheader("ETL with Power Query")
st.markdown("""
We used Power Query to make data cleaning/pre-processing on our datasets, that included:
  - Merging 12 monthly sales files into `Full 2021 Sales` query for better analysis.
  - Merging `Customer List (as of FY2021).txt` and `SSBC Product Offerings.pdf` to `Product_CP` query to include all product relevant attributes.
  - Promoted first rows as headers.
  - Removed NULL values in all datasets.
  - Renamed queries and columns with descriptive names.
  - Changed columns' data types to suitable ones.
  - Built dynamic date table that we'll dive into in the next section.
""")

######################
st.subheader("Creating Date Table")
st.markdown("""
A date table has been created using Power Query that is set to dynamically update based on the fact table’s start and end data.
The date table includes standard fields:
  - Calendar month name and number
  - Calendar year
  - Fiscal period
  - Fiscal year
  - Fiscal quarter - Quarter - FY (e.g., Q1 - FY2021)
  
> Note: Seven Sages' Fiscal year begins on October 1st and runs until September 30th. A transaction on Sept 20th 2020 would fall in FY 2020, but a transaction on October 20th would land in FY 2021
""")

st.subheader("Create Data Model (build relationships between tables)")
st.markdown("""
We ended up with one fact table `Full 2021 Sales` and four dimension tables pointing towards it with an active one-to-many relationship.
A snapshot of the data model is provided below and can be found on `SSBC-Data-Model.png` in this repo.
""")
st.image("./Project/Project_11/SSBC_Data_Model.png", use_column_width=True)

st.subheader("Writing DAX Measures")
st.markdown("""
To satisfy the CFO's requirements, we will need to write six measures—to calculate Sales, 
Cost of Sales and Gross Profit Margin in two different currencies.
The following measures have been created using DAX, are present on the data model, and are clearly labeled:
  - Sales in USD ($)
  - Cost of Sales USD ($)
  - Gross Profit Margin (or GPM) in USD (%)
  - Sales in CAD ($)
  - Unit Sales by Product (%)
  - Share of gross profit by Product type (%)
""")

######################
st.subheader("Build a Report")
st.markdown("""
To satisfy the CFO's requirements, our basic version of the report will have two tabs, one summarizing sales by customer and customer type across quarters and would be labeled `Sales and GPM`. 
The second will simply summarize the percentages of gross profit and unit sales by product and would be labeled `Gross Profit and Unit Sales`.
Both tabs have a very brief executive summary at the bottom.
The full PDF report can be found in the `SSBC-Report` file provided in this repo.
""")
st.image("./Project/Project_11/SSBC-Report-Tab1.png", use_column_width=True)
st.image("./Project/Project_11/SSBC-Report-Tab2.png", use_column_width=True)

######################
st.markdown("""
<center>
<h6> Created by Nolan Nguyen - Internship 2023</h6>
</center>
""", unsafe_allow_html=True)
