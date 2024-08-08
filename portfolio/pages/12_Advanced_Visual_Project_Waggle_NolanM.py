from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Building a Power BI Report for Waggle", page_icon=":bar_chart:")
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
st.title("Building a Power BI Report for Waggle")
st.subheader(
    "Project 2 of Udacity's Data Analysis and Visualization with Microsoft Power BI Nanodegree Program")
st.markdown("[Data Analysis and Visualization with Microsoft Power BI Nanodegree Program](https://www.udacity.com/course/data-analysis-and-visualization-with-power-BI-nanodegree--nd331)")

st.markdown("""
## Project Description:
Waggle is a startup that makes smart devices for pets. Recently, they have been thrilled by the success of their new Lapdog device, a fitness collar that lets owners track their dog’s steps, alerts them when it’s time for a walk, and even repels fleas! Reviews have been fantastic, sales are growing, and—best of all—the product really works!

The product team distributed 1,000 Lapcat prototypes for field testing. Now, after months of data collection, we have been tasked with delivering a boardroom-ready Power BI report that tells the story of how the Lapcat data compares to findings from the dog collar Lapdog devices to either help convince the CEO that Lapcat is the next big thing or a costly mistake to be avoided.

""")

pbix_file_path = './Project/Project_12/Advanced_Visual_Project_Waggle_Result_NolanM.pbix'
data_file_path = './Project/Project_12/Advanced_Visual_Project_Waggle_Nolan_Data.zip'
project_file_path = './Project/Project_12/Advanced_Visual_Project_Waggle_Nolan_Project.zip'

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
                       read_binary_file(pbix_file_path), file_name="Advanced_Visual_Project_Waggle.pbix")
with download_data_zip_col:
    st.download_button("Download Data Files (.zip)",
                       read_binary_file(data_file_path), file_name="Advanced_Visual_Project_Waggle_Data.zip")
with download_project_files_col:
    st.download_button("Download Project Files (.zip)",
                       read_binary_file(project_file_path), file_name="Advanced_Visual_Project_Waggle_NolanM_Project.zip")

st.markdown("Below is a quick demonstration about the project components.")
# Data Model Section
st.markdown("### Data Model:")
st.image("./Project/Project_12/Waggle_Data_Model.png",
         caption="Waggle Data Model", width=1200)

# Report Requirements Section
st.markdown("""
### Report Requirements:
- The CEO is curious about the following questions:
  - Did the average daily steps increase for cats wearing the device as they did for dogs?
  - Were owners of Lapcat devices as satisfied with the product as Lapdog owners?
- The Chief Marketing Officer would like the report to be “on-brand” by including only colors from the Waggle color palette, the Waggle logo, and other approved company logos and icons.
""")
col_1, col_2, col_3 = st.columns(3)
with col_2:
    st.image("./Project/Project_12/Waggle-color-palette.png",
             caption="Waggle color palette", width=400)
st.markdown("""
- The product team trusts us to incorporate other visuals and insights as we see fit but is most interested in comparisons between the dogs and cats using Waggle devices as well as any information about the families who own the pets. They would also like slicers to help them filter and explore on their own.
- The report should include: 
  - At least five slicers on each page with at least one example of a drop-down slicer, at least one example of a slider slicer, at least one example of a hierarchy slicer, at least one example of a slicer with “Select All” enabled, and one example of a slicer with the search box enabled.
  - At least two bookmark features. One must allow users to dynamically swap one visual out with a different one and another must reset all applied filters on the page.
  - Buttons that help users navigate the report tabs. They must respond when users hover over them by changing color or size.
""")

# Report Tabs
st.markdown("""
The report is to include 3 tabs:
- The first page should highlight the CEO’s business questions, specifically calling out the differences in average step count and average user rating between Lapdog and Lapcat devices.
- The second page should focus on insights related to pets using the device.
- The third page should focus on insights related to the families that own the pets.
""")

st.markdown("Full PDF report can be found on `Advanced_Visual_Project_Waggle_Result_NolanM.pdf` file provided on this repo.")

# Report Tab 1
st.markdown("""
### Report Tab 1:
To address the CEO’s questions:
- 2 visualizations were plotted to highlight the difference between `average daily steps` over time recorded on Lapdog devices vs. Lapcat devices displaying the trend over time by year and month.
- 2 visualizations highlighted the difference between the customer `ratings` for Lapdog devices vs. Lapcat devices in addition to the number of rates.
""")
st.image("./Project/Project_12/Advanced_Visual_Project_Waggle_Result_NolanM_Page_1.jpg",
         caption="Waggle Report Tab 1")

# Report Tab 2
st.markdown("""
### Report Tab 2:
To drive insights from the `pets` dataset, the second tab included:
- A visualization that shows the `breed` distribution of cats and dogs.
- 2 visualizations that highlighted both `gender` and `age` distributions along the dataset with `pet type` as hue.
""")
st.image("./Project/Project_12/Advanced_Visual_Project_Waggle_Result_NolanM_Page_2.jpg",
         caption="Waggle Report Tab 2")

# Report Tab 3
st.markdown("""
### Report Tab 3:
To drive insights from the `family` dataset, the third tab included:
- A table that shows important family data.
- A card that shows the count of total pets on the dataset, and has 2 bookmark buttons to show only cat or dog counts.
- A visualization that shows the relation between `household income` and `number of owned pets` along the dataset with `pet type` as hue.
""")
st.image("./Project/Project_12/Advanced_Visual_Project_Waggle_Result_NolanM_Page_3.jpg",
         caption="Waggle Report Tab 3")

######################
st.markdown("""
<center>
<h6> Created by Nolan Nguyen - Internship 2023</h6>
</center>
""", unsafe_allow_html=True)
