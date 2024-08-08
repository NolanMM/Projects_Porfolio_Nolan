from pathlib import Path
from streamlit_tags import st_tags, st_tags_sidebar
import streamlit as st

st.set_page_config(
    page_title="Google Trend Analysis Project", page_icon=":bar_chart:")
st.sidebar.header("Project NolanM")
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = "./styles/main.css"

with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


@st.cache_data
def load_country_list():
    country_list = []
    with open("./data/Project_14/Country_list.txt") as file:
        for line in file:
            country_list.append(line.strip())
    return country_list


st.markdown(
    """
        <style>
            .st-emotion-cache-13ln4jf.ea3mdgi5 {
                max-width: 1200px;
            }
        </style>
    """, unsafe_allow_html=True)

# Title and Project Information
st.title("Google Trend Analysis Project")
st.markdown(
    """
    This project is a Google Trend Analysis project that will analyze the search trends of a specific keyword.
    The project will use the pytrends library to fetch the data from Google Trends and analyze the data using
    various data visualization techniques. The project will also use the Prophet library to forecast the future
    trends of the keyword. The project will be implemented using the Streamlit library.
    """
)
if not all(key in st.session_state for key in ("time_range_1_section", "country_option_1_section", "keyword", "specific_date_option", "start_date", "end_date", "nation_2")):
    st.session_state.specific_date_option = False
    st.session_state.start_date = False
    st.session_state.end_date = False
    st.session_state.nation = ""
    st.session_state.keyword = []

# Trending Data Now
st.header("Trending Now")
st.write("The following are the trending topics now")
country_option_1_section, time_range_1_section, export_button_1_section, _, _, _, _, _ = st.columns(
    8)


# Take input from the user with the list of keywords and the time period with the nation
st.header("Input Data")
keywords = st_tags(label="Please Enter List of Key Words (Max 5)",
                   text="Type and press enter", maxtags=5, value=[])
if keywords:
    st.session_state.keyword = keywords
specific_date_option = st.checkbox("Find the trend for a specific date range")
if specific_date_option:
    st.write("Enter the time period for the analysis")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    st.session_state.specific_date_option = True
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date

country_list = load_country_list()
st.write("Select the nation for the analysis")
nation = st.selectbox(
    "Select Nation", country_list)

analysis_button = st.button("Analyze Trending Data")
if analysis_button:
    st.session_state.nation = nation
    if len(st.session_state.keyword) == 0:
        st.error("Please enter the list of keywords")
    else:
        # When the keyword is valid
        st.write(keywords)
        st.write(nation)
        if not st.session_state.specific_date_option:
            st.session_state.start_date = "today"
            st.session_state.end_date = "1-m"
