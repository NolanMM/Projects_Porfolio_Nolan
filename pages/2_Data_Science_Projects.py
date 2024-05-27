from pathlib import Path
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import os

#load_dotenv(override=True)

# st.set_page_config(page_title="Data_Science_Project", page_icon=":chart_with_upwards_trend:")
# Project = ["🏆 Sales Dashboard - Comparing sales across three stores", "🏆 Income and Expense Tracker - Web app with NoSQL database", "🏆 Desktop Application - Excel2CSV converter with user settings & menubar", "🏆 MyToolBelt - Custom MS Excel add-in to combine Python & Excel"]

# if "Current_Data_Science_Project" not in st.session_state:
#     st.session_state.Current_Data_Science_Project = None
#     project = st.selectbox('Select a project', Project)
#     btn_submit_project = st.button('Submit', on_click=Test_Function, args=(project,))

# if btn_submit_project:
#     st.session_state.Current_Data_Science_Project = project
#     st.write(f"Project Choose: {st.session_state.Current_Data_Science_Project}")