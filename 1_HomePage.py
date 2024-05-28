from pathlib import Path
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv(override=True)

current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / os.getenv("CSS_DIR")
resume_file = current_dir / os.getenv("RESUME_DIR")
profile_pic = current_dir / os.getenv("PROFILE_PIC_DIR")

PAGE_TITLE = "Project Porfolio | NolanM"
PAGE_ICON = ":wave:"
NAME = "Minh Nguyen"
NICKNAME = "NolanM"
DESCRIPTION = """
Data Scientist/Analyst, assisting enterprises by supporting data-driven decision-making.
"""
EMAIL = "minhlenguyen02@gmail.com"

SOCIAL_MEDIA = {
    "LinkedIn": "https://www.linkedin.com/in/nolan2810/",
    "GitHub": "https://github.com/NolanMM"
}
PROJECTS = {
    "🏆 Sales Dashboard - Comparing sales across three stores": "https://youtu.be/Sb0A9i6d320",
    "🏆 Income and Expense Tracker - Web app with NoSQL database": "https://youtu.be/3egaMfE9388",
    "🏆 Desktop Application - Excel2CSV converter with user settings & menubar": "https://youtu.be/LzCfNanQ_9c",
    "🏆 MyToolBelt - Custom MS Excel add-in to combine Python & Excel": "https://pythonandvba.com/mytoolbelt/",
}

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
st.sidebar.header("Project NolanM")
st.markdown(
    """
        <style>
            .st-emotion-cache-13ln4jf.ea3mdgi5 {
                max-width: 900px;
            }
        </style>
    """, unsafe_allow_html=True)

# --- LOAD CSS, PDF & PROFIL PIC ---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
with open(resume_file, "rb") as pdf_file:
    PDFbyte = pdf_file.read()
profile_pic = Image.open(profile_pic)


# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small")
with col1:
    st.image(profile_pic, width=230)

with col2:
    st.title(NAME)
    st.write(DESCRIPTION)
    st.download_button(
        label=" 📄 Download Resume",
        data=PDFbyte,
        file_name=resume_file.name,
        mime="application/octet-stream",
    )
    st.write("📫", EMAIL)

# --- SOCIAL LINKS ---
    cols = st.columns(len(SOCIAL_MEDIA))
    for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
        cols[index].write(f"[{platform}]({link})")


# --- EXPERIENCE & QUALIFICATIONS ---
st.write('\n')
st.subheader("Experience & Qualifications")
st.write("---")
st.write(
    """
- ✔️ 2 Years expereince extracting actionable insights from data
- ✔️ Strong hands on experience and knowledge in Python and Excel
- ✔️ Good understanding of statistical principles and their respective applications
- ✔️ Excellent team-player and displaying strong sense of initiative on tasks
"""
)

# --- SKILLS ---
st.write('\n')
st.subheader("TECHNICAL SKILLS")
st.write("---")
st.write(
    """
- 👩‍💻 Programming Languages: Python, C#, Java, Scala, C, C++, HTML, CSS, JavaScript
- 🔗 Version Control: GitHub, Azure DevOps, Microsoft Teams, Jira
- 🗄️ Database Technology: MySQL, PostgreSQL, Apache SQL, MongoDB
- 🖥️ Front-end Frameworks: React.js, Bootstrap, Tailwind CSS
- 📡 Back-end Frameworks: Flask, ASP .NET Core, Django, Akka HTTP
- 🛢️ Big Data Ecosystem: Data Warehouse, Data Lake, Data Lake House
- 🧠 AI and Machine Learning: Computer Vision, Natural Language Processing, Conversational AI with Azure
"""
)

# --- WORK HISTORY ---
st.write('\n\n')
st.subheader("WORK HISTORY")
st.write("---")

# --- JOB 1
st.write("🚧", "**Information Technology Intern | VinFast Auto Canada Inc**")
st.write("05/2024 - Now")
st.write(
    """
- ► Assist in the development and customization of Salesforce applications to meet business requirements.
- ► Support the integration of Salesforce with other systems and platforms using API techniques.
- ► Contribute to the design and implementation of data models, security models, and workflows within Salesforce.
- ► Participate in the full software development lifecycle, including analysis, design, coding, testing, and deployment.
- ► Collaborate with cross-functional teams to gather and analyze requirements and provide technical support.
- ► Document technical processes, coding best practices, and system configurations.
- ► Engage in continuous learning activities to enhance your knowledge of Salesforce technologies and best practices.
"""
)

# --- JOB 2
st.write('\n')
st.write("🚧", "**Data Analysis Intern | FPT Canada**")
st.write("05/2023 - 09/2023")
st.write(
    """
- ► Developed custom dashboards and analytical presentations targeting key business domains to support strategic decision-making.
- ► Applied advanced data analytics techniques to process and analyze large datasets, enhancing data clarity and utility.
- ► Streamlined data management processes by implementing the CRISP-DM framework to collect, purify, and dissect complex datasets from diverse sources, enhancing data reliability and analytical accuracy.
- ► Collaborated with international research teams, including a three-person group at Mila (Quebec AI Institute), to advance AI and analytics knowledge, resulting in actionable strategies for complex data.
"""
)

# --- Projects & Accomplishments ---
st.write('\n')
st.subheader("Projects & Accomplishments")
st.write("---")
for project, link in PROJECTS.items():
    st.write(f"[{project}]({link})")