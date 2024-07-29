from io import StringIO
from pathlib import Path
import sys
import streamlit as st
import os
import glob
import csv
import pandas as pd
from cassandra.cluster import Cluster

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

# Hero Section
st.title("Data Modeling with Apache Cassandra Project - NolanM")
st.markdown(
    """
    ## **Objective**
    <p>A startup called Sparkify wants to analyze the data they've been collecting on songs and user activity on their new music streaming app. The analysis team is particularly interested in understanding what songs users are listening to. Currently, there is no easy way to query the data to generate the results, since the data reside in a directory of CSV files on user activity on the app.</p>

    <p>They'd like a data engineer to create an Apache Cassandra database which can create queries on song play data to answer the questions. </p>

    #### Project Datasets

    For this project, you'll be working with one dataset: **event_data**. The directory of CSV files partitioned by date. Here are examples of filepaths to two files in the dataset:

    >**event_data/2018-11-08-events.csv**<br>
    >**event_data/2018-11-09-events.csv**

    Below is an example of what an original single event data file, **2018-11-08-events.csv**, looks like.
    ```
    {
        "artist": "Slipknot", 
        "auth": "Logged In", 
        "firstName": "Aiden", 
        "gender": "M", 
        "itemInSession": 0, 
        "lastName": "Ramirez", 
        "length": 192.57424, 
        "lvevel": "paid", 
        "location": "New York-Newark-Jersey City,NY-NJ-PA"
        "method": "PUT"    
        "page": "NextSong"
        "regisgration":1.54028E+12
        "sessionId":19
        "song":"Opinum Of The People"
        "status":200
        "ts":1.5416E+12
        "userId":20    
    }
    ```
    """,  unsafe_allow_html=True)

# Load data
data_path = "./data/Project_10/dataset/event_data"
st.markdown("## Implementation Steps")
st.markdown("""
### I. Data Preprocessing
##### - Creating list of filepaths to process original event csv data files
##### - Processing the files to create the data file csv that will be used for Apache Casssandra tables
""")
# Create a list of files and collect all filepath
file_path_list = glob.glob(os.path.join(data_path, '*'))


def get_all_event_data(file_path_list):
    full_data_rows_list = []

    # for every filepath in the file path list
    for f in file_path_list:

        # reading csv file
        with open(f, 'r', encoding='utf8', newline='') as csvfile:
            # creating a csv reader object
            csvreader = csv.reader(csvfile)
            next(csvreader)

            # extracting each data row one by one and append it
            for line in csvreader:
                full_data_rows_list.append(line)

    return full_data_rows_list


def create_event_datafile_new(full_data_row_list):
    # creating a smaller event data csv file called event_datafile_new.csv
    # that will be used to insert data into the Apache Cassandra tables
    csv.register_dialect(
        'myDialect', quoting=csv.QUOTE_ALL, skipinitialspace=True)

    with open('./data/Project_10/dataset/event_datafile_new.csv', 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, dialect='myDialect')
        writer.writerow(['artist', 'firstName', 'gender', 'itemInSession', 'lastName', 'length',
                         'level', 'location', 'sessionId', 'song', 'userId'])
        for row in full_data_row_list:
            if (row[0] == ''):
                continue
            writer.writerow((row[0], row[2], row[3], row[4], row[5],
                            row[6], row[7], row[8], row[12], row[13], row[16]))


full_data_row_list = get_all_event_data(file_path_list)
create_event_datafile_new(full_data_row_list)


code_preprocessing = """
# Get event data path
filepath = os.getcwd() + '/event_data'

# Create a list of files and collect all filepath
file_path_list = glob.glob(os.path.join(filepath,'*'))

def get_all_event_data(file_path_list):
    full_data_rows_list = []

    # for every filepath in the file path list
    for f in file_path_list:

        # reading csv file
        with open(f, 'r', encoding = 'utf8', newline='') as csvfile:
            # creating a csv reader object
            csvreader = csv.reader(csvfile)
            next(csvreader)

            # extracting each data row one by one and append it
            for line in csvreader:
                full_data_rows_list.append(line)

    return full_data_rows_list

def create_event_datafile_new(full_data_row_list):
    # creating a smaller event data csv file called event_datafile_new.csv
    # that will be used to insert data into the Apache Cassandra tables
    csv.register_dialect(
        'myDialect', quoting=csv.QUOTE_ALL, skipinitialspace=True)

    with open('event_datafile_new.csv', 'w', encoding = 'utf8', newline='') as f:
        writer = csv.writer(f, dialect='myDialect')
        writer.writerow(['artist','firstName','gender','itemInSession','lastName','length',\\
                    'level','location','sessionId','song','userId'])
        for row in full_data_row_list:
            if (row[0] == ''):
                continue
            writer.writerow((row[0], row[2], row[3], row[4], row[5],
                            row[6], row[7], row[8], row[12], row[13], row[16]))

full_data_row_list=get_all_event_data(file_path_list)
create_event_datafile_new(full_data_row_list)

# check the number of rows in your csv file
with open('event_datafile_new.csv', 'r', encoding = 'utf8') as f:
    print(sum(1 for line in f))
"""
st.write("#### 1. ETL Pipeline for Pre-Processing the Files")
st.code(code_preprocessing, language="python")
st.write("The dataset contains the following columns:")
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
df = pd.read_csv(
    './data/Project_10/dataset/event_datafile_new.csv', encoding='utf8')
df.info(verbose=True)
sys.stdout = old_stdout
st.text(mystdout.getvalue())
if "Disconnect" not in st.session_state:
    st.session_state.Disconnect = False
st.markdown("""
### II. Set Up Apache Cassandra Cluster

- <b>1. Creating a Cluster</b>
```python
cluster = Cluster()
session = cluster.connect()
```
""", unsafe_allow_html=True)

# Create a Cluster
cluster = Cluster(['127.0.0.1'])
session = cluster.connect()
st.session_state.Disconnect = False
if "Disconnect" in st.session_state and st.session_state.Disconnect == False:
    st.success("Connected to Apache Cassandra Cluster.")
st.markdown("""
- <b>2. Create Keyspace</b>
```python
try:
    session.execute(\"""
            CREATE KEYSPACE IF NOT EXISTS Minh_Nguyen
            WITH REPLICATION={'class': 'SimpleStrategy', 'replication_factor': 1}\""")
except Exception as e:
    print(e)
```

- <b>3. Set Keyspace</b>
```python
try:
    session.set_keyspace('minh_nguyen')
except Exception as e:
    print(e)
```

- <details> <summary><b>4. Define Queries to Create Tables</b></summary>
    1. Retrieve the artist, song title, and song's length for the music app history entry with sessionId 338 and itemInSession 4 <br>
    2. Provide the artist name, song title (sorted by itemInSession), and the user's first and last name for the user with userid 10 during sessionid 182<br>
    3. List all users (first and last names) who have listened to the song 'All Hands Against His Own' in the music app history
</details>

<b>
5.Create Tables in Apache Cassandra based on the queries provided<br>
</b> <b>
6.Insert Data into Apache Cassandra Tables based on the queries provided<br>
</b> <b>
7.Looping step defined above to process all the files needed</b>

##### Note: The Apache Cassandra database is a NoSQL database, which means that the data is modeled based on the queries you want to run. The queries are provided in the project template, and you will need to model your data based on these queries.
""", unsafe_allow_html=True)

# Create a Cluster
# cluster = Cluster(['127.0.0.1'])
# cluster = Cluster()
# session = cluster.connect()

# Create Keyspace
try:
    session.execute("""
    CREATE KEYSPACE IF NOT EXISTS Minh_Nguyen
    WITH REPLICATION =
    { 'class' : 'SimpleStrategy', 'replication_factor' : 1 }"""
                    )

except Exception as e:
    print(e)

# Set Keyspace
try:
    session.set_keyspace('minh_nguyen')
except Exception as e:
    print(e)
############################################################################################################################################################################
st.markdown("---", unsafe_allow_html=True)
if "Step_1_Query_1" not in st.session_state or "Step_2_Query_1" not in st.session_state or "Step_3_Query_1" not in st.session_state:
    st.session_state.Step_1_Query_1 = False
    st.session_state.Step_2_Query_1 = False
    st.session_state.Step_3_Query_1 = False
st.markdown("### III. Query 1")
st.markdown("""
1. Retrieve the artist, song title, and song's length for the music app history entry with sessionId 338 and itemInSession 4

#### Data Modeling
1. The combination of sessionId and itemInSession is unique, and we need results based on sessionId and itmeInSession, so these two columns could be the Composite Primary Key of the table.
2. Since itemInSession could be skewed to some specific value, it's better to use sessionId as Priamary key so the data could be evenly distributed on the nodes
3. We need the artist name, song title and song length based on the sessionId and itemInSession
Based on the above requirements, we could design the data model as follows:

```SQL
Create table sessions
    (
        sessionId int,
        itemInSession int,
        artist_name text,
        song text,
        song_length float,
        PRIMARY KEY (sessionId, itemInSession)
    ) WITH CLUSTERING ORDER BY (itemInSession DESC)
```

##### III.1. Create Table Session
```python
query = "CREATE TABLE IF NOT EXISTS sessions"
query = query + \
    "(sessionId int, itemInSession int, artist_name text, song text, song_length float, PRIMARY KEY (sessionId, itemInSession))"
try:
    session.execute(query)
except Exception as e:
    print(e)
```
""")
step_1_query_1_btn = st.button("Run Step 1 Query 1")
if step_1_query_1_btn:
    st.session_state.Step_1_Query_1 = True
    st.session_state.Step_2_Query_1 = False
    st.session_state.Step_3_Query_1 = False
    # Create Table Session
    query = "CREATE TABLE IF NOT EXISTS sessions"
    query = query + \
        "(sessionId int, itemInSession int, artist_name text, song text, song_length float, PRIMARY KEY (sessionId, itemInSession))"
    try:
        session.execute(query)
    except Exception as e:
        print(e)
if "Step_1_Query_1" in st.session_state and st.session_state.Step_1_Query_1:
    st.write("Table sessions for Query 1 created successfully.")

st.markdown("""
##### III.2. Insert Data into Table Session
```python
file = 'event_datafile_new.csv'
with open(file, encoding = 'utf8') as f:
    csvreader = csv.reader(f)
    next(csvreader)
    for line in csvreader:

        query = "INSERT INTO sessions(sessionId, itemInSession, artist_name, song, song_length)"
        query = query + " VALUES (%s, %s, %s, %s, %s)"

        session.execute(query, (int(line[8]), int(
            line[3]), line[0], line[9], float(line[5])))
```
""")
step_2_query_1_btn = st.button(
    "Run Step 2 Query 1",  disabled=not st.session_state.Step_1_Query_1)
if step_2_query_1_btn:
    st.session_state.Step_1_Query_1 = True
    st.session_state.Step_2_Query_1 = True
    st.session_state.Step_3_Query_1 = False
    # Insert Data into Table Session
    file = './data/Project_10/dataset/event_datafile_new.csv'
    with open(file, encoding='utf8') as f:
        csvreader = csv.reader(f)
        next(csvreader)
        for line in csvreader:

            query = "INSERT INTO sessions(sessionId, itemInSession, artist_name, song, song_length)"
            query = query + " VALUES (%s, %s, %s, %s, %s)"

            session.execute(query, (int(line[8]), int(
                line[3]), line[0], line[9], float(line[5])))
if "Step_2_Query_1" in st.session_state and st.session_state.Step_2_Query_1:
    st.write("Data inserted into table sessions successfully.")
st.markdown("""  
##### III.3. Query Data from Table Session
```python
query = "select artist_name, song, song_length from sessions WHERE sessionId = 338 and itemInSession = 4"

try:
    rows = session.execute(query)
except Exception as e:
    print(e)
    
for row in rows:
    print (row)
```                  
""")
step_3_query_1_btn = st.button(
    "Run Step 3 Query 1",  disabled=not st.session_state.Step_2_Query_1)
if step_3_query_1_btn:
    st.session_state.Step_1_Query_1 = True
    st.session_state.Step_2_Query_1 = True
    st.session_state.Step_3_Query_1 = True
    query = """
    SELECT artist_name, song, song_length 
    FROM sessions 
    WHERE sessionId = 338 AND itemInSession = 4
    """
    # try:
    rows = session.execute(query)
    print("Query executed successfully.")
    num_artist = 0
    for row in rows:
        num_artist += 1
        st.write("Artist Np: ", num_artist)
        st.write("Artist Name: ", row.artist_name)
        st.write("Song Title: ", row.song)
        st.write("Song Length: ", row.song_length)

############################################################################################################################################################################
st.markdown("---", unsafe_allow_html=True)
if "Step_1_Query_2" not in st.session_state or "Step_2_Query_2" not in st.session_state or "Step_3_Query_2" not in st.session_state:
    st.session_state.Step_1_Query_2 = False
    st.session_state.Step_2_Query_2 = False
    st.session_state.Step_3_Query_2 = False
st.markdown("### IV. Query 2")
st.markdown("""
2. Provide the artist name, song title (sorted by itemInSession), and the user's first and last name for the user with userid 10 during sessionid 182

#### Data Modeling
1. Since we need results based on userid and sessionid, it could make good perfomance to use userid and sessionid as partition key 
to make sure the sessions belonging to the same user could be distributed to same node
2. Since the data needs to be sorted by itemInSession, the itemInSession should be the clustering key
3. Besides Primary Key, we need artist name, song title and user name for this query
Based on the above requirements, we could design the data model as follows:

```
Create table song_play_list
    (
        userid int,
        sessionid int,
        iteminsession int,
        firstname text,
        lastname text,      
        artist_name text,
        song text,     
        PRIMARY KEY((userid, sessionid), iteminsession)
     ) WITH CLUSTERING ORDER BY (iteminsession DESC)

```

##### IV.1. Create Table Session (song_playlist_session)
```python
query = "CREATE TABLE IF NOT EXISTS song_playlist_session"
query = query + "(userid int, sessionid int, iteminsession int, firstname text, lastname text,  artist_name text, song text,  \
PRIMARY KEY((userid, sessionid), iteminsession)) WITH CLUSTERING ORDER BY (iteminsession DESC);" 
try:
    session.execute(query)
except Exception as e:
    print(e)
```
""")
step_1_query_2_btn = st.button("Run Step 1 Query 2")
if step_1_query_2_btn:
    st.session_state.Step_1_Query_2 = True
    st.session_state.Step_2_Query_2 = False
    st.session_state.Step_3_Query_2 = False
    # Create Table Session
    query = "CREATE TABLE IF NOT EXISTS song_playlist_session"
    query = query + "(userid int, sessionid int, iteminsession int, firstname text, lastname text,  artist_name text, song text,  \
    PRIMARY KEY((userid, sessionid), iteminsession)) WITH CLUSTERING ORDER BY (iteminsession DESC);"
    try:
        session.execute(query)
    except Exception as e:
        print(e)
if "Step_1_Query_2" in st.session_state and st.session_state.Step_1_Query_2:
    st.write("Table sessions for Query 2 created successfully.")

st.markdown("""
##### IV.2. Insert Data into Table Session (song_playlist_session)
```python
file = 'event_datafile_new.csv'
with open(file, encoding = 'utf8') as f:
    csvreader = csv.reader(f)
    next(csvreader) # skip header
    
    for line in csvreader:
        
        query = "INSERT INTO song_playlist_session(userid, sessionid, iteminsession, firstname, lastname,  artist_name, song)"
        query = query + " VALUES (%s, %s, %s, %s, %s, %s, %s)"
     
        session.execute(query, (int(line[10]), int(line[8]), int(line[3]), line[1], line[4], line[0], line[9]))
```
""")
step_2_query_2_btn = st.button(
    "Run Step 2 Query 2",  disabled=not st.session_state.Step_1_Query_2)
if step_2_query_2_btn:
    st.session_state.Step_1_Query_2 = True
    st.session_state.Step_2_Query_2 = True
    st.session_state.Step_3_Query_2 = False
    # Insert Data into Table Session
    file = './data/Project_10/dataset/event_datafile_new.csv'
    with open(file, encoding='utf8') as f:
        csvreader = csv.reader(f)
        next(csvreader)  # skip header
        for line in csvreader:
            query = "INSERT INTO song_playlist_session(userid, sessionid, iteminsession, firstname, lastname,  artist_name, song)"
            query = query + " VALUES (%s, %s, %s, %s, %s, %s, %s)"
            session.execute(query, (int(line[10]), int(line[8]), int(
                line[3]), line[1], line[4], line[0], line[9]))
if "Step_2_Query_2" in st.session_state and st.session_state.Step_2_Query_2:
    st.write("Data inserted into table sessions song_playlist_session successfully.")
st.markdown("""  
##### IV.3. Query Data from Table Session (song_playlist_session)
```python
query = "select artist_name, song, firstname,lastname, iteminsession from song_playlist_session where userid=10 and sessionid=182"

try:
    rows = session.execute(query)
except Exception as e:
    print(e)
    
for row in rows:
    #print ( row.artist_name,row.song,row.firstname, row.lastname, row.iteminsession)
    print(row)
```                  
""")
step_3_query_2_btn = st.button(
    "Run Step 3 Query 2",  disabled=not st.session_state.Step_2_Query_2)
if step_3_query_2_btn:
    st.session_state.Step_1_Query_2 = True
    st.session_state.Step_2_Query_2 = True
    st.session_state.Step_3_Query_2 = True
    query = """
    SELECT artist_name, song, firstname, lastname, iteminsession 
    FROM song_playlist_session 
    WHERE sessionid = 182 AND userid=10
    """
    # try:
    rows = session.execute(query)
    print("Query executed successfully.")
    num_artist = 0
    for row in rows:
        num_artist += 1
        st.write("Artist Np: ", num_artist)
        st.write("Artist Name: ", row.artist_name)
        st.write("First Name: ", row.firstname)
        st.write("Last Name: ", row.lastname)

############################################################################################################################################################################
st.markdown("---", unsafe_allow_html=True)
if "Step_1_Query_3" not in st.session_state or "Step_2_Query_3" not in st.session_state or "Step_3_Query_3" not in st.session_state:
    st.session_state.Step_1_Query_3 = False
    st.session_state.Step_2_Query_3 = False
    st.session_state.Step_3_Query_3 = False
st.markdown("### V. Query 3")
st.markdown("""
3. List all users (first and last names) who have listened to the song 'All Hands Against His Own' in the music app history

#### Data Modeling
1. Since we need the results based on song name, firstName and lastName might be not unique, we need to use song name and userid as Primary Key
2. Use the song name as partition key so the users data listened to the same song could be distributed on the same node, it will improve the query performance
3. we need firatName and lastName as well for this query
Based on the above requirements, we could design the data model as follows:

```
Create table users_playlist 
    (
        song text,
        user_id,
        firstname text,
        lastname text,  
        PRIMARY KEY (song, user_id)  
    ) 
```

##### V.1. Create Table Session (users_playlist)
```python
query = "CREATE TABLE IF NOT EXISTS users_playlist"
query = query + "(song text,  userid int, firstname text, lastname text,  PRIMARY KEY (song, userid)) WITH CLUSTERING ORDER BY (userid DESC);" 
try:
    session.execute(query)
except Exception as e:
    print(e)
```
""")
step_1_query_3_btn = st.button("Run Step 1 Query 3")
if step_1_query_3_btn:
    st.session_state.Step_1_Query_3 = True
    st.session_state.Step_2_Query_3 = False
    st.session_state.Step_3_Query_3 = False
    # Create Table Session
    query = "CREATE TABLE IF NOT EXISTS users_playlist"
    query = query + \
        "(song text,  userid int, firstname text, lastname text,  PRIMARY KEY (song, userid)) WITH CLUSTERING ORDER BY (userid DESC);"
    try:
        session.execute(query)
    except Exception as e:
        print(e)
if "Step_1_Query_3" in st.session_state and st.session_state.Step_1_Query_3:
    st.write("Table sessions for Query 3 created successfully.")

st.markdown("""
##### V.2. Insert Data into Table Session (users_playlist)
```python
file = 'event_datafile_new.csv'
with open(file, encoding = 'utf8') as f:
    csvreader = csv.reader(f)
    next(csvreader)
    for line in csvreader:
        query = "INSERT INTO users_playlist(song, userid, firstname, lastname)"
        query = query + " VALUES (%s, %s, %s, %s)"
        session.execute(query, (line[9], int(line[10]), line[1], line[4]))
```
""")
step_2_query_3_btn = st.button(
    "Run Step 2 Query 3",  disabled=not st.session_state.Step_1_Query_3)
if step_2_query_3_btn:
    st.session_state.Step_1_Query_3 = True
    st.session_state.Step_2_Query_3 = True
    st.session_state.Step_3_Query_3 = False
    # Insert Data into Table Session
    file = './data/Project_10/dataset/event_datafile_new.csv'
    with open(file, encoding='utf8') as f:
        csvreader = csv.reader(f)
        next(csvreader)
        for line in csvreader:
            query = "INSERT INTO users_playlist(song, userid, firstname, lastname)"
            query = query + " VALUES (%s, %s, %s, %s)"
            session.execute(query, (line[9], int(line[10]), line[1], line[4]))
if "Step_2_Query_3" in st.session_state and st.session_state.Step_2_Query_3:
    st.write("Data inserted into table sessions users_playlist successfully.")
st.markdown("""  
##### V.3. Query Data from Table Session (users_playlist)
```python
query = "select userid, firstname, lastname from users_playlist where song='All Hands Against His Own'"
try:
    rows = session.execute(query)
except Exception as e:
    print(e)
for row in rows:
    print ( row.userid, row.firstname, row.lastname)
```                  
""")
step_3_query_3_btn = st.button(
    "Run Step 3 Query 3",  disabled=not st.session_state.Step_2_Query_3)
if step_3_query_3_btn:
    st.session_state.Step_1_Query_3 = True
    st.session_state.Step_2_Query_3 = True
    st.session_state.Step_3_Query_3 = True
    query = """
    SELECT artist_name, song, firstname, lastname, iteminsession 
    FROM song_playlist_session 
    WHERE sessionid = 182 AND userid=10
    """
    # try:
    rows = session.execute(query)
    print("Query executed successfully.")
    num_artist = 0
    for row in rows:
        num_artist += 1
        st.write("Artist Np: ", num_artist)
        st.write("Artist Name: ", row.artist_name)
        st.write("First Name: ", row.firstname)
        st.write("Last Name: ", row.lastname)

############################################################################################################################################################################
st.markdown("---", unsafe_allow_html=True)
if "Table_1" not in st.session_state or "Table_2" not in st.session_state or "Table_3" not in st.session_state:
    st.session_state.Table_1 = False
    st.session_state.Table_2 = False
    st.session_state.Table_3 = False
st.markdown("### VI. Drop Tables")
st.markdown("""
4. Drop the table before closing out the sessions""")
st.markdown("""
##### VI.1. Drop Table Session (sessions)
```python
query = "drop table sessions"
try:
    rows = session.execute(query)
except Exception as e:
    print(e)
```
""")
drop_table_session_btn = st.button(
    "Drop Table Session",  disabled=st.session_state.Table_1)
if drop_table_session_btn:
    st.session_state.Table_1 = True
    query = "drop table sessions"
    try:
        rows = session.execute(query)
    except Exception as e:
        print(e)
if "Table_1" in st.session_state and st.session_state.Table_1:
    st.write("Drop table sessions successfully.")

st.markdown("""
##### VI.2. Drop Table song_playlist_session
```python
query = "drop table song_playlist_session"
try:
    rows = session.execute(query)
except Exception as e:
    print(e)
```
""")
drop_table_song_playlist_session_btn = st.button(
    "Drop Table Song Playlist Session",  disabled=st.session_state.Table_2)
if drop_table_song_playlist_session_btn:
    st.session_state.Table_2 = True
    query = "drop table song_playlist_session"
    try:
        rows = session.execute(query)
    except Exception as e:
        print(e)
if "Table_2" in st.session_state and st.session_state.Table_2:
    st.write("Drop table song playlist sessions successfully.")

st.markdown("""
##### VI.3. Drop Table users_playlist
```python
query = "drop table users_playlist"
try:
    rows = session.execute(query)
except Exception as e:
    print(e)
```
""")
drop_table_users_playlist_btn = st.button(
    "Drop Table Users Playlist Session",  disabled=st.session_state.Table_3)
if drop_table_users_playlist_btn:
    st.session_state.Table_3 = True
    query = "drop table users_playlist"
    try:
        rows = session.execute(query)
    except Exception as e:
        print(e)
if "Table_3" in st.session_state and st.session_state.Table_3:
    st.write("Drop table users playlist sessions successfully.")


############################################################################################################################################################################
st.markdown("---", unsafe_allow_html=True)
if "Disconnect" not in st.session_state or "Connect" not in st.session_state:
    st.session_state.Disconnect = False
    st.session_state.Connect = False
st.markdown("### VII. Disconnect and Close the Session")
st.markdown("""
5. Closing out the sessions""")
st.markdown("""
##### VII.1. Close the session and cluster connection
```python
session.shutdown()
cluster.shutdown()
```
""")
if "Disconnect" in st.session_state and st.session_state.Disconnect == False:
    closing_session_btn = st.button(
        "Close Connection Session")
    if closing_session_btn:
        st.session_state.Disconnect = True
        session.shutdown()
        cluster.shutdown()

if "Disconnect" in st.session_state and st.session_state.Disconnect:
    st.success("Connection Session Closed.")
