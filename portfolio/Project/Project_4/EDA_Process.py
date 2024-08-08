import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

def exploring_data_analysis_process(df, df_sort_by_time):
    # Distribution of numeric variables
    numeric_vars = ['likes_count', 'views_count', 'dislikes_count', 'tag_count', 'comments_count', 'title_length', 'reactions']
    
    st.subheader("Distribution of Numeric Variables")
    # Histogram of like_count
    fig = px.histogram(df, x="likes_count", title=f'Distribution of Likes Count')
    st.plotly_chart(fig)

    histogram_eda_col1, histogram_eda_col2  = st.columns(2)

    with histogram_eda_col1:
        for var in numeric_vars[1:4]:
            fig = px.histogram(df, x=var, title=f'Distribution of {var}')
            st.plotly_chart(fig)
    
    with histogram_eda_col2:
        for var in numeric_vars[4:]:
            fig = px.histogram(df, x=var, title=f'Distribution of {var}')
            st.plotly_chart(fig)
    st.write("---")

    # Checking Null Items in dataset
    st.subheader("Checking Null Items in Dataset")
    null_counts = df.isnull().sum().reset_index()
    null_counts.columns = ['Column', 'Null Count']
    fig_null = px.bar(null_counts, x='Column', y='Null Count', title='Null Counts in Dataset')
    st.plotly_chart(fig_null)
    st.write("---")

    # Boxplot of numeric variables to check for anomalies or outliers
    st.subheader("Boxplot of Numeric Variables")
    Box_plot_col1, Box_plot_col2 = st.columns(2)
    Box_plot_group_1 = ["likes_count", "reactions"]
    Box_plot_group_2 = ["tag_count", "title_length", "comments_count"]
    # Boxplot of views_count
    with Box_plot_col1:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=df["views_count"], name="views_count"))
        fig_box.update_layout(title='Boxplot of View Count')
        st.plotly_chart(fig_box)

        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=df["comments_count"], name="comments_count"))
        fig_box.update_layout(title='Boxplot of Comments Count')
        st.plotly_chart(fig_box)

    with Box_plot_col2:
        fig_box = go.Figure()
        for var in numeric_vars:
            if var in Box_plot_group_1:
                fig_box.add_trace(go.Box(y=df[var], name=var))
        fig_box.update_layout(title='Boxplot of Numeric Variables')
        st.plotly_chart(fig_box)
    
        fig_box = go.Figure()
        for var in numeric_vars:
            if var in Box_plot_group_2:
                fig_box.add_trace(go.Box(y=df[var], name=var))
        fig_box.update_layout(title='Boxplot of Numeric Variables')
        st.plotly_chart(fig_box)
    st.write("---")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    correlation_matrix = df[numeric_vars].corr()
    fig_heatmap = px.imshow(correlation_matrix, text_auto=True, aspect="auto", title='Correlation Heatmap', height = 800)
    st.plotly_chart(fig_heatmap)
    
    st.write("---")
    # Relationship between views_count and other numeric variables
    relationship_group_1 = ["likes_count","comments_count", "reactions"]
    relationship_group_2 = ["dislikes_count", "tag_count", "title_length"]
    st.subheader("Relationship between Views Count and Other Numeric Variables")
    relationship_col1, relationship_col2 = st.columns(2)
    with relationship_col1:
        for var in numeric_vars:
            if var in relationship_group_1:
                fig_scatter = px.scatter(df, x=var, y='views_count', title=f'Relationship between {var} and Views Count', trendline='ols')
                st.plotly_chart(fig_scatter)
    
    with relationship_col2:
        for var in numeric_vars:
            if var in relationship_group_2:
                fig_scatter = px.scatter(df, x=var, y='views_count', title=f'Relationship between {var} and Views Count', trendline='ols')
                st.plotly_chart(fig_scatter)
    st.write("---")
    summarize_view_trend_analysis(df_sort_by_time)

def summarize_view_trend_analysis(df_sort_by_time):
    """
    The summarize_view_trend_analysis function generates visual summaries of the trend analysis for total views over different time periods such as years, months, periods within a day, and hours. It groups the provided DataFrame by various time-related columns and calculates the total views for each group. It then creates plots to visualize these trends using Plotly and Streamlit.

    Parameters:
        df_sort_by_time (pandas.DataFrame, optional): The DataFrame containing sorted data based on the 'published' column. Defaults to the value of df_sort_by_time.

    Visualization Details:
        - Trend of Total Views Over Year: This line plot illustrates the trend of total views over years.
        - Trend of Total Views Over Months: This line plot illustrates the trend of total views over months.
        - Total Views by Period in Day: This bar plot displays the total views categorized by periods within a day.
        - Total Views by Hour: This line plot shows the total views across different hours of the day.
    """
    # Group by publish_year and calculate total views
    st.subheader("View Trend Analysis")

    # Group by publish_period and calculate total views
    df_period_in_day_total_views = df_sort_by_time.groupby('publish_period')['views_count'].sum().reset_index()
    fig_period = px.bar(df_period_in_day_total_views, x='publish_period', y='views_count', title='Total Views by Period in Day')
    st.plotly_chart(fig_period)

    view_trend_col1, view_trend_col2 = st.columns(2)
    with view_trend_col1:
        # Group by publish_year and calculate total views
        df_year_total_views = df_sort_by_time.groupby('publish_year')['views_count'].sum().reset_index()
        fig_year = px.line(df_year_total_views, x='publish_year', y='views_count', title='Trend of Total Views Over Year')
        st.plotly_chart(fig_year)

        # Group by publish_month and calculate total views
        df_month_total_views = df_sort_by_time.groupby(['publish_month', 'publish_year'])['views_count'].sum().reset_index()
        fig_month = px.line(df_month_total_views, x='publish_month', y='views_count', color='publish_year', title='Trend of Total Views Over Months', labels={'views_count': 'Total Views'})
        st.plotly_chart(fig_month)

    with view_trend_col2:
        # Group by publish_month and calculate total video count
        df_month_total_video = df_sort_by_time.groupby(['publish_month', 'publish_year'])['video_id'].count().reset_index()
        fig_video = px.line(df_month_total_video, x='publish_month', y='video_id', color='publish_year', title='Trend of Total Video Upload Over Months', labels={'video_id': 'Total Videos'})
        st.plotly_chart(fig_video)

        # Group by publish_hour and calculate total views
        df_hour_in_day_total_views = df_sort_by_time.copy()
        # Create new columns for publish_hour and publish_day_in_week
        df_hour_in_day_total_views['publish_day_in_week'] = df_hour_in_day_total_views['published'].dt.day_name()
        df_hour_in_day_total_views = df_hour_in_day_total_views.groupby(['publish_hour', 'publish_day_in_week'])['views_count'].sum().reset_index()
        fig_hour = px.line(df_hour_in_day_total_views, x='publish_hour', y='views_count', color='publish_day_in_week', title='Total Views by Hour and Day in Week', labels={'views_count': 'Total Views', 'publish_hour': 'Hour', 'publish_day_in_week': 'Day in Week'})
        st.plotly_chart(fig_hour)
    
    #st.write(df_sort_by_time['publish_day_in_week'].head(10))
    st.write("---")
    