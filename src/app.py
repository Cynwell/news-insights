# src/app.py
import streamlit as st
import requests
import pandas as pd
import seaborn as sns
# import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from report_preparation.report_engine import ReportEngine

# Set up the Streamlit app
st.set_page_config(page_title="NewsInsights", layout="wide")
st.title("NewsInsights")

# Create layout

with st.container():
    # Search bar and region dropdown
    search_query = st.text_input("Search for news articles")
    region = st.selectbox("Select region", ["All", "United States", "China"])
    submit_button = st.button("Submit")

with st.container():
    section_height = 700
    top_n_topics = 5
    # First row with two columns
    col1, col2 = st.columns(2)

    with col1:

        if submit_button:
            # Fetch data from FastAPI server
            params = {"q": search_query, "r": region}
            response = requests.get("http://localhost:8000/search", params=params)
            data = response.json()

            # Initialize the ReportEngine
            report_engine = ReportEngine(news_data=pd.DataFrame.from_dict(data['news_articles']))

            # Unzip thematic topics
            thematic_topics = data["thematic_topics"]

            # Sort thematic topics by frequency
            thematic_topics = pd.DataFrame.from_dict(thematic_topics, orient='index', columns=['Frequency']).reset_index()
            thematic_topics.columns = ['Topic', 'Frequency']
            thematic_topics = thematic_topics.sort_values(by=['Frequency', 'Topic'], ascending=[False, True]).reset_index(drop=True)[:top_n_topics]
            print(thematic_topics)

            # Display thematic topics frequency plot with horizontal bars using Plotly
            fig = px.bar(thematic_topics, x='Frequency', y='Topic', orientation='h', title='Thematic Topics Frequency', height=section_height)
            fig.update_traces(text=thematic_topics['Frequency'], textposition='outside')
            fig.update_layout(yaxis={'autorange':'reversed'})
            st.plotly_chart(fig,)

    with col2:
        if submit_button:
            heatmap_data = report_engine.build_heatmap()
            # print('Heatmap:\n', heatmap_data)

            # Reorder heatmap_data based on the sorted thematic topics
            sorted_topics = thematic_topics['Topic'].tolist()
            # print('Sorted Topics:\n', sorted_topics)

            # Filter heatmap_data to match the length of thematic_topics
            heatmap_data = heatmap_data[heatmap_data['topics'].isin(sorted_topics)]
            heatmap_data = heatmap_data.set_index('topics').loc[sorted_topics].reset_index()


            # Prepare data for Plotly heatmap
            z = heatmap_data.drop(columns='topics').values
            x = heatmap_data.columns[1:]  # Assuming first column is 'topic'
            y = heatmap_data['topics']

            # Create Plotly heatmap
            fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale='Plasma'))
            fig.update_layout(title='Emotion Frequency Heatmap', xaxis_title='Emotions', yaxis_title='Topics', yaxis={'autorange':'reversed'}, height=section_height)
            
            # Display the heatmap
            st.plotly_chart(fig)

# Second row for news articles table
with st.container():
    if submit_button:
        news_articles = pd.DataFrame(data["news_articles"])
        news_importance_rating = pd.Series(data["news_importance_rating"])
        news_articles["importance_rating"] = news_articles["title"].map(news_importance_rating)
        news_articles = news_articles.sort_values(by="importance_rating", ascending=False)
        st.dataframe(news_articles.reset_index(drop=True),
                    hide_index=True,
                    use_container_width=True)
        