# src/main.py
# The main is a FastAPI server

# Use streamlit to build a webpage. The webpage is an analytics dashboard that displays the results of the analysis performed by the report_engine.py and news_engine.py scripts. The dashboard should include the following components:.

# ## Layout:
# - Website Title: "NewsInsights"
# - 1 page layout
# - The screen should be divided into 2 rows
#     - The first row should be further divided into 2 columns
#         - The first column display one search bar for searching news articles, with a submit button, and a dropdown for selecting the region. Below the search bar is the thematic topics's frequency plot mentioned in the news articles in the form of a bar chart in horizontal orientation, descending.
#         - The second column should display the sentiment analysis results in the form of a heatmap.
#     - The second row should display the news articles with the highest importance rating, in the form of a table.
# - The dashboard should be interactive, allowing users to search for news articles and select regions to display thematic topics and sentiment analysis results.
# - Upon clicking the submit button, the dashboard should display the thematic topics's frequency plot and sentiment analysis results for the news articles related to the search query and region selected.

# ## API:
# 1. - Search for news articles
#     - GET /search
#         - q (contains keywords): str
#         - r (region): str
#     - Returns the news articles related to the search query and region
#         - A JSON containing:
#             - news articles: List[Dict[str, str]]
#             - thematic topics: List[str]
#             - sentiment analysis: Dict[str, float]
#             - news importance rating: Dict[str, float]
#         - Example response:
#             ```json
#             {
#                 "news_articles": [
#                     {
#                         "title": "News Article 1",
#                         "content": "This is the content of News Article 1"
#                     },
#                     {
#                         "title": "News Article 2",
#                         "content": "This is the content of News Article 2"
#                     }
#                 ],
#                 "thematic_topics": ["topic1", "topic2", "topic3"],
#                 "emotion_analysis": {
#                     "joy": [3, 2, 1],
#                     "anger": [1, 0, 2],
#                     "trust": [1, 1, 0]
#                 },
#                 "news_importance_rating": {
#                     "News Article 1": [0.8],
#                     "News Article 2": [0.6]
#                 }
#             }
#             ```

import fastapi
from fastapi import FastAPI, Query
from typing import List, Dict
import pandas as pd
from ast import literal_eval

app = FastAPI()

# Function to read and process the CSV file
def read_news_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, converters={"thematic_topics": literal_eval, "emotions": literal_eval})

# Load the news data
news_data = read_news_data('../data/processed/news.csv')

@app.get("/search")
# def search_news(q: str = Query(..., description="Keywords to search for"), r: str = Query(..., description="Region to filter by")):
def search_news(q: str = Query(..., description="Keywords to search for"), r: str = Query(..., description="Region to filter by")):
    # Filter news articles based on the search query and region
    # filtered_news = news_data[
    #     (news_data['content'].str.contains(q, case=False)) & 
    #     (news_data['country'].str.contains(r, case=False))
    # ]
    filtered_news = news_data[(news_data['content'].str.contains(q, case=False))]
    
    # Prepare the response
    news_articles = filtered_news.to_dict(orient='records')
    thematic_topics = filtered_news['thematic_topics'].explode().value_counts().to_dict()
    sentiment_analysis = filtered_news['emotions'].explode().value_counts().to_dict()
    news_importance_rating = filtered_news.set_index('title')['importance_rating'].to_dict()

    response = {
        "news_articles": news_articles,
        "thematic_topics": thematic_topics,
        "sentiment_analysis": sentiment_analysis,
        "news_importance_rating": news_importance_rating
    }
    
    return response

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
