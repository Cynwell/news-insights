# # Work Plan

# 1. **Thematic Topics**: The NLP model should be able to discover thematic topics related to the search input. The topics should be related to a specific region like global/US/Europe/Asia/EMEAP/Emerging Markets.

#    - Expected Input: Search input (region)
#    - Expected Output: List of thematic topics related to the region
#    - Example: 
#      - Input: "US"
#      - Output: ["US Elections", "US Economy", "US-China Trade War"]

# 2. **News Importance Rating**: The model should rate the news based on its importance using prompt engineering.

#    - Expected Input: News article
#    - Expected Output: Importance rating (1-5)
#    - Example: 
#      - Input: News article about US Elections
#      - Output: 5

# 3. **Sentiment Analysis**: The model should perform sentiment analysis over the news and extract the sentiment of the respondent's view/comments.

#    - Expected Input: News article and respondent's comments
#    - Expected Output: Sentiment (Positive, Neutral, Negative)
#    - Example: 
#      - Input: News article about US Elections and respondent's comments
#      - Output: Negative

# 4. **Emotion Analysis**: The model should identify the emotion expressed by the news and respondents. You can use Plutchik's Wheel of Emotions as a reference for a list of emotions commonly appeared in the news. The emotions include: joy, trust, fear, surprise, sadness, disgust, anger, and anticipation.

#    - Expected Input: News article and respondent's comments
#    - Expected Output: Emotion (e.g., joy, trust, fear, surprise, sadness, disgust, anger, anticipation)
#    - Example: 
#      - Input: News article about US Elections and respondent's comments
#      - Output: Anger

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.api import AzureOpenAIClient

class NewsEngine:
    # NewsEngine is a class that interacts with the Azure OpenAI API to perform the following tasks:
    # 1. Get Thematic Topics
    # 2. Get News Importance Rating
    # 3. Get Sentiment Analysis
    # 4. Get Emotion Analysis
    # And finally writing the results to a new CSV file as a metadata for the news articles.


    def __init__(self, endpoint, api_key, deployment_name):
        self.client = AzureOpenAIClient(endpoint, api_key, deployment_name)

    # TODO: Implement get_thematic_topics method using prompt engineering
    def get_thematic_topics(self, region):
        messages = [
            {
                "role": "user",
                "content": region
            }
        ]
        response = self.client.get_completion(messages, max_tokens=10)
        return response

    # TODO: Implement get_news_importance_rating method using prompt engineering
    def get_news_importance_rating(self, news_article):
        messages = [
            {
                "role": "user",
                "content": news_article
            }
        ]
        response = self.client.get_completion(messages, max_tokens=10)
        return response

    # TODO: Implement get_sentiment_analysis method using prompt engineering
    def get_sentiment_analysis(self, news_article, comments):
        messages = [
            {
                "role": "user",
                "content": news_article
            },
            {
                "role": "user",
                "content": comments
            }
        ]
        response = self.client.get_completion(messages, max_tokens=10)
        return response

    # TODO: Implement get_emotion_analysis method using prompt engineering
    def get_emotion_analysis(self, news_article, comments):
        messages = [
            {
                "role": "user",
                "content": news_article
            },
            {
                "role": "user",
                "content": comments
            }
        ]
        response = self.client.get_completion(messages, max_tokens=10)
        return response

    def run_pipeline(self, df, output_file="data/processed/news.csv"):
        thematic_topics = []
        importance_rating = []
        sentiment = []
        emotions = []

        for content in df['content']:
            response = self.get_thematic_topics(content)
            thematic_topics.append(response)

        for content in df['content']:
            response = self.get_news_importance_rating(content)
            importance_rating.append(response)

        for content in df['content']:
            response = self.get_sentiment_analysis(content, content)
            sentiment.append(response)

        for content in df['content']:
            response = self.get_emotion_analysis(content, content)
            emotions.append(response)

        df['thematic_topics'] = thematic_topics
        df['importance_rating'] = importance_rating
        df['sentiment'] = sentiment
        df['emotions'] = emotions
        df.to_csv("data/processed/news_metadata.csv", index=False)
    

if __name__ == "__main__":
    # Read the raw data from data/raw/news.csv that has the following columns: title, content, date
    from dotenv import load_dotenv
    import os
    import pandas as pd

    load_dotenv()

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPEN_AI_API_KEY")
    deployment_name = os.getenv("DEPLOYMENT_NAME")

    df = pd.read_csv("data/raw/news.csv")

    news_engine = NewsEngine(endpoint, api_key, deployment_name)
    news_engine.run_pipeline(df, output_file="data/processed/news.csv")
    # The processed data will have these columns:
    # title, content, date, thematic_topics, importance_rating, sentiment, emotions
