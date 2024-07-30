# Work Plan

**I only got 3 days.**

## Description
This is a prototype built for economicists to help them quantify the impact of news.

- **Topic**: From News to Insights
- **One-liner**: We help economists build quantifiable data source from news using NLP (or LLM / prompt engineering)
- **Pain point**: Seeing an important news != can study the news impact immediately
- **Motivations**: Various research have shown that news could be a source of rich information on predicting crisis and historical events
- **What we do**: Extract / Build quantifiable data source from news using NLP (or LLM / prompt engineering)

In this prototype, the economist would able to first specifiy the region of interest, then a list of pre-retrieved thematic topic would come up and rank according to it's importance (frequency of being mentioned). Also, I plan to devise three graphs to smoothen economists' workflow.

1. Topic-based sentiment visualization (heatmap).
    - X: Topic
    - Y: Emotion / Stance
    - Data: Correlation between two time series data, the current box (topic + emotion / stance) and the economic indicator
2. Topic sentiment trend over time (with historical events)
3. A topic sentiment's predictive power towards historical event's factor analysis visualization.

News data will be retrieved from the GDELT dataset, and the economic indicator will be retrieved from the World Bank dataset.

# Data Preparation

1. Download raw data from somewhere, like GDELT dataset

2. Preprocess the data with `src/preprocess/news_engine.py`

3. Save the preprocessed data to `data/processed/news.csv` with the following columns:
    - title: str
    - content: str
    - date: datetime
    <!-- - source: str -->
    <!-- - Country: str -->
    - thematic_topics: list
    - importance_rating: float (from 0 to 1)
    - sentiment: float (from -1 to 1)
    - emotions: list
    <!-- - Event ID: str -->

4. Download raw data from somewhere, like World Bank dataset

5. Preprocess the data with `src/preprocess/economic_engine.py`

6. Save the preprocessed data to `data/processed/economic.csv` with the following columns:
    - indicator: str
    - date: datetime
    - value: float
    - country: str

# Data Modeling

1. Load the preprocessed data from `data/processed/news.csv` and `data/processed/economic.csv`

2. Build the following graphs for Tableau visualization with `src/report_preparation/`:
    - Topic-based sentiment visualization (heatmap)
        - Steps
            1. Aggregate the data by topic and emotion / stance
            2. Calculate the correlation between the current box (topic + emotion / stance) and the economic indicator
            3. Plot the heatmap
    - Topic sentiment trend over time (with historical events)
        - Steps
            1. Aggregate the data by topic and date
            2. Calculate the sentiment trend over time
            3. Retrieve the historical events
            4. Plot the sentiment trend over time with historical events
    - A topic sentiment's predictive power towards historical event's Vector Autoregression (VAR) and Impulse Response Function (IRF) visualization
        - Steps
            1. Aggregate the data by topic and date
            2. Calculate the sentiment trend over time
            3. Retrieve the historical events
            4. Calculate the Vector Autoregression (VAR) and Impulse Response Function (IRF)
            5. Plot the VAR and IRF visualization

3. Save the graphs to `data/reports/figures/`

# Deployment (Streamlit Frontend + FastAPI Backend)

1. Navigate into the directory `src/` with:
    ```bash
    cd src/
    ```

2. Start FastAPI backend with:
    ```bash
    python main.py
    ```

3. Start Streamlit frontend with:
    ```bash
    streamlit run app.py
    ```