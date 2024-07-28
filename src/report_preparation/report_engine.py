import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# TODO: Implement the ReportEngine
class ReportEngine:
    def __init__(self):
        self.news_data = pd.read_csv('../../data/processed/news.csv')
        self.economic_data = pd.read_csv('../../data/processed/economic_indicators.csv')
        self.historical_events = pd.read_csv('../../data/processed/historical_events.csv')
        self.report_path = '../../data/reports/figures/'

    def build_heatmap(self):
        # Prepare the data for heatmap visualization in Tableau
        # Aggregate the data by topic and emotion / stance
        # Calculate the correlation between the current box (topic + emotion / stance) and the economic indicator
        # Plot the heatmap
        # Aggregate data by topic and emotion/stance

        # Aggregate data by topic and emotion/stance
        numeric_columns = self.news_data.select_dtypes(include='number').columns
        aggregated_data = self.news_data.groupby(['thematic_topics', 'emotions'])[numeric_columns].mean().reset_index()
        
        # Exclude non-numeric columns before calculating the correlation matrix
        numeric_aggregated_data = aggregated_data[numeric_columns]

        # Calculate correlation between topic-emotion pairs and economic indicators
        correlation_matrix = numeric_aggregated_data.corr()
        
        # Plot the heatmap using matplotlib
        plt.figure(figsize=(10, 8))
        plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
        plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
        plt.title('Topic-based Sentiment Heatmap')
        plt.savefig(self.report_path + 'heatmap.png')
        plt.show()

    def build_trend(self):
        # Prepare the data for trend visualization in Tableau
        # Aggregate the data by topic and date
        # Calculate the sentiment trend over time
        # Retrieve the historical events
        # Plot the sentiment trend over time with historical events

        # Aggregate data by topic and date
        numeric_columns = self.news_data.select_dtypes(include='number').columns
        trend_data = self.news_data.groupby(['thematic_topics', 'date'])[numeric_columns].mean().reset_index()
        
        # Calculate sentiment trend over time
        sentiment_trend = trend_data.pivot(index='date', columns='thematic_topics', values='sentiment')
        
        # Retrieve historical events (mocked for now)
        # Used self.historical_events
        
        # Plot sentiment trend over time with historical events
        plt.figure(figsize=(12, 6))
        for column in sentiment_trend.columns:
            plt.plot(sentiment_trend.index, sentiment_trend[column], label=column)
        
        for _, row in self.historical_events.iterrows():
            plt.axvline(x=row['date'], color='r', linestyle='--')
            plt.text(row['date'], plt.ylim()[1], row['event'], rotation=90, verticalalignment='bottom')
        
        plt.title('Topic Sentiment Trend Over Time')
        plt.legend()
        plt.savefig(self.report_path + 'trend.png')
        plt.show()

    def build_var_irf(self, n=3):
        # Prepare the data for VAR and IRF visualization in Tableau
        # Aggregate the data by topic and date
        # Calculate the sentiment trend over time
        # Calculate the Vector Autoregression (VAR) and Impulse Response Function (IRF)
        # Plot the VAR and IRF visualization

        # Aggregate data by topic and date
        numeric_columns = self.news_data.select_dtypes(include='number').columns
        var_data = self.news_data.groupby(['thematic_topics', 'date'])[numeric_columns].mean().reset_index()
        
        # Calculate sentiment trend over time
        sentiment_trend = var_data.pivot(index='date', columns='thematic_topics', values='sentiment')

        # Calculate the mean sentiment for each topic
        mean_sentiment = sentiment_trend.mean().sort_values(ascending=False)

        # Select the top n topics
        n = min(n, len(mean_sentiment))
        top_n_topics = mean_sentiment.head(n).index

        # Filter the sentiment_trend DataFrame to include only the top n topics
        sentiment_trend_top_n = sentiment_trend[top_n_topics].ffill().bfill()

        # Ensure there are no NaN or inf values
        if np.isnan(sentiment_trend_top_n.values).any():
            raise ValueError("Data contains NaN values. Please check the data source.")
        if np.isinf(sentiment_trend_top_n.values).any():
            raise ValueError("Data contains infinity values. Please check the data source.")

        # Standardize the data to handle outliers and scale issues
        scaler = StandardScaler()
        sentiment_trend_top_n = pd.DataFrame(scaler.fit_transform(sentiment_trend_top_n), 
                                            index=sentiment_trend_top_n.index, 
                                            columns=sentiment_trend_top_n.columns)

        # Check for multicollinearity using VIF
        vif_data = pd.DataFrame()
        vif_data["feature"] = sentiment_trend_top_n.columns
        vif_data["VIF"] = [variance_inflation_factor(sentiment_trend_top_n.values, i) for i in range(sentiment_trend_top_n.shape[1])]
        
        if vif_data["VIF"].max() > 10:
            raise ValueError("High multicollinearity detected. Consider removing highly correlated variables.")

        # Check the size of the dataset
        num_observations = sentiment_trend_top_n.shape[0]
        num_variables = sentiment_trend_top_n.shape[1]

        # Ensure there are enough observations
        if num_observations <= num_variables:
            raise ValueError("Not enough observations for the number of variables. Consider reducing the number of variables or increasing the dataset size.")

        # Calculate Vector Autoregression (VAR)
        model = VAR(sentiment_trend_top_n)
        try:
            results = model.fit(maxlags=n+1, ic='aic')
        except np.linalg.LinAlgError as e:
            raise ValueError("Matrix is not positive definite. Please check the data and ensure it is well-conditioned.") from e
        
        # Calculate Impulse Response Function (IRF)
        irf = results.irf(10)
        
        # Plot VAR and IRF visualization
        irf.plot(orth=False)
        plt.title('Impulse Response Function (IRF)')
        plt.savefig(self.report_path + 'irf.png')
        plt.show()

    def run_pipeline(self):
        self.build_heatmap()
        self.build_trend()
        self.build_var_irf()


if __name__ == '__main__':
    report_engine = ReportEngine()
    report_engine.run_pipeline()
