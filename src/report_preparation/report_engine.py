import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import os

# TODO: Implement the ReportEngine
class ReportEngine:
    def __init__(self, news_data=None, economic_data=None, historical_events=None, data_path=None, report_path=None):
        if news_data is not None or economic_data is not None or historical_events is not None:
            self.data_path = None
            self.report_path = None
            self.news_data = news_data
            self.economic_data = economic_data
            self.historical_events = historical_events
        else:
            self.data_path = data_path
            self.report_path = report_path
            self.news_data = pd.read_csv(os.path.join(self.data_path, 'processed/news.csv'))
            self.economic_data = pd.read_csv(os.path.join(self.data_path, 'processed/economic_indicators.csv'))
            self.historical_events = pd.read_csv(os.path.join(self.data_path, 'processed/historical_events.csv'))

    # TODO: Implement the build_thematic_keywords_barplot method
    def build_thematic_keywords_barplot(self):
        pass

    def build_heatmap(self):
        # Calculate topics frequency
        topic_no_counts = self._calculate_topic_frequency()

        # Calculate each topic's emotion frequency
        final_df = self._calculate_emotion_frequency(topic_no_counts)

        # Sort by highest frequency of topic
        sorted_all = final_df.sort_values(by='count', ascending=False).reset_index(drop=True).fillna(0)

        # Prepare data for heatmap
        sorted_data = sorted_all.drop('count', axis=1)

        # # Plot heatmap
        # self._plot_heatmap(sorted_data)

        # Return data for heatmap
        return sorted_data

    def _calculate_topic_frequency(self):
        topic_no_counts = {}
        for _, row in self.news_data.iterrows():
            topics = set(row['thematic_topics'])
            for topic in topics:
                topic_no_counts[topic] = topic_no_counts.get(topic, 0) + 1
        return topic_no_counts

    def _calculate_emotion_frequency(self, topic_no_counts):
        final_df = pd.DataFrame()
        for topic in topic_no_counts.keys():
            filtered_df = self.news_data[self.news_data['thematic_topics'].apply(lambda x: topic in x)]
            emotion_no_counts = self._count_emotions(filtered_df)
            topic_df = self._create_topic_dataframe(topic, topic_no_counts[topic], emotion_no_counts)
            final_df = pd.concat([final_df, topic_df], axis=0)
        return final_df

    def _count_emotions(self, filtered_df):
        emotion_no_counts = {}
        for _, row in filtered_df.iterrows():
            emotions = set(row['emotions'])
            for emotion in emotions:
                emotion_no_counts[emotion] = emotion_no_counts.get(emotion, 0) + 1
        return emotion_no_counts

    def _create_topic_dataframe(self, topic, count, emotion_no_counts):
        topic_df = pd.DataFrame({'topics': [topic], 'count': [count]})
        emotion_df = pd.DataFrame(emotion_no_counts, index=[0])
        return pd.concat([topic_df, emotion_df], axis=1)

    def _plot_heatmap(self, sorted_data):
        fig, ax = plt.subplots(figsize=(12, 8))
        heatmap = sns.heatmap(sorted_data.iloc[:, 1:], annot=True, cmap='coolwarm', fmt='.1f', linewidths=0.5, ax=ax)
        
        # Set titles and labels
        ax.set_title('Topic-based Sentiment Heatmap')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Topics')
        
        # Set ticks
        ax.set_xticks(np.arange(len(sorted_data.columns[1:])) + 0.5)
        ax.set_xticklabels(sorted_data.columns[1:], ha='center')
        ax.set_yticks(np.arange(len(sorted_data['topics'])) + 0.5)
        ax.set_yticklabels(sorted_data['topics'], rotation=0, ha='right')
        
        # Save and show the plot
        if self.report_path is not None:
            plt.savefig(os.path.join(self.report_path, 'heatmap.png'))
        # plt.show()
        return fig

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
        # self.build_thematic_keywords_barplot()
        self.build_heatmap()
        # self.build_trend()
        # self.build_var_irf()


if __name__ == '__main__':
    report_engine = ReportEngine(data_path='../../data', report_path='../../data/reports')
    # report_engine.run_pipeline()
    fig = report_engine.build_heatmap()
    plt.show()
