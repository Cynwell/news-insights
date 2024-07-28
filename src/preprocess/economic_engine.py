import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# from src.api import AzureOpenAIClient

class EconomicEngine:
    # EconomicEngine is a class that interacts with the Azure OpenAI API to perform the following tasks:
    # 1. Get Economic Indicator Prediction
    # 2. Get Economic Indicator Forecast
    # 3. Get Economic Indicator Analysis
    # And finally writing the results to a new CSV file as a metadata for the economic indicators.

    # TODO: Implement EconomicEngine class by using API from World Bank dataset
    def __init__(self, endpoint, api_key, deployment_name):
        # self.client = AzureOpenAIClient(endpoint, api_key, deployment_name)
        pass

    # # Implement get_economic_indicator_prediction method using prompt engineering
    # def get_economic_indicator_prediction(self, indicator_data):
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": indicator_data
    #         }
    #     ]
    #     response = self.client.get_completion(messages, max_tokens=50)
    #     return response

    # # Implement get_economic_indicator_forecast method using prompt engineering
    # def get_economic_indicator_forecast(self, indicator_data):
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": indicator_data
    #         }
    #     ]
    #     response = self.client.get_completion(messages, max_tokens=50)
    #     return response

    # # Implement get_economic_indicator_analysis method using prompt engineering
    # def get_economic_indicator_analysis(self, indicator_data):
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": indicator_data
    #         }
    #     ]
    #     response = self.client.get_completion(messages, max_tokens=50)
    #     return response

    # TODO: Implement run_pipeline method to run the pipeline
    def run_pipeline(self, df, output_file="data/processed/economic_indicators.csv"):
        predictions = []
        forecasts = []
        analyses = []

        # for indicator_data in df['indicator']:
        #     response = self.get_economic_indicator_prediction(indicator_data)
        #     predictions.append(response)

        # for indicator_data in df['indicator']:
        #     response = self.get_economic_indicator_forecast(indicator_data)
        #     forecasts.append(response)

        # for indicator_data in df['indicator']:
        #     response = self.get_economic_indicator_analysis(indicator_data)
        #     analyses.append(response)

        # df['predictions'] = predictions
        # df['forecasts'] = forecasts
        # df['analyses'] = analyses
        df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Read the raw data from data/raw/economic.csv that has the following columns: indicator, date, value, country
    df = pd.read_csv("data/raw/economic.csv")

    # TODO: Initialize the EconomicEngine with the World Bank API credentials
    economic_engine = EconomicEngine(...)
    economic_engine.run_pipeline(df, output_file="data/processed/economic_indicators.csv")
    # The processed data will have these columns:
    # indicator, date, value, country, predictions, forecasts, analyses
