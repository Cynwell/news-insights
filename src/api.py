# Connect to Azure OpenAI API and get the response
# I will need to set the system prompt via functions, user input, and the API key

import os
from openai import AzureOpenAI


class AzureOpenAIClient:
    def __init__(self, endpoint, api_key, deployment_name):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-05-01-preview",
            azure_endpoint=endpoint,
        )
        self.deployment = deployment_name

    def get_completion(
        self,
        messages,
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
    ):
        completion = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream,
        )
        return completion.to_json()


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()


    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPEN_AI_API_KEY")
    deployment_name = os.getenv("DEPLOYMENT_NAME")

    client = AzureOpenAIClient(endpoint, api_key, deployment_name)
    messages = [{"role": "user", "content": "Hi"}]
    response = client.get_completion(messages, max_tokens=10)
    print(response)
    """
{
  "id": "chatcmpl-9pbsis5gd7BYM48aDDzLClMvyPai8",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "Hello! How can I assist you today?",
        "role": "assistant"
      },
      "content_filter_results": {
        "hate": {
          "filtered": false,
          "severity": "safe"
        },
        "self_harm": {
          "filtered": false,
          "severity": "safe"
        },
        "sexual": {
          "filtered": false,
          "severity": "safe"
        },
        "violence": {
          "filtered": false,
          "severity": "safe"
        }
      }
    }
  ],
  "created": 1722087372,
  "model": "gpt-4-32k",
  "object": "chat.completion",
  "system_fingerprint": null,
  "usage": {
    "completion_tokens": 9,
    "prompt_tokens": 8,
    "total_tokens": 17
  },
  "prompt_filter_results": [
    {
      "prompt_index": 0,
      "content_filter_results": {
        "hate": {
          "filtered": false,
          "severity": "safe"
        },
        "self_harm": {
          "filtered": false,
          "severity": "safe"
        },
        "sexual": {
          "filtered": false,
          "severity": "safe"
        },
        "violence": {
          "filtered": false,
          "severity": "safe"
        }
      }
    }
  ]
}
"""
