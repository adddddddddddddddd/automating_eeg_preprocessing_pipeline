import os
import base64

from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import logging  
logging.basicConfig(level=logging.INFO)
api_key = os.environ["MISTRAL_API_KEY"]
model = "magistral-medium-2509"

from pydantic import BaseModel, Field, field_validator
client = Mistral(api_key=api_key)


class EEGSlowDriftAnalysis(BaseModel):
    slow_drift_probability: float = Field(
        description="Probability between 0 and 1 indicating whether the EEG data shows signs of slow drifts"
    )
    
    @field_validator('slow_drift_probability')
    @classmethod
    def validate_probability(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Probability must be between 0 and 1')
        return v
    justification: str = Field(
        description="A brief explanation of the reasoning behind the assigned probability, referencing specific features in the EEG plot such as baseline shifts, trends across channels, or other relevant observations."
    )
messages = [{
  "role": "system",
  "content": [
    {
      "type": "text",
      "text": "# HOW YOU SHOULD THINK AND ANSWER\n\nFirst draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.\n\nYour thinking process must follow the template below:"
    },
    {
      "type": "thinking",
      "thinking": [
        {
          "type": "text",
          "text": "Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user."
        }
      ]
    },
    {
      "type": "text",
      "text": "Here, provide a self-contained response."
    }
  ]
}]
messages += [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "You are a helpful assistant for EEG data analysis. I will give you an image of an EEG plot. I want you to analyze the plot and say whether the data shows signs of slow drifts or not.In raw EEG recordings, slow drifts appear as a gradual upward or downward shift in the signal baseline across channels Respond by giving a probability between 0 and 1. 1 means the data is very likely to show slow drifts, 0 means it is very unlikely. Add a brief justification for your answer."
            },
            {
                "type": "image_url",
                "image_url": "https://image2url.com/r2/default/images/1772306560778-fe9e6950-cc51-4834-8d1f-3273333377f2.png"
            }
        ]
    }
]

chat_response = client.chat.parse(
    model=model,
    messages=messages,
    prompt_mode="reasoning",
    response_format=EEGSlowDriftAnalysis
)
logging.info(f"Model response: {chat_response.choices[0].message.content}")

