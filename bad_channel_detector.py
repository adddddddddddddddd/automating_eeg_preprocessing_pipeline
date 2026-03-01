import os
import base64

from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import logging  
logging.basicConfig(level=logging.INFO)
api_key = os.environ["MISTRAL_API_KEY"]
model = "magistral-small-2509"

from pydantic import BaseModel, Field, field_validator
client = Mistral(api_key=api_key)


class BadChannelAnalysis(BaseModel):
    bad_channels_to_remove: list = Field(
        description="List of bad channels to remove based on the analysis"
    )
    justification: str = Field(
        description="A brief explanation of the reasoning behind the selected bad channels, referencing specific features in the ICA plots such as artifacts, noise, or other relevant observations."
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
                "text": """You are a helpful assistant for EEG data analysis. I will give you an image of EEG channels. 
                I want you to analyze the plot and identify which channels should be removed. To answer, here are the rule to identify bad channels:
                - Channels with flat line or almost flat line across the entire recording are likely bad channels and should be tagged as removed.
                Also do a brief justification for your answer."""
            },
            {
                "type": "image_url",
                "image_url": "https://image2url.com/r2/default/images/1772315435288-58fcdbac-5cd9-4aa4-a48b-9c31a09412d5.png"
            }
        ]
    }
]

chat_response = client.chat.parse(
    model=model,
    messages=messages,
    prompt_mode="reasoning",
    response_format=BadChannelAnalysis,
    temperature=0.1,
)
logging.info(f"Model response: {chat_response.choices[0].message.content}")