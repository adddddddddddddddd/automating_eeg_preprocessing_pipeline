import os
import base64

from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import logging  
logging.basicConfig(level=logging.INFO)
api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-small-2506"

client = Mistral(api_key=api_key)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this image?"
            },
            {
                "type": "image_url",
                "image_url": "https://image2url.com/r2/default/images/1772305816515-4f9a58aa-1afa-4efc-8ff2-875aa5534fd9.jpg"
            }
        ]
    }
]

chat_response = client.chat.complete(
    model=model,
    messages=messages
)
logging.info(f"Model response: {chat_response.choices[0].message.content}")