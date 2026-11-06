import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {os.getenv('OPENAI_ROUTER_KEY')}"
  },
  data=json.dumps({
    "model": "liquid/lfm-2.5-2.6b:free",
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  })
)


print(response.json())