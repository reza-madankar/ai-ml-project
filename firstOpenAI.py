import os
from openai import OpenAI

# gets API Key from environment variable OPENROUTER_API_KEY
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENAI_ROUTER_KEY"),
)

completion = client.chat.completions.create(
  model="liquid/lfm-2.5-2.6b:free",
  # pass extra_body to access OpenRouter-only arguments.
  # extra_body={
    # "models": [
    #   "openai/gpt-4o",
    #   "mistralai/mixtral-8x22b-instruct"
    # ]
  # },
  messages=[
    {
      "role": "user",
      "content": "Say this is a test",
    },
  ],
)
print(completion.choices[0].message.content)