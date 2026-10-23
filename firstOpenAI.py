from openai import OpenAI
from dotenv import load_dotenv
import os
import yaml


load_dotenv()

with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

print(config['models']['liquid'])

client = OpenAI(
  base_url=config['base_url'],
  api_key=os.getenv("OPENAI_ROUTER_KEY"),
)

completion = client.chat.completions.create(
  model=config['model'],
  messages=[
    {
      "role": "user",
      "content": "Say this is a test",
    },
  ],
)

print(completion.choices[0].message.content)