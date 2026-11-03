import json
import os

import requests
import yaml
from dotenv import load_dotenv
from openai import OpenAI


def calculate(operation, a, b):
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    raise ValueError("Use multiply or divide")


def get_exchange_rate(from_currency, to_currency):
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    url = f"https://api.frankfurter.app/latest?from={from_currency}&to={to_currency}"
    data = requests.get(url, timeout=10).json()
    rate = data["rates"][to_currency]
    return {
        "from": from_currency,
        "to": to_currency,
        "rate": rate,
        "date": data["date"],
    }


# OpenAI needs this JSON so it knows which tools it can call.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get the latest exchange rate between two currencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": ["from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Multiply or divide two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["multiply", "divide"]},
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        },
    },
]


class CurrencyAI:
    def __init__(self, config_path="config.yaml"):
        load_dotenv()
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        provider = config["providers"][config["provider"]]
        self.model = provider["model"]
        self.client = OpenAI(
            base_url=provider["base_url"],
            api_key=os.getenv(provider["api_key_env"]),
        )

    def convert(self, user_text):
        messages = [
            {
                "role": "system",
                "content": (
                    "Convert currency. First call get_exchange_rate. "
                    "Then call calculate (amount * rate). "
                    "Do not invent numbers."
                ),
            },
            {"role": "user", "content": user_text},
        ]

        # The model may call tools more than once (rate, then math).
        for _ in range(5):
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
            )
            reply = completion.choices[0].message
            messages.append(reply)

            print("-------------------------------- ###")
            print("reply: ", reply)
            print("messages: ", messages)
            print("--------------------------------")
            # No tool call means the model is done and we can show the answer.
            if not reply.tool_calls:
                return reply.content

            for tool_call in reply.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if name == "get_exchange_rate":
                    result = get_exchange_rate(**args)
                elif name == "calculate":
                    result = calculate(**args)
                else:
                    result = {"error": f"Unknown tool: {name}"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

                print("messages: ", messages)
                print("tool_call: ", tool_call)
                print("result: ", result)

        return "Too many tool calls."


if __name__ == "__main__":
    text = input("What do you want to convert? ")
    print(CurrencyAI().convert(text))
