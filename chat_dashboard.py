"""Simple Streamlit chat bot.

Run it from the project folder:

    uv run streamlit run chat_dashboard.py

How a chat app works
--------------------
1. Streamlit reruns this whole file every time you type or click.
2. Normal variables disappear on each rerun. st.session_state keeps the chat.
3. Each message is a dict: {"role": "user" or "assistant", "content": "..."}.
4. On every new message we send the full list to the model, so it remembers.
5. stream=True returns the answer one token at a time. st.write_stream
   paints those tokens as they arrive, instead of waiting for the full reply.
6. st.spinner("Thinking...") is the spinner. It stays on only until the
   first token arrives, then the typed reply replaces it.
"""

import os

import streamlit as st
import yaml
from dotenv import load_dotenv
from openai import OpenAI


def load_client(config_path="config.yaml"):
    """Read config.yaml and build the same OpenAI client as currencyAI.py."""
    load_dotenv()
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    provider = config["providers"][config["provider"]]
    client = OpenAI(
        base_url=provider["base_url"],
        api_key=os.getenv(provider["api_key_env"]),
    )
    return client, provider["model"]


st.set_page_config(page_title="Chat Bot", page_icon="💬")
st.title("Chat Bot")
st.caption("Type a message. The model sees the whole conversation, not only the last line.")

client, model = load_client()

# First visit: start with an empty conversation.
# Later visits in the same browser tab reuse this list.
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    st.write(f"Model: `{model}`")
    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful tutor. Answer in short, clear sentences.",
        height=140,
        help="Instructions for the model. The user never sees this as a chat bubble.",
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Redraw every saved message. This runs again after each reply.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_text = st.chat_input("Type a message")
if not user_text:
    st.stop()

# Save and show the user's line before we call the model.
st.session_state.messages.append({"role": "user", "content": user_text})
with st.chat_message("user"):
    st.markdown(user_text)

# The API call: system instructions first, then every user and assistant turn.
api_messages = [{"role": "system", "content": system_prompt}]
api_messages.extend(st.session_state.messages)

def token_stream():
    """Yield each text token as the model produces it."""
    stream = client.chat.completions.create(
        model=model,
        messages=api_messages,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content
        if token:
            yield token


with st.chat_message("assistant"):
    try:
        # Spinner lives here. The block ends as soon as the first token arrives.
        with st.spinner("Thinking..."):
            tokens = token_stream()
            first_token = next(tokens, "")

        if first_token:
            def reply_tokens():
                yield first_token
                yield from tokens

            reply = st.write_stream(reply_tokens()) or "(empty reply)"
        else:
            reply = "(empty reply)"
            st.markdown(reply)
    except Exception as error:
        reply = f"Request failed: {error}"
        st.markdown(reply)

st.session_state.messages.append({"role": "assistant", "content": reply})
