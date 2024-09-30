import os
from dotenv import load_dotenv
import chainlit as cl
import openai
import json
import re
from movie_functions import get_now_playing_movies, get_showtimes, get_reviews, pick_random_movie

load_dotenv()

# Note: If switching to LangSmith, uncomment the following, and replace @observe with @traceable
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# client = wrap_openai(openai.AsyncClient())

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
 
client = AsyncOpenAI()

gen_kwargs = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 500
}

SYSTEM_PROMPT = """\
You are the helpful agent responsible for finding movies for users. 
When a user inquires about movies currently playing, you should call the function in the following format.

1. When user asks about what's playing in movie theater
{
    "function":"get_now_playing_movies",
    "rationale": "explain why you called this function"
    
}
2. When user asks specific show time of movies
{
    "function":"get_showtimes(title, location)",
    "rationale": "explain why you called this function"
    
}
3. When user asks about review of specific movie
{
    "function":"get_reviews(movie_id)",
    "rationale": "explain why you called this function"
}

4. when there are multiple requests from the user at once, you may need to run multiple function calls.
"""


@observe
@cl.on_chat_start
def on_chat_start():    
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)
    
    await response_message.update()

    return response_message

@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})
    
    response_message = await generate_response(client, message_history, gen_kwargs)

    message_history.append({"role": "assistant", "content": response_message.content})
    
    cl.user_session.set("message_history", message_history)

    print(response_message.content)

    if response_message.content.startswith("{"):
        try:
            function_call = json.loads(response_message.content)
            if function_call.get("function") == "get_now_playing_movies":
                result = get_now_playing_movies()
                message_history.append({"role": "assistant", "content": result})
                await cl.Message(content=f"Result: {result}").send()
            elif function_call.get("function") == "get_showtimes":
                title = function_call.get("title", None)
                location = function_call.get("location", None)
                if not title or not location:
                    await cl.Message(content="Please provide both title and location for showtimes.").send()
                else:
                    result = get_showtimes(title, location)
                    message_history.append({"role": "assistant", "content": result})
                    await cl.Message(content=f"Result: {result}").send()
            elif function_call.get("function") == "get_reviews":
                result = get_reviews()
                message_history.append({"role": "assistant", "content": result})
                await cl.Message(content=f"Result: {result}").send()
            else:
                message_history.append({"role": "assistant", "content": "Unsupported function call."})
                await cl.Message(content="Unsupported function call.").send()
        except json.JSONDecodeError:
            response_message.content = "Invalid JSON format for function call."


if __name__ == "__main__":
    cl.main()