from dotenv import load_dotenv
from openai import OpenAI

# Small, cheap chat model. Check https://platform.openai.com/docs/models
# and swap this for a newer small model if one is available.
MODEL = "gpt-5.4-nano"

load_dotenv()  # reads OPENAI_API_KEY from .env into the environment
client = OpenAI()  # picks the key up automatically

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        # Experiment: change this to "Answer in one sentence."
        {
            "role": "system",
            "content": "You are a trash-talking UFC fighter. Always speak in that voice: "
            "use fight metaphors, short punchy sentences, and trash talk. Never break character.",
        },
        # Experiment: ask "How does login work in my project?"
        {"role": "user", "content": "What is a REST API?"},
    ],
)

print(response.choices[0].message.content)
print()
print(response.usage)  # prompt_tokens + completion_tokens = what you pay for
