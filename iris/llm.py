import os
from groq import Groq

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set in environment")
        _client = Groq(api_key=api_key)
    return _client


def _ask(prompt: str) -> str:
    response = _get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def get_treatments(disease: str) -> str:
    return _ask(
        f"What are the treatment options and precautions for {disease}? "
        "Write in concise bullet points."
    )


def get_diet_chart(concern: str) -> str:
    return _ask(
        f"For {concern}, list 4 foods to eat and 4 foods to avoid. "
        "Keep it brief and in bullet points."
    )
