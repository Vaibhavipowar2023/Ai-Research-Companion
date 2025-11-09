from openai import OpenAI
import os
import json

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY. Set it in your environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)

def try_llm_completion(prompt: str, expect_json: bool = False, max_tokens: int = 500):
    """
    Calls the OpenAI model and returns either raw text or parsed JSON.
    """
    messages = [
        {"role": "system", "content": "You are a helpful, concise research assistant."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )

        content = response.choices[0].message.content.strip()

        if expect_json:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"error": "Model did not return valid JSON.", "raw": content}

        return content

    except Exception as e:
        return {"error": str(e)}

def generate_abstractive(prompt: str, max_tokens: int = 300) -> str:
    """
    Generates a concise abstractive summary for a given research prompt.
    """
    resp = try_llm_completion(prompt, expect_json=False, max_tokens=max_tokens)

    if isinstance(resp, dict) and "error" in resp:
        raise RuntimeError(f"LLM Error: {resp['error']}")

    return resp
