import os
import requests
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def ask_mistral(prompt, system_prompt="You are a helpful AI teacher. Answer like you're teaching a student."):
    """Call the Mistral API and return the model's response."""
    if not MISTRAL_API_KEY:
        raise ValueError("Mistral API key not found. Set MISTRAL_API_KEY in .env file.")

    url = "https://api.mistral.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-medium",  # You can use mistral-small, mistral-medium, etc.
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except requests.exceptions.HTTPError as e:
        print("❌ Mistral API Error:", e)
        raise
