"""
Quick utility: list the Groq models actually available to your API key.

Usage:
    python scripts/list_groq_models.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

from groq import Groq

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("GROQ_API_KEY not found in .env")
    raise SystemExit(1)

client = Groq(api_key=api_key)
models = client.models.list()

print(f"\n{len(models.data)} models available to your Groq account:\n")
for m in sorted(models.data, key=lambda x: x.id):
    print(f"  - {m.id}")
