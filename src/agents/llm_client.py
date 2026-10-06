"""
Universal LLM caller with provider abstraction and automatic fallback.
"""
import os
import json
from typing import List, Dict, Any, Optional
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "openai/gpt-oss-20b")

# Provider clients
groq_client = None
hf_primary_client = None
hf_fallback_client = None
openai_client = None

if LLM_PROVIDER == "groq":
    from groq import Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY or GROQ_API_KEY == "gsk_your_key_here":
        print("\n[ERROR] GROQ_API_KEY is missing.")
        print("Add it to .env and rerun.\n")
        import sys
        sys.exit(1)
    else:
        groq_client = Groq(api_key=GROQ_API_KEY)
elif LLM_PROVIDER == "huggingface":
    from huggingface_hub import InferenceClient
    HF_TOKEN = os.getenv("HF_TOKEN")
    hf_primary_client = InferenceClient(
        base_url=f"https://api-inference.huggingface.co/models/{PRIMARY_MODEL}/v1",
        token=HF_TOKEN,
    )
    hf_fallback_client = InferenceClient(
        base_url=f"https://api-inference.huggingface.co/models/{FALLBACK_MODEL}/v1",
        token=HF_TOKEN,
    )
elif LLM_PROVIDER == "openai":
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("\n[ERROR] OPENAI_API_KEY is missing.")
        print("Add it to .env and rerun.\n")
        import sys
        sys.exit(1)
    else:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

@traceable(name="LLM Call")
def call_llm(messages: List[Dict[str, str]], max_tokens: int = 1500, temperature: float = 0.1) -> str:
    """
    Call the primary LLM, falling back to the secondary if it fails.
    """
    import time
    import random
    
    last_error = None
    backoff_schedule = [0, 1, 2, 4]  # Maximum 4 attempts
    
    for attempt, delay in enumerate(backoff_schedule):
        if delay > 0:
            jitter = random.uniform(-0.3, 0.3)
            sleep_time = max(0, delay + jitter)
            print(f"[RETRY] Attempt {attempt + 1}/4 for LLM call. Sleeping for {sleep_time:.2f}s...")
            time.sleep(sleep_time)
            
        if LLM_PROVIDER == "groq":
            if not groq_client:
                raise RuntimeError("Groq client is not initialized. Please set GROQ_API_KEY.")
                
            # Primary
            try:
                response = groq_client.chat.completions.create(
                    messages=messages,
                    model=PRIMARY_MODEL,
                    max_completion_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                print(f"Primary ({PRIMARY_MODEL}) failed on attempt {attempt + 1}: {e}")

            # Fallback
            try:
                response = groq_client.chat.completions.create(
                    messages=messages,
                    model=FALLBACK_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                print(f"Fallback ({FALLBACK_MODEL}) failed on attempt {attempt + 1}: {e}")
                
        elif LLM_PROVIDER == "huggingface":
            # Primary
            try:
                response = hf_primary_client.chat.completions.create(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                print(f"Primary ({PRIMARY_MODEL}) failed on attempt {attempt + 1}: {e}")

            # Fallback
            try:
                response = hf_fallback_client.chat.completions.create(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                print(f"Fallback ({FALLBACK_MODEL}) failed on attempt {attempt + 1}: {e}")
                
        elif LLM_PROVIDER == "openai":
            if not openai_client:
                raise RuntimeError("OpenAI client is not initialized. Please set OPENAI_API_KEY.")
            
            # Primary
            try:
                response = openai_client.chat.completions.create(
                    messages=messages,
                    model=PRIMARY_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                print(f"Primary ({PRIMARY_MODEL}) failed on attempt {attempt + 1}: {e}")

            # Fallback
            try:
                response = openai_client.chat.completions.create(
                    messages=messages,
                    model=FALLBACK_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                print(f"Fallback ({FALLBACK_MODEL}) failed on attempt {attempt + 1}: {e}")
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

    raise RuntimeError(f"All 4 attempts failed: {last_error}")

def call_llm_json(messages: List[Dict[str, str]], max_tokens: int = 1500, temperature: float = 0.1) -> Any:
    """
    Helper to call the LLM and parse the response as JSON.
    Includes robust partial JSON recovery and a fallback repair prompt.
    """
    # Reinforce strict JSON-only output. Always append this even if the
    # system prompt already mentions "JSON" generically — reasoning models
    # (e.g. gpt-oss) benefit from the explicit "no explanation, no preamble"
    # instruction specifically, which a generic mention doesn't cover.
    messages.append({
        "role": "system",
        "content": "Return ONLY valid JSON. No markdown fences, no explanation, no reasoning text before or after the JSON."
    })
        
    response_text = call_llm(messages, max_tokens=max_tokens, temperature=temperature)
    
    def clean_json(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Reasoning models (e.g. gpt-oss) often prepend chain-of-thought or
        # commentary before the actual JSON, which the fence-stripping above
        # doesn't catch. Trim anything before the first '{' or '[' so we start
        # parsing at the actual JSON, not the preamble.
        first_brace = cleaned.find("{")
        first_bracket = cleaned.find("[")
        candidates = [p for p in (first_brace, first_bracket) if p != -1]
        if candidates:
            start = min(candidates)
            if start > 0:
                cleaned = cleaned[start:]
                
            last_brace = cleaned.rfind("}")
            last_bracket = cleaned.rfind("]")
            end_candidates = [p for p in (last_brace, last_bracket) if p != -1]
            if end_candidates:
                end = max(end_candidates)
                if end < len(cleaned) - 1:
                    cleaned = cleaned[:end+1]

        return cleaned.strip()

    cleaned = clean_json(response_text)
    
    try:
        return json.loads(cleaned), "success", response_text
    except json.JSONDecodeError as e:
        # 1. Attempt partial JSON recovery
        try:
            start = cleaned.find('[')
            end = cleaned.rfind('}')
            if start != -1 and end != -1 and end > start:
                partial = cleaned[start:end+1] + "\n]"
                return json.loads(partial), "partial_recovery", response_text
        except json.JSONDecodeError:
            pass
            
        # 2. Fallback Repair Prompt
        print(f"[WARNING] JSON parse failed, triggering fallback repair prompt.")
        repair_messages = [
            {
                "role": "system",
                "content": "Return ONLY valid JSON. Do not add explanations."
            },
            {
                "role": "user",
                "content": f"Repair this JSON:\n{response_text}"
            }
        ]
        
        try:
            repaired_text = call_llm(repair_messages, max_tokens=max_tokens, temperature=temperature)
            cleaned_repaired = clean_json(repaired_text)
            return json.loads(cleaned_repaired), "fallback_repair", repaired_text
        except Exception as repair_error:
            print(f"[ERROR] JSON repair failed: {repair_error}")
            return None, "failed", response_text
