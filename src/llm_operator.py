import os, re, logging, requests, time
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# ── Swap model here if needed ─────────────────────────────────
# GPT-4o-mini free tier: ~3 RPM → use 22s delay
# Gemini 2.5 Flash free tier: ~10 RPM → use 7s delay
API_PROVIDER = "openai"   # or "gemini"
OPENAI_MODEL  = "gpt-4o-mini"
GEMINI_MODEL  = "gemini-2.5-flash"
REQUEST_DELAY = 22        # seconds between calls — adjust per provider


def build_evolution_prompt(K, parent_fracs, sum_rates):
    n_vars = K + 1
    prompt = (
        f"You are an expert in LEO satellite communication system optimization. "
        f"There is a power allocation optimization problem for RSMA systems to "
        f"maximize the sum rate of all {K} ground stations (GSs).\n\n"
        f"The problem has 1 objective with {n_vars} variables:\n"
        f"- Objective: Maximize the sum rate performance (higher is better)\n"
        f"- Variables (each is a fraction of total power in [0,1]; "
        f"all values >= 0, SUM <= 1.0):\n"
        f"  Variable 1  : power fraction of the common stream\n"
        f"  Variables 2-{n_vars}: power fraction of each private stream\n\n"
        f"Existing strategies with objective values (higher is better):\n"
    )
    for rate, sol in sorted(zip(sum_rates, parent_fracs), key=lambda x: x[0]):
        prompt += f"<solution>{', '.join(f'{v:.4f}' for v in sol)}</solution>\nObjective: {rate:.4f}\n"
    prompt += (
        f"\nTask: Provide a new <solution> that:\n"
        f"1. Is distinct from all above strategies\n"
        f"2. Maximizes the sum-rate objective\n"
        f"3. Has exactly {n_vars} non-negative values whose sum is <= 1.0\n\n"
        f"Think step by step. Output ONLY: <solution>v1, v2, ..., v{n_vars}</solution>"
    )
    return prompt


def parse_solution(text, K):
    matches = re.findall(r"<solution>(.*?)</solution>", text, re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    for raw in reversed(matches):
        tokens = re.split(r"[,\s]+", raw.strip())
        try:
            vals = [float(t) for t in tokens if t]
            if len(vals) == K + 1:
                return np.array(vals, dtype=float)
        except ValueError:
            continue
    return None


def _call_openai(prompt, K):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.7,
    }
    resp = requests.post("https://api.openai.com/v1/chat/completions",
                         headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return parse_solution(text, K)


def _call_gemini(prompt, K):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={api_key}")
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 256, "temperature": 0.7},
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return parse_solution(text, K)


def llm_generate_offspring(K, parent_fracs, sum_rates):
    time.sleep(REQUEST_DELAY)  # ← prevents 429 before it happens

    prompt = build_evolution_prompt(K, parent_fracs, sum_rates)

    # Define primary and secondary functions based on your selection
    if API_PROVIDER == "openai":
        primary_call = _call_openai
        fallback_call = _call_gemini
        primary_name, fallback_name = "OpenAI", "Gemini"
    else:
        primary_call = _call_gemini
        fallback_call = _call_openai
        primary_name, fallback_name = "Gemini", "OpenAI"

    # Try the primary API first
    try:
        result = primary_call(prompt, K)
        if result is not None:
            return result
    except Exception as e:
        logger.warning(f"Primary {primary_name} API failed: {e}. Switching to {fallback_name}...")

    # If primary fails, immediately try the fallback API
    try:
        return fallback_call(prompt, K)
    except Exception as fallback_e:
        logger.warning(f"Fallback {fallback_name} API also failed: {fallback_e}")
        return None