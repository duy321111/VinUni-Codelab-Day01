"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call OpenAI using OPENAI_API_KEY and return text, latency and usage."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        )
        latency = time.perf_counter() - start
        usage = response.usage
        return response.choices[0].message.content or "", latency, {
            "input_tokens": (usage.prompt_tokens or 0) if usage else 0,
            "output_tokens": (usage.completion_tokens or 0) if usage else 0,
        }
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig

        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig

        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens'
        from the response metadata (e.g. response.usage_metadata).
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        start = time.perf_counter()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature, top_p=top_p, max_output_tokens=max_tokens
            ),
        )
        latency = time.perf_counter() - start
        usage = response.usage_metadata
        return response.text or "", latency, {
            "input_tokens": (usage.prompt_token_count or 0) if usage else 0,
            "output_tokens": (usage.candidates_token_count or 0) if usage else 0,
        }
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call Anthropic using ANTHROPIC_API_KEY; never route to Gemini."""
    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    try:
        start = time.perf_counter()
        response = client.messages.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        # Retain the lab signature, but omit sampling parameters for compatibility
        # with Anthropic API versions that no longer accept them.
        latency = time.perf_counter() - start
        return "".join(block.text for block in response.content
                       if block.type == "text"), latency, {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """Compare GPT-4o, GPT-4o-mini and Gemini using estimated lab rates."""
    results = {}
    for name, model, call in (
        ("gpt4o", OPENAI_MODEL, call_openai),
        ("gpt4o_mini", OPENAI_MINI_MODEL, call_openai),
        ("gemini_flash", GEMINI_MODEL, call_gemini),
    ):
        text, latency, usage = call(prompt, model=model)
        rates = PRICING_1M_TOKENS[model]
        cost = (
            usage["input_tokens"] * rates["input"]
            + usage["output_tokens"] * rates["output"]
        ) / 1_000_000
        results[name] = {
            "response": text, "latency": latency, "cost": cost, **usage
        }
    return results


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history = []
    try:
        while True:
            try:
                prompt = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.lower() in {"quit", "exit"}:
                break
            if not prompt:
                continue
            user_message = types.Content(
                role="user", parts=[types.Part.from_text(text=prompt)]
            )
            contents = history + [user_message]
            pieces = []
            print("Gemini: ", end="", flush=True)
            for chunk in client.models.generate_content_stream(
                model=GEMINI_MODEL, contents=contents
            ):
                if chunk.text:
                    pieces.append(chunk.text)
                    print(chunk.text, end="", flush=True)
            print()
            history = (contents + [types.Content(
                role="model", parts=[types.Part.from_text(text="".join(pieces))]
            )])[-6:]
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    if max_retries < 0 or base_delay < 0:
        raise ValueError("max_retries and base_delay must be non-negative")
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * 2 ** attempt)


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    return [{**compare_models(prompt), "prompt": prompt} for prompt in prompts]


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    def escape_cell(value: str) -> str:
        return value.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    rows = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    labels = {"gpt4o": "GPT-4o", "gpt4o_mini": "GPT-4o-mini",
              "gemini_flash": "Gemini-Flash"}
    for result in results:
        for key, label in labels.items():
            if key not in result:
                continue
            stats = result[key]
            text = stats["response"]
            if len(text) > 50:
                text = text[:47] + "..."
            rows.append(
                f"| {escape_cell(result['prompt'])} | {label} | {escape_cell(text)} "
                f"| {stats['latency']:.2f}s | {stats['input_tokens']}/{stats['output_tokens']} "
                f"| ${stats['cost']:.8f} |"
            )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
