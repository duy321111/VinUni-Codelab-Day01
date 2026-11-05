"""
Day 1 — LLM API Foundation
Test suite for student solutions and master keys.

Run from the 02-lab/ folder:
    pytest tests/ -v

All external API calls are mocked — no real API keys or internet connection required.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Locate directories relative to this test file
TESTS_DIR = Path(__file__).parent
LAB_DIR = TESTS_DIR.parent

# Resolve module paths
SOLUTION_PATH = LAB_DIR / "solution-code" / "solution.py"
TEMPLATE_PATH = LAB_DIR / "starter-code" / "template.py"

def _load_module(path: Path, unique_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Target file not found at: {path}")
    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Dynamically load the solution module if it exists (for grading), otherwise test the starter template
if SOLUTION_PATH.exists():
    _m = _load_module(SOLUTION_PATH, "solution")
else:
    _m = _load_module(TEMPLATE_PATH, "template")

# Import target functions
call_openai = getattr(_m, "call_openai")
call_gemini = getattr(_m, "call_gemini")
call_anthropic = getattr(_m, "call_anthropic")
compare_models = getattr(_m, "compare_models")
streaming_chatbot = getattr(_m, "streaming_chatbot")
retry_with_backoff = getattr(_m, "retry_with_backoff")
batch_compare = getattr(_m, "batch_compare")
format_comparison_table = getattr(_m, "format_comparison_table")


# ---------------------------------------------------------------------------
# Mock Generators
# ---------------------------------------------------------------------------
def _make_openai_response(text: str = "Hello from OpenAI"):
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    resp.usage = usage
    return resp


def _make_gemini_response(text: str = "Hello from Gemini"):
    resp = MagicMock()
    resp.text = text
    usage = MagicMock()
    usage.prompt_token_count = 12
    usage.candidates_token_count = 25
    resp.usage_metadata = usage
    return resp


def _make_anthropic_response(text: str = "Hello from Anthropic"):
    resp = MagicMock()
    content_part = MagicMock()
    content_part.type = "text"
    content_part.text = text
    resp.content = [content_part]
    usage = MagicMock()
    usage.input_tokens = 15
    usage.output_tokens = 30
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
class TestProviderCalls(unittest.TestCase):

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai", "ANTHROPIC_API_KEY": "test-anthropic"})
    def test_providers_use_separate_clients(self):
        cases = (
            (call_openai, "openai.OpenAI", "test-openai", _make_openai_response(), "Hello from OpenAI", 10, 20),
            (call_anthropic, "anthropic.Anthropic", "test-anthropic", _make_anthropic_response(), "Hello from Anthropic", 15, 30),
        )
        for fn, target, key, response, expected, tokens_in, tokens_out in cases:
            with self.subTest(provider=target), patch(target) as factory, patch.object(_m, "call_gemini") as gemini:
                client = factory.return_value
                endpoint = client.chat.completions.create if fn is call_openai else client.messages.create
                endpoint.return_value = response
                text, latency, usage = fn("Hello", temperature=0.2, top_p=0.8, max_tokens=100)
                factory.assert_called_once_with(api_key=key)
                self.assertEqual(text, expected)
                self.assertGreaterEqual(latency, 0)
                self.assertEqual(usage, {"input_tokens": tokens_in, "output_tokens": tokens_out})
                self.assertEqual(endpoint.call_args.kwargs["messages"], [{"role": "user", "content": "Hello"}])
                self.assertEqual(endpoint.call_args.kwargs["max_tokens"], 100)
                gemini.assert_not_called()
                client.close.assert_called_once()

    def test_clients_close_on_failure(self):
        for fn, target in ((call_openai, "openai.OpenAI"), (call_anthropic, "anthropic.Anthropic")):
            with self.subTest(provider=target), patch(target) as factory:
                client = factory.return_value
                endpoint = client.chat.completions.create if fn is call_openai else client.messages.create
                endpoint.side_effect = RuntimeError("API unavailable")
                with self.assertRaises(RuntimeError):
                    fn("Hello")
                client.close.assert_called_once()


class TestCallGemini(unittest.TestCase):

    @patch("google.genai.Client")
    def test_gemini_returns_tuple_structure_new_sdk(self, MockGenAIClient):
        mock_client = MagicMock()
        MockGenAIClient.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response("Gemini Mock Response")

        text, latency, usage = call_gemini("Hello")

        self.assertEqual(text, "Gemini Mock Response")
        self.assertIsInstance(latency, float)
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(usage["input_tokens"], 12)
        self.assertEqual(usage["output_tokens"], 25)


class TestCompareModels(unittest.TestCase):

    def test_compare_uses_openai_and_gemini_with_lab_rates(self):
        response = ("response", 0.4, {"input_tokens": 10, "output_tokens": 20})
        with patch.object(_m, "call_openai", return_value=response) as openai, patch.object(_m, "call_gemini", return_value=response) as gemini:
            result = compare_models("Prompt")
            self.assertEqual(openai.call_count, 2)
            openai.assert_any_call("Prompt", model=_m.OPENAI_MODEL)
            openai.assert_any_call("Prompt", model=_m.OPENAI_MINI_MODEL)
            gemini.assert_called_once_with("Prompt", model=_m.GEMINI_MODEL)
            self.assertEqual(set(result), {"gpt4o", "gpt4o_mini", "gemini_flash"})
            for key, model in (("gpt4o", _m.OPENAI_MODEL), ("gpt4o_mini", _m.OPENAI_MINI_MODEL), ("gemini_flash", _m.GEMINI_MODEL)):
                rates = _m.PRICING_1M_TOKENS[model]
                self.assertAlmostEqual(result[key]["cost"], (10 * rates["input"] + 20 * rates["output"]) / 1_000_000)
            table = format_comparison_table([{**result, "prompt": "Prompt"}])
            for label in ("GPT-4o", "GPT-4o-mini", "Gemini-Flash"):
                self.assertIn(label, table)


class TestRetryWithBackoff(unittest.TestCase):

    def test_succeeds_immediately(self):
        val = retry_with_backoff(lambda: "success")
        self.assertEqual(val, "success")

    def test_succeeds_after_retries(self):
        state = {"attempts": 0}

        def flaky_fn():
            state["attempts"] += 1
            if state["attempts"] < 3:
                raise ValueError("Fail")
            return "ok"

        val = retry_with_backoff(flaky_fn, max_retries=3, base_delay=0.001)
        self.assertEqual(val, "ok")
        self.assertEqual(state["attempts"], 3)

    def test_raises_permanent_exception(self):
        def permanent_fail():
            raise RuntimeError("Permanent")

        with self.assertRaises(RuntimeError):
            retry_with_backoff(permanent_fail, max_retries=2, base_delay=0.001)


class TestBatchCompareAndFormat(unittest.TestCase):

    def test_batch_runs_and_formats_table(self):
        with patch.object(_m, "compare_models") as mock_comp:
            def _get_mock(prompt):
                return {
                    "gemini_flash": {"response": "Gemini response content here", "latency": 0.4, "cost": 0.00000675, "input_tokens": 10, "output_tokens": 20}
                }
            mock_comp.side_effect = _get_mock

            results = batch_compare(["Q1", "Q2"])
            self.assertEqual(mock_comp.call_count, 2)
            mock_comp.assert_any_call("Q1")
            mock_comp.assert_any_call("Q2")
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["prompt"], "Q1")

            table = format_comparison_table(results)
            self.assertIsInstance(table, str)
            self.assertIn("Prompt", table)
            self.assertIn("Gemini-Flash", table)
            self.assertIn("Q1", table)


class TestStreamingChatbot(unittest.TestCase):

    @patch("builtins.input", side_effect=["quit"])
    @patch("google.genai.Client")
    def test_exits_cleanly(self, MockClient, mock_input):
        # Ensure it quits cleanly without entering infinite loop
        try:
            streaming_chatbot()
        except SystemExit:
            pass


if __name__ == "__main__":
    unittest.main()
