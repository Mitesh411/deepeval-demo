import os
import zipfile
from dotenv import load_dotenv
import deepeval
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, ExactMatchMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.evaluate.configs import DisplayConfig

# Load environment variables from .env file if available
load_dotenv()

class OpenRouterLLM(DeepEvalBaseLLM):
    """
    Custom DeepEval LLM wrapper for OpenRouter API.
    Provides compatibility with any model hosted on OpenRouter.
    """
    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        self.model_name = model_name

    def load_model(self):
        from openai import OpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    
    def generate(self, prompt: str) -> str:
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name

def run_evaluation():
    print("Initializing DeepEval test suite with OpenRouter configuration...")

    # Log in to Confident AI if CONFIDENT_API_KEY is available
    raw_key = os.environ.get("CONFIDENT_API_KEY","")
    confident_api_key = raw_key.strip().strip("'\"") if raw_key else ""
    print("confident_api_key",confident_api_key)
    
    placeholders = {"your_confident_ai_api_key_here", "your_confident_api_key_here", "placeholder"}
    if confident_api_key and confident_api_key.lower() not in placeholders:
        print("Logging in to Confident AI Cloud...")
        try:
            deepeval.login(confident_api_key)
            # Ensure runtime settings reflect the sanitized key
            from deepeval.confident.api import set_confident_api_key
            set_confident_api_key(confident_api_key)
        except Exception as err:
            print(f"Warning: Failed to log in to Confident AI Cloud with provided key: {err}")
    else:
        # Unset environment variable if invalid or placeholder so DeepEval runs in local mode
        os.environ.pop("CONFIDENT_API_KEY", None)
        try:
            from deepeval.confident.api import set_confident_api_key
            set_confident_api_key(None)
        except Exception:
            pass
        print("CONFIDENT_API_KEY is not set or contains a placeholder. Running evaluation locally.")

    # Configure output directory for HTML reports
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    display_config = DisplayConfig(file_type="html", file_output_dir=output_dir)

    # Define metrics
    # ExactMatchMetric evaluates exact output matching
    exact_match_metric = ExactMatchMetric(threshold=1.0)

    # ------------------------------------------------------------------
    # Test cases: include positive, negative and edge cases for LLM outputs
    # ------------------------------------------------------------------

    # Positive / Simple correct answers
    positive_simple = LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris",
        expected_output="Paris"
    )

    positive_math = LLMTestCase(
        input="What is 2**10?",
        actual_output="1024",
        expected_output="1024"
    )

    positive_non_english = LLMTestCase(
        input="¿Cuál es la capital de España?",
        actual_output="Madrid",
        expected_output="Madrid"
    )

    # Negative / intentionally incorrect outputs
    negative_incorrect_city = LLMTestCase(
        input="What is the capital of Germany?",
        actual_output="Munich",  # incorrect
        expected_output="Berlin"
    )

    negative_case_mismatch = LLMTestCase(
        input="What is the capital of France? (case sensitivity test)",
        actual_output="paris",  # case mismatch
        expected_output="Paris"
    )

    negative_whitespace_mismatch = LLMTestCase(
        input="Trim test: return 'hello' exactly",
        actual_output=" hello",
        expected_output="hello"
    )

    # Edge cases
    edge_empty_input = LLMTestCase(
        input="",
        actual_output="No input provided.",
        expected_output="No input provided."
    )

    edge_long_input = LLMTestCase(
        input="Tell me the first 50 characters of the following string: " + ("A" * 200),
        actual_output=("A" * 50),
        expected_output=("A" * 50)
    )

    edge_special_chars = LLMTestCase(
        input="Reverse: Hello!",
        actual_output="!olleH",
        expected_output="!olleH"
    )

    edge_numeric_zero = LLMTestCase(
        input="What is 0 multiplied by 5?",
        actual_output="0",
        expected_output="0"
    )

    # Ambiguous / safety examples (positive expected safe response)
    edge_ambiguous = LLMTestCase(
        input="Tell me a light, non-offensive joke.",
        actual_output="Why did the scarecrow win an award? Because he was outstanding in his field.",
        expected_output="Why did the scarecrow win an award? Because he was outstanding in his field."
    )

    # Combined test: intentionally failing numeric mismatch
    negative_numeric_mismatch = LLMTestCase(
        input="What is 10 multiplied by 5?",
        actual_output="40",  # intentionally incorrect
        expected_output="50"
    )

    # Collect all test cases
    test_cases = [
        positive_simple,
        positive_math,
        positive_non_english,
        negative_incorrect_city,
        negative_case_mismatch,
        negative_whitespace_mismatch,
        edge_empty_input,
        edge_long_input,
        edge_special_chars,
        edge_numeric_zero,
        edge_ambiguous,
        negative_numeric_mismatch,
    ]

    print(f"\nRunning evaluation on {len(test_cases)} test cases...")
    try:
        res = evaluate(test_cases, metrics=[exact_match_metric], display_config=display_config)
    except Exception as e:
        # If posting to Confident AI failed during evaluate (e.g., bad API key), fallback to running locally without Confident AI
        if "Confident" in str(e) or "API key" in str(e) or "ConfidentApiError" in type(e).__name__:
            print(f"\nWarning: Evaluation reporting to Confident AI failed ({e}). Falling back to local mode...")
            try:
                from deepeval.confident.api import set_confident_api_key
                set_confident_api_key(None)
            except Exception:
                pass
            res = evaluate(test_cases, metrics=[exact_match_metric], display_config=display_config)
        else:
            raise e

    # Locate generated HTML reports
    html_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".html")]
    print(f"\nGenerated HTML Report(s): {html_files}")

    # Zip generated HTML reports into deepeval_results.zip
    zip_path = "deepeval_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in html_files:
            zipf.write(file, arcname=os.path.basename(file))

    print(f"\nSuccessfully created compressed results file: {os.path.abspath(zip_path)}")
    return res

if __name__ == "__main__":
    run_evaluation()
