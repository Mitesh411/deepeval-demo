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
    raw_key = os.environ.get("CONFIDENT_API_KEY", confident_us_proj_Asq2JfReDTpZ5RlQSyA4kqxHFfx7hH+W3JwLE9r06aU=)
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

    # 1. Passing Test Case: LLM answer matches expected output exactly
    passing_test_case = LLMTestCase(
    #    input="What is the capital of France?",
    #    actual_output="Paris",
    #    expected_output="Paris"
        input="What is the primary function of an LLM evaluation framework?",
        actual_output=
        "An LLM evaluation framework provides systematic metrics and benchmark assertions to test the accuracy, relevancy, safety, and hallucination rates of LLM responses.",
        expected_output="An LLM evaluation framework provides systematic metrics and benchmark assertions to test the accuracy, relevancy, safety, and hallucination rates of LLM responses."
    )

    # 2. Intentionally Failing Test Case: LLM answer differs from expected output
    failing_test_case = LLMTestCase(
        input="What is the capital of Germany?",
        actual_output="Munich",  # Intentionally incorrect actual output (Capital is Berlin)
        expected_output="Berlin"
    )


    passing_test_case2 = LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris",
        expected_output="Paris"
    )

    passing_test_case3 = LLMTestCase(
        input="What is 10 multiplied by 5?",
        actual_output="40",
        expected_output="50"
    )




    test_cases = [passing_test_case, failing_test_case, passing_test_case2,passing_test_case3]

    print("\nRunning evaluation on test cases...")
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
