# DeepEval & OpenRouter LLM Testing Guide

This repository provides a comprehensive, step-by-step guide and practical demonstration for evaluating Large Language Models (LLMs) using **DeepEval** and the **OpenRouter API**. It also demonstrates how to save evaluation reports into a compressed HTML zip archive and how DeepEval handles test case failures.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation & Setup](#installation--setup)
4. [Configuring DeepEval with OpenRouter API](#configuring-deepeval-with-openrouter-api)
   - [Method 1: Using DeepEval CLI (Recommended)](#method-1-using-deepeval-cli-recommended)
   - [Method 2: Using Custom Model Class (Python API)](#method-2-using-custom-model-class-python-api)
5. [Running LLM Evaluations](#running-llm-evaluations)
6. [Integrating Confident AI Cloud](#integrating-confident-ai-cloud)
7. [Demonstrating Error Handling (Intentionally Failing Test Case)](#demonstrating-error-handling-intentionally-failing-test-case)
8. [Saving and Zipping HTML Results](#saving-and-zipping-html-results)
9. [Example Execution & Output](#example-execution--output)
10. [Project Structure](#project-structure)

---

## Overview

[DeepEval](https://github.com/confident-ai/deepeval) is an open-source LLM evaluation framework designed to make unit testing LLM outputs fast, accurate, and easy. By integrating with [OpenRouter](https://openrouter.ai/), developers can test their applications against hundreds of LLMs (e.g., OpenAI, Anthropic, Meta Llama, Mistral) through a single unified API endpoint.

---

## Prerequisites

Before getting started, ensure you have the following installed on your system:

- **Python**: Version `3.10` or higher (`3.12+` recommended).
- **pip**: Python package installer.
- **OpenRouter API Key**: Obtainable from [openrouter.ai/keys](https://openrouter.ai/keys).

To verify your Python version:

```bash
python3 --version
```

---

## Installation & Setup

1. **Clone the Repository** (or navigate to your working directory):

   ```bash
   git clone <repository-url>
   cd deepeval-demo
   ```

2. **Install Dependencies**:

   Install `deepeval` along with `openai` and `python-dotenv`:

   ```bash
   pip install -r requirements.txt
   ```

   *Alternatively, install directly via pip:*

   ```bash
   pip install deepeval openai python-dotenv
   ```

3. **Configure Environment Variables**:

   Create a `.env` file in the root directory and add your OpenRouter API key and Confident AI API key:

   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   CONFIDENT_API_KEY=your_confident_ai_api_key_here
   ```

---

## Configuring DeepEval with OpenRouter API

DeepEval provides native integration for OpenRouter via CLI or programmatic custom model wrappers.

### Method 1: Using DeepEval CLI (Recommended)

You can set OpenRouter as your active LLM provider across DeepEval commands using the `deepeval set-openrouter` CLI command:

```bash
deepeval set-openrouter --model "openai/gpt-4o-mini" --save dotenv:.env.local
```

**Parameters Explained**:
- `--model` / `-m`: The OpenRouter model ID to use for metrics requiring an LLM judge (e.g., `openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`, `meta-llama/llama-3.3-70b-instruct`).
- `--save`: Persists configuration to environment variables in `.env.local` (or `.env`).

### Method 2: Using Custom Model Class (Python API)

If you prefer explicit programmatic configuration or need custom request parameters, subclass `DeepEvalBaseLLM`:

```python
import os
from openai import OpenAI
from deepeval.models import DeepEvalBaseLLM

class OpenRouterLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        self.model_name = model_name

    def load_model(self):
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
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
```

---

## Running LLM Evaluations

DeepEval supports evaluating `LLMTestCase` objects using a wide variety of metrics, such as `ExactMatchMetric`, `AnswerRelevancyMetric`, `FaithfulnessMetric`, and `GEval`.

Run the evaluation script provided in this repository:

```bash
python3 run_eval.py
```

---

## Integrating Confident AI Cloud

To view evaluation runs, test cases, metrics, and regressions in the Confident AI Cloud dashboard:

1. **Obtain Confident AI API Key**: Get your API key from the Confident AI dashboard.
2. **Local Execution**: Set `CONFIDENT_API_KEY` in your `.env` file or environment. `run_eval.py` will automatically authenticate using `deepeval.login(CONFIDENT_API_KEY)` and upload test runs to Confident AI Cloud upon completion.
3. **GitHub Actions Integration**: Set `CONFIDENT_API_KEY` as a repository secret in GitHub Settings (`Settings -> Secrets and variables -> Actions`). The `.github/workflows/deepeval.yml` pipeline will pass `CONFIDENT_API_KEY` to the evaluation step and upload test results on every push, pull request, or workflow dispatch.
4. **Manual Workflow Execution & Downloadable Artifacts**:
   - You can manually trigger the evaluation workflow at any time via the GitHub repository's **Actions** tab by selecting **DeepEval Evaluation Pipeline** and clicking **Run workflow** (`workflow_dispatch`).
   - After execution finishes, navigate to the workflow run details page under **Artifacts** to download the generated `deepeval-html-reports` zip archive containing HTML report files (`reports/*.html`) and `deepeval_results.zip`.

---

## Demonstrating Error Handling (Intentionally Failing Test Case)

To demonstrate how DeepEval identifies and logs failing test cases, `run_eval.py` includes two test cases:

1. **Passing Test Case**:
   - **Input**: `"What is the capital of France?"`
   - **Actual Output**: `"Paris"`
   - **Expected Output**: `"Paris"`
   - **Status**: `PASSED` (Score: 1.00)

2. **Intentionally Failing Test Case**:
   - **Input**: `"What is the capital of Germany?"`
   - **Actual Output**: `"Munich"` *(Intentionally incorrect)*
   - **Expected Output**: `"Berlin"`
   - **Status**: `FAILED` (Score: 0.00)

### How DeepEval Handles Failures

When a metric threshold is not met:
- DeepEval logs a detailed CLI breakdown showing input, actual output, expected output, metric score, threshold, and reason for failure.
- Detailed metrics and reasoning are compiled into the HTML dashboard for debugging.
- The overall pass rate is computed (e.g., 50.0% Pass Rate).

---

## Saving and Zipping HTML Results

DeepEval generates visual HTML dashboards summarizing test runs.

In `run_eval.py`, saving to HTML and archiving into a ZIP file is configured using `DisplayConfig`:

```python
import os
import zipfile
from deepeval import evaluate
from deepeval.evaluate.configs import DisplayConfig

# Configure display options to generate HTML output in 'reports' directory
display_config = DisplayConfig(file_type="html", file_output_dir="reports")

# Execute evaluation
res = evaluate(test_cases, metrics=[metric], display_config=display_config)

# Find generated HTML file(s) and create zip archive
html_files = [os.path.join("reports", f) for f in os.listdir("reports") if f.endswith(".html")]

zip_path = "deepeval_results.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for file in html_files:
        zipf.write(file, arcname=os.path.basename(file))
```

---

## Example Execution & Output

Executing `python3 run_eval.py` produces terminal output similar to:

```text
Initializing DeepEval test suite with OpenRouter configuration...

Running evaluation on test cases...

╭──────────────────────────────────────────────────────────────────────────────╮
│ 🚀 DeepEval Evaluation Results                                               │
╰──────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────╮
│ ✅ test_case_0 (Passed 1 metrics)                                            │
╰──────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────╮
│                                                                              │
│  ❌ test_case_1                                                              │
│  ├──   Input:              What is the capital of Germany?                   │
│  │     Actual Output:      Munich                                            │
│  │     Expected Output:    Berlin                                            │
│  └── Metrics                                                                 │
│       Status ┃ Metric      ┃ Score ┃ Threshold ┃ Reason                      │
│      ━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│        FAIL  │ Exact Match │ 0.00  │ 1.00      │ The actual and expected     │
│              │             │       │           │ outputs are different.      │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────╮
│ Aggregate Metrics                                                            │
│                                                                              │
│  Metric        ┃ Average Score   ┃ Pass Rate                        ┃ Total  │
│ ━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━ │
│  Exact Match   │ 0.50            │ 50.00% | passed=1 | failed=1     │ 2      │
╰──────────────────────────────────────────────────────────────────────────────╯

✅ HTML Dashboard saved to: reports/evaluation_20261014_102221.html

Generated HTML Report(s): ['reports/evaluation_20261014_102221.html']

Successfully created compressed results file: /app/deepeval_results.zip
```

---

## Project Structure

```text
deepeval-demo/
├── .env                  # Environment variables (OpenRouter API key)
├── requirements.txt      # Project dependencies
├── run_eval.py           # Evaluation script (OpenRouter setup, tests, zip export)
├── deepeval_results.zip  # Compressed zip archive containing HTML report
├── reports/              # Generated HTML reports folder
└── README.md             # Project documentation and setup guide
```
