# AI Product Roadmap: DeepEval & OpenRouter Pipeline

## Executive Summary
This roadmap establishes the strategic direction for the **DeepEval & OpenRouter LLM Evaluation Pipeline**. As an enterprise AI testing platform, the product currently provides custom OpenRouter model integration, exact-match test suites, HTML report compilation, and GitHub Actions CI automation.

To evolve from a static demonstration harness into an enterprise-grade AI quality assurance (AI QA) platform, this roadmap defines prioritized, actionable initiatives across data ingestion, advanced evaluation metrics, automated regression detection, CI/CD developer workflows, and model benchmarking.

---

## Current State Analysis

### Strengths
- **OpenRouter Model Interoperability**: Flexible custom wrapper (`OpenRouterLLM`) connecting to hundreds of LLMs through a unified API endpoint.
- **Fail-Safe Cloud Authentication**: Resilient fallback architecture that gracefully degrades to local execution if Confident AI Cloud credentials are missing or invalid.
- **Automated Reporting Pipeline**: Automated HTML report generation and ZIP bundling in both local runs and GitHub Actions CI.

### Gaps & Opportunities
- **Static Hardcoded Test Cases**: Test cases are currently hardcoded in Python; lack of support for external datasets (JSONL, CSV, Hugging Face).
- **Single Metric Scope**: Assertions currently rely primarily on `ExactMatchMetric`, which is ill-suited for generative, open-ended, and non-deterministic LLM tasks.
- **No CI Quality Gates**: CI workflow executes evaluations and archives reports, but does not block pull requests when regression thresholds fail.
- **No Performance / Token Cost Telemetry**: Absence of latency, prompt tokens, completion tokens, and cost tracking across model evaluations.
- **Lack of Multi-Model Comparison**: Evaluations run on a single model at a time without side-by-side benchmarking matrices.

---

## Roadmap Overview & Prioritization Matrix

Prioritization is calculated using Impact vs. Feasibility:
- **Phase 1: Near-Term (Core Reliability & Quality Gates)** - High Impact / High Feasibility
- **Phase 2: Mid-Term (Advanced Metrics & Dataset Expansion)** - High Impact / Medium Feasibility
- **Phase 3: Long-Term (Benchmarking & Continuous Observability)** - Very High Impact / Lower Feasibility

| Item ID | Feature / Initiative | Priority | Dev Effort (SP / Hrs) | Test Effort (SP / Hrs) | Target Phase |
|:---|:---|:---:|:---:|:---:|:---:|
| **FEAT-101** | External Dataset Loader (JSONL / CSV / Parquet) | P0 (Must Have) | 8 SP (24h) | 5 SP (12h) | Phase 1 |
| **FEAT-102** | Configurable CI/CD Quality Gates & Threshold Enforcer | P0 (Must Have) | 5 SP (16h) | 3 SP (8h) | Phase 1 |
| **FEAT-103** | Cost & Latency Performance Telemetry Tracking | P1 (Should Have) | 5 SP (16h) | 3 SP (8h) | Phase 1 |
| **FEAT-201** | Multi-Metric Suite Integration (Faithfulness, Relevancy, Hallucination) | P0 (Must Have) | 8 SP (24h) | 5 SP (16h) | Phase 2 |
| **FEAT-202** | Automated PR Commenting Bot with Evaluation Diffs | P1 (Should Have) | 5 SP (16h) | 3 SP (8h) | Phase 2 |
| **FEAT-203** | CLI Test Suite Runner & Dynamic Argument Parser | P2 (Nice to Have) | 3 SP (10h) | 2 SP (6h) | Phase 2 |
| **FEAT-301** | Multi-Model Side-by-Side Comparison Matrix | P1 (Should Have) | 13 SP (40h) | 8 SP (20h) | Phase 3 |
| **FEAT-302** | Synthetic Evaluation Data Generator using LLMs | P2 (Nice to Have) | 8 SP (24h) | 5 SP (16h) | Phase 3 |

*(Note: 1 Story Point [SP] $\approx$ 3-4 engineering hours)*

---

## Detailed Feature Specifications

### Phase 1: Near-Term (Core Reliability & Quality Gates)

#### FEAT-101: External Dataset Loader (JSONL / CSV / Parquet)
- **Description**: Decouple test cases from Python code by implementing a robust dataset loader module (`loaders/dataset_loader.py`). Supports loading standard benchmark formats (`.jsonl`, `.csv`, `.parquet`) containing fields `input`, `expected_output`, `context`, and metadata tags.
- **User Problem Solved**: Eliminates developer dependency when QA or domain experts want to add or update hundreds of evaluation test prompts.
- **Technical Scope**:
  - Implement streaming and batched ingestion via `pandas` or `pyarrow`.
  - Validate dataset schema using `pydantic`.
  - Add sample datasets in `datasets/golden_evaluation_set.jsonl`.
- **Estimated Development Effort**: 8 Story Points (24 hours).
- **Estimated Testing Effort**: 5 Story Points (12 hours).

#### FEAT-102: Configurable CI/CD Quality Gates & Threshold Enforcer
- **Description**: Build an automated pass/fail enforcement mechanism that evaluates overall pass rates and metric-specific thresholds, exiting with non-zero exit codes if thresholds are breached.
- **User Problem Solved**: Currently, the CI pipeline uploads artifacts even if test cases fail. This feature empowers engineering teams to block merging PRs that introduce prompt regressions.
- **Technical Scope**:
  - Support command-line and environment thresholds: `--min-pass-rate 0.85`.
  - Add summary output in GitHub Actions step summary (`$GITHUB_STEP_SUMMARY`).
  - Configure PR status checks.
- **Estimated Development Effort**: 5 Story Points (16 hours).
- **Estimated Testing Effort**: 3 Story Points (8 hours).

#### FEAT-103: Cost & Latency Performance Telemetry Tracking
- **Description**: Add real-time telemetry extraction to `OpenRouterLLM` wrapper to record generation latency (p50, p95, p99), token usage (prompt, completion, total), and estimated dollar cost per test run based on OpenRouter pricing headers.
- **User Problem Solved**: Teams often optimize for accuracy without realizing their prompt modifications double token consumption and latency.
- **Technical Scope**:
  - Capture OpenRouter completion headers (`usage.prompt_tokens`, `usage.completion_tokens`).
  - Log execution time delta using `time.perf_counter()`.
  - Include aggregate cost and latency tables in the HTML report and CLI output.
- **Estimated Development Effort**: 5 Story Points (16 hours).
- **Estimated Testing Effort**: 3 Story Points (8 hours).

---

### Phase 2: Mid-Term (Advanced Metrics & Developer Workflows)

#### FEAT-201: Multi-Metric Suite Integration (Faithfulness, Relevancy, Hallucination)
- **Description**: Expand beyond `ExactMatchMetric` to integrate DeepEval's cognitive and LLM-as-a-judge metrics: `AnswerRelevancyMetric`, `FaithfulnessMetric`, `HallucinationMetric`, and `GEval`.
- **User Problem Solved**: Exact match is inadequate for complex generative tasks, summarization, and RAG pipelines where phrasing varies while meaning remains accurate.
- **Technical Scope**:
  - Configure `OpenRouterLLM` as the judging LLM across custom metrics.
  - Add configurable metric thresholds in `eval_config.yaml`.
  - Create specialized test scenarios for Retrieval-Augmented Generation (RAG).
- **Estimated Development Effort**: 8 Story Points (24 hours).
- **Estimated Testing Effort**: 5 Story Points (16 hours).

#### FEAT-202: Automated PR Commenting Bot with Evaluation Diffs
- **Description**: Integrate a GitHub Actions bot that compares the evaluation score of the PR branch against the target `main` branch and publishes a markdown diff summary as a pull request comment.
- **User Problem Solved**: Developers and reviewers can immediately see regression diffs without manually downloading and opening ZIP reports.
- **Technical Scope**:
  - Fetch benchmark results artifact from the latest `main` workflow run.
  - Calculate delta score ($\Delta$ pass rate, regression/improvement count).
  - Use GitHub Actions API (`actions/github-script`) to post or update the PR comment.
- **Estimated Development Effort**: 5 Story Points (16 hours).
- **Estimated Testing Effort**: 3 Story Points (8 hours).

#### FEAT-203: CLI Test Suite Runner & Dynamic Argument Parser
- **Description**: Replace hardcoded execution in `run_eval.py` with an extensible CLI (`eval-cli`) using `argparse` or `typer`.
- **User Problem Solved**: Enables developers to run focused subsets of evaluations locally (e.g., `--category edge_cases`, `--model anthropic/claude-3.5-sonnet`, `--metrics exact_match,relevancy`).
- **Technical Scope**:
  - Implement arguments: `--model`, `--dataset`, `--metrics`, `--tags`, `--output-dir`.
  - Add dry-run mode and verbose debug logging.
- **Estimated Development Effort**: 3 Story Points (10 hours).
- **Estimated Testing Effort**: 2 Story Points (6 hours).

---

### Phase 3: Long-Term (Benchmarking & Continuous Observability)

#### FEAT-301: Multi-Model Side-by-Side Comparison Matrix
- **Description**: Execute test suites concurrently across a matrix of multiple LLM providers (e.g., `openai/gpt-4o-mini`, `anthropic/claude-3.5-haiku`, `meta-llama/llama-3.3-70b-instruct`) and generate a comparative leaderboard.
- **User Problem Solved**: Empowers product teams to make data-driven decisions on model selection, balancing price, speed, and accuracy tradeoffs.
- **Technical Scope**:
  - Parallel asynchronous request execution using `asyncio` and `OpenRouterLLM.a_generate()`.
  - Generate consolidated comparison matrix (CSV/HTML) with radar charts of accuracy vs. cost vs. latency.
- **Estimated Development Effort**: 13 Story Points (40 hours).
- **Estimated Testing Effort**: 8 Story Points (20 hours).

#### FEAT-302: Synthetic Evaluation Data Generator using LLMs
- **Description**: Provide automated tooling to generate high-quality synthetic test cases, edge cases, and adversarial perturbations (misspellings, injections, jailbreak attempts) from reference seed documents.
- **User Problem Solved**: Addresses the cold-start problem of dataset curation for emerging AI products.
- **Technical Scope**:
  - Implement generator pipeline using DeepEval's `Synthesizer`.
  - Filter low-quality synthetic generations through rule-based heuristics.
- **Estimated Development Effort**: 8 Story Points (24 hours).
- **Estimated Testing Effort**: 5 Story Points (16 hours).

---

## Resource & Capacity Allocation

```
Total Engineering Effort:  55 Story Points (~170 Hours)
Total QA & Testing Effort: 34 Story Points (~106 Hours)
Overall Project Scope:     89 Story Points (~276 Hours)
```

### Milestone Schedule
- **Sprint 1 - 2 (Weeks 1-4)**: Deliver Phase 1 (FEAT-101, FEAT-102, FEAT-103) -> Production-ready quality gates in CI.
- **Sprint 3 - 4 (Weeks 5-8)**: Deliver Phase 2 (FEAT-201, FEAT-202, FEAT-203) -> Semantic evaluations and PR bot integration.
- **Sprint 5 - 6 (Weeks 9-12)**: Deliver Phase 3 (FEAT-301, FEAT-302) -> Multi-model benchmarking matrix and synthetic data generation.

---

## Risk Management & Mitigation

| Risk | Impact | Probability | Mitigation Strategy |
|:---|:---:|:---:|:---|
| **OpenRouter API Rate Limits & Throttling** | High | Medium | Implement exponential backoff retry logic and concurrent request rate limiting in `OpenRouterLLM`. |
| **LLM-as-a-Judge Evaluation Flakiness** | Medium | High | Fix evaluation temperature to `0.0`, specify structured few-shot evaluation prompts, and average judge results across repeated runs. |
| **High API Cost during CI Matrix Runs** | High | Medium | Enforce small golden dataset subsets (max 20 samples) on PRs; run full benchmark suites nightly on scheduled triggers. |
| **Confident AI Cloud Downtime** | Low | Low | Maintained existing automatic local fallback mechanism to guarantee test execution never fails due to cloud unavailability. |

---

## Success Metrics & KPIs
1. **Developer Adoption**: 100% of LLM application pull requests verified by automated evaluation quality gates.
2. **Defect Detection**: Minimum 40% increase in pre-deployment detection of prompt regressions and hallucination errors.
3. **Cost Efficiency**: 20-30% reduction in inference costs achieved by identifying equivalent performance from lower-cost models via FEAT-301 matrix comparisons.
4. **Pipeline Execution Time**: PR test suite execution completed in under 3 minutes for PR gates.
