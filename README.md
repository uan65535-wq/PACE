# PACE: Perception-Amortized Chunked Execution

Code for the paper **"PACE: Perception-Amortized Chunked Execution for Reducing Redundant Perception in GUI Agents"** (under review).

GUI agents usually take a fresh screenshot and make a full model call before every action. PACE lets one model call
plan a short chunk of actions and executes each follow-up action only after a zero-model-cost check confirms that the
page state it depends on (active tab, URL, scroll position, the element under its target point, and any entered text)
is unchanged. If a check fails, the rest of the chunk is discarded and the agent observes again, so the worst case is
the always-observe agent it is built on.

## Repository layout

| File | Role | Paper |
|---|---|---|
| `mind2web_agent_deepseek.py` | Always-observe baseline agent (one screenshot and one model call per action) | Sec. 5.1 |
| `mind2web_agent_chunk_deepseek.py` | **PACE** = the baseline plus chunked execution and pre-execution verification | Sec. 3–4 |
| `run_batch_baseline_deepseek_paired.py` | Batch runner for the baseline in the paired experiment | Sec. 5.1 |
| `run_batch_chunk_deepseek.py` | Batch runner for PACE | Sec. 5.1 |
| `run_batch_mind2web_deepseek.py` | Generic batch runner for the baseline agent (used by the paired runner) | — |
| `mind2web_agent_deepseek_llmlingua.py`, `mind2web_agent_chunk_deepseek_llmlingua.py` | Baseline and PACE with LLMLingua-2 compression of the action history | Sec. 5.6 |
| `run_batch_deepseek_llmlingua.py`, `run_batch_chunk_deepseek_llmlingua.py` | Batch runners for the two compressed arms | Sec. 5.6 |
| `deepseek_encoding.py` | DeepSeek-V4.1 prompt encoder, used only to count text tokens (copied unchanged from the model's Hugging Face repository, MIT License) | Sec. 5.1 |
| `webjudge/` | WebJudge evaluator ported from Online-Mind2Web (MIT License) and its judge-model client | Sec. 5.1–5.2 |
| `run_webjudge.py`, `tools/paired_webjudge.py` | Run WebJudge on one arm / on both paired arms | Sec. 5.2 |
| `tools/queue_paired.json` | Task order of the paired experiment (the paper uses the first 100 tasks) | Sec. 5.1 |
| `tools/paper_numbers.py`, `tools/lingua_numbers.py` | Recompute every number in Sections 5.1–5.5 and 5.6 from the run logs | Sec. 5 |
| `tools/test_chunk_guard.py` | Offline tests of the verification checks in a local browser (no API calls) | Sec. 4.4 |

Code comments are partly in Chinese and record the development history.

## Setup

Python 3.10+ (tested with 3.13).

```bash
pip install -r requirements.txt
playwright install chromium
```

The agents call `deepseek/deepseek-v4.1-flash` through an OpenAI-compatible endpoint (configured as `client` near the
top of each agent file; the experiments used `https://api.qnaigc.com/v1`). WebJudge calls `Qwen/Qwen3-VL-32B-Instruct`
(configured in `webjudge/judge_client.py`). API keys are read from the environment:

```bash
export QINIU_API_KEY="..."   # agent model
export JUDGE_API_KEY="..."   # judge model (optionally JUDGE_BASE_URL)
```

On first use the code downloads the Online-Mind2Web task list (`osunlp/Online-Mind2Web`, CC BY 4.0), the
DeepSeek-V4.1-Flash tokenizer (for text-token counting) and, for the compression arms, the LLMLingua-2 model
`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` from Hugging Face.

## Running the paired experiment

Tasks are run in batches of five in the order of `tools/queue_paired.json`: each batch is run first with PACE and then
immediately with the baseline, so that both arms see the websites on the same day.

```bash
python run_batch_chunk_deepseek.py --task-ids <id1>,<id2>,<id3>,<id4>,<id5>
python run_batch_baseline_deepseek_paired.py --task-ids <id1>,<id2>,<id3>,<id4>,<id5>
```

Results are written to `runs/chunk_deepseek_v3/` and `runs/baseline_deepseek_paired/` (one `result.json` and the
screenshots per task, plus per-call token logs in `logs/token_calibration.jsonl`). Runs resume by default: a task
that already has a `result.json` is skipped. The compression arms are run the same way with
`run_batch_chunk_deepseek_llmlingua.py` and `run_batch_deepseek_llmlingua.py`.

Evaluate task success and recompute the paper's tables:

```bash
python tools/paired_webjudge.py
python tools/paper_numbers.py
python tools/lingua_numbers.py
```

The exclusion rules (environment errors, access denied or blank page within five steps, and two manually excluded
tasks with asymmetric access) are implemented in `tools/paper_numbers.py` and described in Section 5.1.

## Data

The run logs, trajectories and WebJudge results behind the paper are being organized and will be released with the
published article.

## License

MIT License (see `LICENSE`). `deepseek_encoding.py` (DeepSeek, MIT License) and the WebJudge prompts and algorithm in
`webjudge/webjudge.py` (OSU Natural Language Processing, MIT License) keep their original licenses.
