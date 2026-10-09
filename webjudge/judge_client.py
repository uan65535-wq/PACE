"""Client and rate-limit settings for the WebJudge judge model.

The judge model is Qwen/Qwen3-VL-32B-Instruct behind an OpenAI-compatible endpoint.
Set the API key in the environment before running the evaluation:

    export JUDGE_API_KEY="..."            # bash
    $env:JUDGE_API_KEY = "..."            # PowerShell

JUDGE_BASE_URL can override the endpoint (default: SiliconFlow, the one used in the paper).
"""
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("JUDGE_API_KEY"),
    base_url=os.environ.get("JUDGE_BASE_URL", "https://api.siliconflow.cn/v1"),
)
MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"

# Tokens-per-minute budget of the judge account and the fraction of it the limiter uses.
TPM_LIMIT = 40000
TPM_SAFETY_MARGIN = 0.75
