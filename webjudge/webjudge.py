"""WebJudge automatic evaluation for Online-Mind2Web.

Port of src/methods/webjudge_online_mind2web.py from OSU-NLP-Group/Online-Mind2Web
(Xue et al., "An Illusion of Progress? Assessing the Current State of Web Agents", COLM 2025),
Copyright (c) 2025 OSU Natural Language Processing, MIT License.
The three system prompts (key-point extraction, per-screenshot scoring, final judgement)
are copied verbatim from the original. Two adaptations, neither of which changes the method:
1. Judge model: Qwen/Qwen3-VL-32B-Instruct (configured in webjudge/judge_client.py) instead of
   o4-mini, so absolute success rates are comparable between our two arms but not with
   published leaderboard numbers.
2. Execution: tasks and screenshots are scored sequentially with a token-rate limiter and
   retries instead of the original asyncio/multiprocess execution.
"""

import base64
import re
import time
from pathlib import Path

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from webjudge.judge_client import MODEL_NAME, TPM_LIMIT, TPM_SAFETY_MARGIN, client

MAX_IMAGE = 50  # 跟官方一致：最终判决prompt里最多带多少张"关键截图"

_token_usage_log: list[tuple[float, int]] = []


def _estimate_tokens(messages: list[dict]) -> int:
    """跟mind2web_agent_raps.py里VisionAgent._estimate_tokens()同一套
    粗略估算规则（文本2字符/token、每张图固定1000token占位值），独立
    维护一份而不是导入那个类方法，是因为它是绑定在VisionAgent实例上的
    方法，为了单独调它去实例化一个VisionAgent反而要多承担agent那边
    __init__可能有的副作用，不值得。
    """
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content) // 2
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    total += len(part.get("text", "")) // 2
                elif part.get("type") == "image_url":
                    total += 1000
    return total


def _wait_for_token_budget(estimated_tokens: int) -> None:
    """跟mind2web_agent_raps.py里_wait_for_token_budget()逻辑完全一致，
    但用的是这个模块自己的_token_usage_log——WebJudge和两个agent脚本
    从来不会在同一个进程里跑（评测总是等agent批次跑完之后单独执行），
    共不共享这个滑动窗口列表不影响正确性，分开维护只是让这个模块可以
    独立导入/测试，不依赖import时agent模块内部的可变全局状态。
    """
    now = time.monotonic()
    while _token_usage_log and now - _token_usage_log[0][0] > 60:
        _token_usage_log.pop(0)
    used = sum(t for _, t in _token_usage_log)
    budget = TPM_LIMIT * TPM_SAFETY_MARGIN
    if used + estimated_tokens > budget:
        if _token_usage_log:
            wait_seconds = 60 - (now - _token_usage_log[0][0]) + 0.5
            if wait_seconds > 0:
                print(f"  ⏳ WebJudge预计本次调用约{estimated_tokens}token，"
                      f"过去60秒已用约{used}token，等待{wait_seconds:.1f}秒...")
                time.sleep(wait_seconds)


def _call_llm_with_retry(messages: list[dict], max_retries: int = 5) -> str:
    """跟mind2web_agent_raps.py里VisionAgent._call_llm_with_retry()同一套
    重试语义（只重试429/5xx/网络层问题，5次退避，5s起倍增封顶60s）。
    """
    estimated_tokens = _estimate_tokens(messages)
    _wait_for_token_budget(estimated_tokens)
    delay = 5.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0,
                max_tokens=1024,
            )
            usage = getattr(response, "usage", None)
            tokens_for_rate_limit = (
                usage.total_tokens
                if usage is not None and getattr(usage, "total_tokens", None)
                else estimated_tokens
            )
            _token_usage_log.append((time.monotonic(), tokens_for_rate_limit))
            return response.choices[0].message.content.strip()
        except (RateLimitError, InternalServerError,
                APIConnectionError, APITimeoutError) as e:
            if attempt == max_retries - 1:
                raise
            print(f"  ⏳ WebJudge触发可重试错误（{type(e).__name__}），"
                  f"{delay:.0f}秒后重试（第{attempt + 1}/{max_retries}次）...")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("重试循环异常退出，不应该走到这里")


def _encode_image_b64(image_path: Path) -> str:
    """直接读取截图文件原始字节做base64。官方encode_image()会用PIL把
    图片重新转码成JPEG——那是因为官方代码可能收到任意来源/格式的输入
    图片；咱们这边截图统一是run_task()自己截出来的PNG，格式已知固定，
    没必要多引入一个PIL依赖/多一次转码开销。
    """
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


# ============================================================
# 以下三个函数的system prompt原文逐字保留自官方
# src/methods/webjudge_online_mind2web.py，只把原来的async/
# asyncio.gather改成顺序调用、模型调用换成上面的_call_llm_with_retry。
# ============================================================

def identify_key_points(task: str) -> str:
    """第一步：从任务描述里拆出几条明确写在任务文字里的"关键点"（不是
    推断出来的隐含要求）。system prompt跟官方一字不差。
    """
    system_msg = """You are an expert tasked with analyzing a given task to identify the key points explicitly stated in the task description.

**Objective**: Carefully analyze the task description and extract the critical elements explicitly mentioned in the task for achieving its goal.

**Instructions**:
1. Read the task description carefully.
2. Identify and extract **key points** directly stated in the task description.
   - A **key point** is a critical element, condition, or step explicitly mentioned in the task description.
   - Do not infer or add any unstated elements.
   - Words such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" must go through the sort function(e.g., the key point should be "Filter by highest").

**Respond with**:
- **Key Points**: A numbered list of the explicit key points for completing this task, one per line, without explanations or additional details."""
    text = f"Task: {task}"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    return _call_llm_with_retry(messages)


def judge_image(task: str, image_path: Path, key_points: str) -> dict:
    """第二步：给单张截图打1-5分，判断它是不是完成任务所需的关键证据。
    system prompt跟官方一字不差。返回{"response": 原始回复,
    "score": 解析出的整数分（解析失败记0）}。
    """
    system_msg = """You are an expert evaluator tasked with determining whether an image contains information about the necessary steps to complete a task.

**Objective**: Analyze the provided image and decide if it shows essential steps or evidence required for completing the task. Use your reasoning to explain your decision before assigning a score.

**Instructions**:
1. Provide a detailed description of the image, including its contents, visible elements, text (if any), and any notable features.

2. Carefully examine the image and evaluate whether it contains necessary steps or evidence crucial to task completion:
- Identify key points that could be relevant to task completion, such as actions, progress indicators, tool usage, applied filters, or step-by-step instructions.
- Does the image show actions, progress indicators, or critical information directly related to completing the task?
- Is this information indispensable for understanding or ensuring task success?
- If the image contains partial but relevant information, consider its usefulness rather than dismissing it outright.

3. Provide your response in the following format:
- **Reasoning**: Explain your thought process and observations. Mention specific elements in the image that indicate necessary steps, evidence, or lack thereof.
- **Score**: Assign a score based on the reasoning, using the following scale:
    - **1**: The image does not contain any necessary steps or relevant information.
    - **2**: The image contains minimal or ambiguous information, unlikely to be essential.
    - **3**: The image includes some relevant steps or hints but lacks clarity or completeness.
    - **4**: The image contains important steps or evidence that are highly relevant but not fully comprehensive.
    - **5**: The image clearly displays necessary steps or evidence crucial for completing the task.

Respond with:
1. **Reasoning**: [Your explanation]
2. **Score**: [1-5]"""
    text = f"""**Task**: {task}

**Key Points for Task Completion**: {key_points}

The snapshot of the web page is shown in the image."""

    jpg_base64_str = _encode_image_b64(image_path)
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{jpg_base64_str}"}},
            ],
        },
    ]
    response = _call_llm_with_retry(messages)
    return {"response": response}


def evaluate_task(task_description: str, action_history: list[str],
                   screenshot_paths: list[Path], score_threshold: int = 3) -> dict:
    """完整跑一遍WebJudge三段式评测，返回：
    {
        "predicted_label": 1 或 0（1=success，跟官方extract_predication同一套
                             "response按status:切开、看有没有success"解析规则）,
        "response": 最终判决LLM的原始回复文本（含Thoughts:和Status:两行）,
        "key_points": 第一步拆出来的关键点文本,
        "image_judge_record": 每张截图的打分记录（跟官方auto_eval()写进
                               result.json的image_judge_record字段同结构）,
    }
    """
    system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's action history, key points for task completion, some potentially important web pages in the agent's trajectory and their reasons, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!
*Important Evaluation Criteria*:
1: The filtered results must be displayed correctly. If filters were not properly applied (i.e., missing selection, missing confirmation, or no visible effect in results), the task is not considered successful.
2: You must carefully check whether these snapshots and action history meet these key points. Ensure that specific filter conditions, such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" are correctly applied using the filter function(e.g., sort function).
3: Certain key points or requirements should be applied by the filter. Otherwise, a search with all requirements as input will be deemed a failure since it cannot guarantee that all results meet the requirements!
4: If the task requires filtering by a specific range of money, years, or the number of beds and bathrooms, the applied filter must exactly match the given requirement. Any deviation results in failure. To ensure the task is successful, the applied filter must precisely match the specified range without being too broad or too narrow.
Examples of Failure Cases:
- If the requirement is less than $50, but the applied filter is less than $25, it is a failure.
- If the requirement is $1500-$2500, but the applied filter is $2000-$2500, it is a failure.
- If the requirement is $25-$200, but the applied filter is $0-$200, it is a failure.
- If the required years are 2004-2012, but the filter applied is 2001-2012, it is a failure.
- If the required years are before 2015, but the applied filter is 2000-2014, it is a failure.
- If the task requires exactly 2 beds, but the filter applied is 2+ beds, it is a failure.
5: Some tasks require a submission action or a display of results to be considered successful.
6: If the retrieved information is invalid or empty(e.g., No match was found), but the agent has correctly performed the required action, it should still be considered successful.
7: If the current page already displays all available items, then applying a filter is not necessary. As long as the agent selects items that meet the requirements (e.g., the cheapest or lowest price), the task is still considered successful.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process based on double-checking each key points and the evaluation criteria>
Status: "success" or "failure"
"""

    key_points_raw = identify_key_points(task_description)
    key_points = key_points_raw.replace("\n\n", "\n")
    try:
        key_points = key_points.split("**Key Points**:")[1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())
    except IndexError:
        key_points = key_points.split("Key Points:")[-1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())

    image_judge_record = []
    kept_images_b64 = []
    kept_thoughts = []
    score_pattern = r"[1-5]"
    for image_path in screenshot_paths:
        judged = judge_image(task_description, image_path, key_points)
        response = judged["response"]
        try:
            score_text = response.split("Score")[1]
            thought = (
                response.split("**Reasoning**:")[-1]
                .strip().lstrip("\n").split("\n\n")[0].replace("\n", " ")
            )
            score = int(re.findall(score_pattern, score_text)[0])
        except Exception:
            score = 0
            thought = ""
        image_judge_record.append({"Response": response, "Score": score})

        if score >= score_threshold:
            kept_images_b64.append(_encode_image_b64(image_path))
            if thought:
                kept_thoughts.append(thought)

    kept_images_b64 = kept_images_b64[:MAX_IMAGE]
    kept_thoughts = kept_thoughts[:MAX_IMAGE]

    actions_text = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(action_history))
    if kept_images_b64:
        prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}

The potentially important snapshots of the webpage in the agent's trajectory and their reasons:
{thoughts}"""
        text = prompt.format(
            task=task_description, key_points=key_points,
            last_actions=actions_text,
            thoughts="\n".join(f"{i + 1}. {t}" for i, t in enumerate(kept_thoughts)),
        )
    else:
        # 一张截图都没达到分数阈值——官方代码这种情况下不带图，只靠任务
        # 描述+关键点+动作历史做最终判决
        prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}"""
        text = prompt.format(task=task_description, key_points=key_points, last_actions=actions_text)

    content = [{"type": "text", "text": text}]
    for b64 in kept_images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": content},
    ]
    final_response = _call_llm_with_retry(messages)

    try:
        predicted_label = 1 if "success" in final_response.lower().split("status:")[1] else 0
    except IndexError:
        predicted_label = 0

    return {
        "predicted_label": predicted_label,
        "response": final_response,
        "key_points": key_points,
        "image_judge_record": image_judge_record,
    }
