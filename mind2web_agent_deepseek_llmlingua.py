# ==============================================================================
# 【实验分支 / NOT OFFICIAL】baseline + LLMLingua-2（DeepSeek线）（2026-09-27，由tools/make_lingua_deepseek.py生成，不要手改）
# 源文件：mind2web_agent_deepseek.py（原banner保留在下面）。唯一改动：动作历史每条在追加时用LLMLingua-2压缩一次
#   （"LLMLingua-2压缩动作历史"一节，baseline版和chunk版逐字相同，取自Qwen线mind2web_agent_llmlingua.py），
#   输出目录runs/baseline_deepseek_llmlingua/。用于验证叠加：chunk+LLMLingua-2 vs baseline+LLMLingua-2（100对，任务同主实验）。
# 配套批跑脚本：run_batch_deepseek_llmlingua.py
# ==============================================================================
# ==============================================================================
# 【正式版 / OFFICIAL】baseline —— deepseek/七牛云线（reorder版，2026-09-18确认）
# 当前用于baseline-vs-raps正式对比，改动前先复制一份新文件再改，不要直接改这个文件。
# 配套批跑脚本：run_batch_mind2web_deepseek.py | 状态见 docs/HANDOFF.md 最新entry
# ==============================================================================
"""Online-Mind2Web 单任务执行脚本（视觉模型 + 坐标点击）。

跟baseline.py（WebArena-Verified）的关键区别：
1. 没有BrowserGym/gym.make()这层——Online-Mind2Web官方仓库只提供WebJudge评测器，
   不提供跑agent的执行框架，这部分需要自己直接操作Playwright。
2. 动作空间从"AXTree文本 + click(bid)"换成"截图 + click(x, y)"，
   模型也必须换成视觉模型（纯文本模型看不到截图，没法选坐标）。
3. 产出格式必须符合Online-Mind2Web的v2 submission schema，才能喂给WebJudge打分：
   https://github.com/OSU-NLP-Group/Online-Mind2Web/blob/main/data/schema_v2/README.md
   每个任务一个目录：result.json + trajectory/0000.png, 0001.png, ...

本文件是mind2web_agent.py的实验性分支（换模型/换平台消融）：模型从SiliconFlow的
Qwen3-VL-32B-Instruct换成七牛云AI推理平台（https://api.qnaigc.com/v1）的
DeepSeek-V4.1-Flash（API model字符串"deepseek/deepseek-v4.1-flash"），其余逻辑
不变。因为是有风险的实验，特意复制成独立文件改，不动原本已跑出干净消融数据的
mind2web_agent.py。

API key、model字符串、base64图片支持这三项都已用真实API调用验证通过（2026-09-14）：
文本调用+视觉调用（真图片，模型正确读出了图中的边框颜色和文字）都成功，用量
统计里能看到`reasoning_tokens`字段——说明这个模型默认会输出推理过程，就算没有
显式要求"思考模式"，实际产生的completion_tokens会比纯文本模型预期的更多，
批量跑之前应该留意这一点对成本/时长的影响。raps那边（mind2web_agent_raps_
deepseek.py）已经真机跑通过Megabus（5步完成）和MTA（25步完成，其中第16/17/19
步是真正零截图的复用步）两个已知任务，感知/复用/TASK_COMPLETE验证/DOM-diff
校验/重复动作检测这些机制全部按预期触发，机制本身在新模型上运转正常——本文件
（baseline）没有raps那些机制，但共用的执行层（点击/输入/滚动/重试）跟raps
用的是同一套代码，可信度可以类推。TPM_LIMIT已按下面【TODO】说的调过。

【TODO - 需要你本地确认后再跑】：
- load_task()：Online-Mind2Web数据集（HuggingFace: osunlp/Online-Mind2Web）里
  每条任务记录的具体字段名，我这边没有下载到数据集本体核实，下面的load_task()
  只是占位实现，先用一个手写的示例任务跑通端到端流程。等你下载好数据集
  （pip install datasets; from datasets import load_dataset;
   load_dataset("osunlp/Online-Mind2Web")）之后，把字段名对上改掉load_task()即可。
- JUDGE当前先跳过，本脚本只负责跑出trajectory，WebJudge打分是下一步单独的脚本。
"""

import os
import re
import json
import time
import base64
from datetime import datetime
from pathlib import Path

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from playwright.sync_api import sync_playwright

# ===== 视觉token估算：用真实tokenizer反推图像部分token消耗 =====
# 七牛云的API同样不提供prompt_tokens的模态拆分，沿用原来对SiliconFlow验证过的
# 方法：用DeepSeek-V4.1-Flash的公开tokenizer精确算出messages里"去掉图片"那部分
# 文本真实会被tokenize成多少token，再拿真实返回的actual_prompt_tokens减去这个
# 数，反推出图像部分的token估算值。注意HF_TOKENIZER_NAME跟调API用的MODEL_NAME
# 是两个不同的字符串——七牛云的model参数是平台自定的"deepseek-flash"，不是
# HuggingFace仓库名，tokenizer要单独指定官方仓库deepseek-ai/DeepSeek-V4.1-Flash
# 才能下载到（这跟SiliconFlow那边MODEL_NAME恰好等于HF仓库名"Qwen/Qwen3-VL-32B-
# Instruct"的巧合不一样，换平台后不能再假设两者相同）。这套方法尚未针对DeepSeek
# 做过真机校准，第一次跑通后应该找一个已知任务核对反推出的image_tokens_estimated
# 是否落在合理量级（DeepSeek-V4.1-Flash的图像编码token开销公式跟Qwen-VL的
# smart_resize不同，不能直接套用之前~1000/张这个经验数字做sanity check）。
HF_TOKENIZER_NAME = "deepseek-ai/DeepSeek-V4.1-Flash"
_REAL_TOKENIZER = None


def _get_real_tokenizer():
    global _REAL_TOKENIZER
    if _REAL_TOKENIZER is None:
        from transformers import AutoTokenizer
        _REAL_TOKENIZER = AutoTokenizer.from_pretrained(HF_TOKENIZER_NAME)
    return _REAL_TOKENIZER


def _messages_text_only(messages: list[dict]) -> list[dict]:
    """把messages里的图片部分整个去掉，只留文本——用来精确tokenize
    "如果这次调用没有图片，文本部分会占多少token"，而不是简单地把
    image_url这个key删掉（chat template看到content是list时的渲染方式
    跟看到纯字符串时不一定一样，统一转成纯字符串content更贴近真实情况）。
    """
    result = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            text = "\n".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        else:
            text = content
        result.append({"role": m["role"], "content": text})
    return result


def _real_text_tokens(messages: list[dict]) -> int | None:
    """真实tokenizer精确算出messages去掉图片后的文本token数。
    任何环节失败（比如离线导致tokenizer加载不了）都返回None，
    不能让这个纯度量用途的计算拖垮正常的agent执行流程。

    DeepSeek-V4.1的tokenizer_config.json不带chat_template字段，标准的
    tokenizer.apply_chat_template()在这个模型上直接抛ValueError（已验证），
    改用DeepSeek官方提供的独立拼prompt模块deepseek_encoding.encode_messages()
    （逐字节vendor自HuggingFace deepseek-ai/DeepSeek-V4.1-Flash仓库，说明见
    该文件头部注释）。thinking_mode写死"chat"（非思考模式）只是这个度量用途
    下的一个近似——七牛云API服务端实际用的thinking_mode/reasoning_effort是
    多少我们看不到，这个反推法本来就是近似值，不影响它跟actual_prompt_tokens
    做减法这件事的可用性。
    """
    try:
        tokenizer = _get_real_tokenizer()
        text_only = _messages_text_only(messages)
        import deepseek_encoding
        templated = deepseek_encoding.encode_messages(text_only, thinking_mode="chat")
        return len(tokenizer.encode(templated))
    except Exception:
        return None

# ===== 配置七牛云AI推理平台的API（换模型实验，跟原SiliconFlow key完全独立）=====
# base_url/OpenAI SDK兼容性/model字符串已经查过官方文档+控制台核实：
# https://developer.qiniu.com/aitokenapi/13379/real-time-ai-interface-api
# QINIU_API_KEY已用真实key验证通过（2026-09-14，文本+视觉调用都成功，见对话
# 记录）。改成只认环境变量、不在源码里硬编码真实key——踩过一次坑：硬编码进去
# 之后，这个文件本身就变成"包含明文密钥的文件"，Claude Code的凭据泄露防护会
# 直接拦掉之后所有碰这个文件的Bash调用（包括无害的py_compile语法检查），很烦。
# 运行前设置环境变量：
#   PowerShell: $env:QINIU_API_KEY = "sk-...你的key..."
#   bash:       export QINIU_API_KEY="sk-...你的key..."
QINIU_API_KEY = os.environ.get("QINIU_API_KEY", "REPLACE_WITH_YOUR_QINIU_API_KEY")
client = OpenAI(
    api_key=QINIU_API_KEY,
    base_url="https://api.qnaigc.com/v1",
)
# DeepSeek-V4.1-Flash在七牛云平台的官方API model字符串。注意："AI大模型广场"
# 营销页（https://www.qiniu.com/ai/models）上写的"deepseek-flash"是别名/展示名，
# 不是真实可调用的model id——实测该字符串返回400 model_not_found。真实id是通过
# GET /v1/models接口核实的："deepseek/deepseek-v4.1-flash"，已用真实API调用
# （文本+视觉）验证可用。
MODEL_NAME = "deepseek/deepseek-v4.1-flash"

# ===== 真实费用计算：deepseek/deepseek-v4.1-flash在七牛云的官方单价 =====
# 价格来自模型广场页面（https://qiniu.com/ai/models）服务端渲染时嵌入的
# 真实计价JSON（pricing_rules_v2字段，2026-09-16抓取核实），单位：元/K
# tokens。七牛云对这个模型区分"缓存输入"和"非缓存输入"两档单价——命中
# prompt缓存的token按远低于原价计费，这是本文件要采集"命中缓存token数"
# 和"实际费用"这两个指标的前提（SiliconFlow上跑的Qwen3-VL-32B-Instruct
# 完全没有这个机制，价格页对应列全是"-"，验证过换平台前那边拿不到这个
# 数据）。
# 官方文档没有写明"高峰/空闲"具体是几点到几点，这里统一按高峰价计算
# （所有三项的空闲价都正好是高峰价的一半），也就是说下面算出来的
# estimated_cost_yuan_peak是一个保守估算，实际花费可能更低（最多能低到
# 这里数字的一半）。
PRICE_YUAN_PER_K = {
    "output": 0.008,           # 输出（高峰）；空闲价0.004
    "cached_input": 0.00004,   # 缓存输入（高峰）；空闲价0.00002
    "noncached_input": 0.002,  # 非缓存输入（高峰）；空闲价0.001
}

# 重要更正（2026-09-14）：一开始把TPM_LIMIT从40000调到150000，依据是网上查到
# 的"七牛云官方文档默认额度TPM 500万/RPM 500"，但拿这个key做真机探测后发现
# 完全不是这么回事——**真实瓶颈根本不是TPM，是RPM**，而且实测这个key的RPM
# 允许量远低于文档说的500：连续快速发8次请求，第6次就收到
# `429 rate_limit_rpm_exceeded`；就算把请求间隔拉到2秒（理论上相当于30
# RPM），20次请求里也有10次被限流，而且越到后面失败越密集。真实原因未知
# （这个key大概率是新用户/试用档位，跟文档写的默认额度不是一回事），但
# 结论很清楚：原来"只按token总量限速"的_wait_for_token_budget完全没有
# 覆盖到这个真实瓶颈——不管TPM_LIMIT设多少，只要请求发得太密集就会被限流。
# 加了MIN_REQUEST_INTERVAL_SECONDS做请求间隔层面的限速（见_call_llm_with_
# retry），这才是这个平台真正需要的防线；TPM_LIMIT保留但只当第二道保险，
# 数值不必再较真。
TPM_LIMIT = 150000
TPM_SAFETY_MARGIN = 0.75
_token_usage_log: list[tuple[float, int]] = []

# 实测2秒间隔仍然会被限流（20次里10次429，且越往后越密集），保守起见拉到
# 5秒/次（约12 RPM）作为默认下限——配合已有的指数退避重试（RateLimitError
# 撞上时还会再等5/10/20/40/60秒），双保险。这个值没有精确调优过，只是"实测
# 2秒不够、留了明显余量"的保守选择，如果批量跑还是频繁撞429，往上调；如果
# 从来不撞，可以尝试往下调换取速度。
MIN_REQUEST_INTERVAL_SECONDS = 5.0
_last_llm_call_time: list[float] = [0.0]

MAX_STEPS = 30  # Online-Mind2Web官方论文/leaderboard常见步数上限在25-30左右
URGENT_REMAINING_STEPS = 6

# 输出目录用runs/baseline_deepseek_cache_v2/，不是runs/baseline_deepseek_cache/
# ——2026-09-18把user消息顺序改成task_instruction+history_txt在前、budget_txt
# 在后（见build_messages里的注释），这是会实质性改变缓存命中率/费用的改动，
# 旧目录runs/baseline_deepseek_cache/里的15个任务是改动前的代码跑出来的，
# 不能跟新代码的结果混在一起对比。raps那边（mind2web_agent_raps_deepseek.py）
# 同一天做了同一处改动，对照用的是runs/raps_deepseek_cache_v3/。
_SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = _SCRIPT_DIR / "runs" / "baseline_deepseek_llmlingua" / "logs"
SUBMISSION_DIR = _SCRIPT_DIR / "runs" / "baseline_deepseek_llmlingua" / "submissions"

# 方法：LLMLingua-2（Pan et al., Findings of ACL 2024），按原论文惯例"指令和问题不动、
# 只压上下文"——system prompt（指令）和任务描述（问题）原样发送，只压动作历史（上下文）。
# 实现：action_history_log换成_CompressingHistory，每条记录在append时压缩一次、之后
# 原样复用（不是每步把整段历史重压一遍——本机CPU上整段重压第25步要约3秒，逐条压
# 每步只处理一行）。只压"Step N: ..."动作行中间的动作描述：行首"Step N:"和行尾
# "| SUCCESS/| FAILED"是结构性标记（模型靠它判断上一步成没成功），原样保留；
# 不以"Step N:"开头的行（chunk版特有的"后续动作未执行"系统提示）不压——它不是
# 动作历史，而且baseline没有这种行，压了只会让chunk单方面吃亏。
# 压缩器加载/推理失败时原样使用未压缩文本，不让省token的旁路拖垮任务。
# 本机CPU实测（2026-09-24）：加载约50秒、内存约+450MB（峰值约830MB），所以不要跟
# 别的批跑同时跑。
LLMLINGUA_MODEL_NAME = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
LLMLINGUA_RATE = 0.5
_COMPRESSOR = None
_STEP_PREFIX_RE = re.compile(r"^(Step \d+: )")
_STATUS_SUFFIX_RE = re.compile(r"( \| (?:SUCCESS|FAILED))$")


def _get_compressor():
    global _COMPRESSOR
    if _COMPRESSOR is None:
        from llmlingua import PromptCompressor
        _COMPRESSOR = PromptCompressor(
            model_name=LLMLINGUA_MODEL_NAME, use_llmlingua2=True, device_map="cpu",
        )
    return _COMPRESSOR


def _compress_history_line(line: str, task_id: str | None, index: int) -> str:
    m = _STEP_PREFIX_RE.match(line)
    if not m:
        return line
    prefix, body = m.group(1), line[m.end():]
    ms = _STATUS_SUFFIX_RE.search(body)
    suffix = ms.group(1) if ms else ""
    if ms:
        body = body[:ms.start()]
    record = {"timestamp": datetime.now().isoformat(), "task_id": task_id,
              "history_index": index, "original": body}
    try:
        r = _get_compressor().compress_prompt(body, rate=LLMLINGUA_RATE)
        compressed = r.get("compressed_prompt") or body
        record.update(compressed=compressed, origin_tokens=r.get("origin_tokens"),
                      compressed_tokens=r.get("compressed_tokens"))
    except Exception as e:
        compressed = body
        record.update(compressed=body, error=f"{e.__class__.__name__}: {e}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "compression_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return prefix + compressed + suffix


class _CompressingHistory(list):
    """append时自动压缩的动作历史列表，其余行为跟list完全一样。"""

    def __init__(self, task_id: str | None):
        super().__init__()
        self.task_id = task_id

    def append(self, line: str) -> None:
        super().append(_compress_history_line(line, self.task_id, len(self)))

# 视口尺寸固定，直接决定截图分辨率、也决定模型返回的坐标是在哪个坐标系下解读。
# 固定下来是为了让"模型看到的截图"和"Playwright实际点击时用的坐标"始终对齐，
# 不会因为窗口大小变化导致模型给的坐标点错地方。
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 800

# 是否用headless模式。建议先设成False（有头模式）跑通几个真实任务，
# 一是能肉眼观察agent在页面上的实际点击行为，二是headed Chromium比
# headless版本更不容易被网站bot防护拦截（见run_task里browser.launch处的
# 详细说明）。等确认pipeline稳定、要批量跑很多任务追求速度时，
# 再考虑切回True——但切回True之后如果发现某些网站又开始出现Access Denied，
# 大概率还是bot检测的问题，不是代码逻辑坏了。
HEADLESS = False

SYSTEM_PROMPT = """你是一个网页浏览agent，通过看截图和坐标点击来完成任务。

每一步你会看到当前页面的截图。你需要输出一个JSON对象，包含以下字段：
- "thought": 你的推理过程（一两句话，不要太长）
- "action": 动作类型，必须是以下之一：
  - {{"type": "CLICK", "x": <int>, "y": <int>, "description": "点击了什么"}}
  - {{"type": "TYPE", "x": <int>, "y": <int>, "text": "<要输入的文本>", "description": "..."}}
    （TYPE会先在(x,y)处点击聚焦，再输入text）
  - {{"type": "SCROLL", "direction": "down"或"up", "amount": <int，同样是0-1000归一化尺度下的量>, "description": "..."}}
  - {{"type": "PRESS_KEY", "key": "<按键名，如Enter/Escape/Tab>", "description": "..."}}
  - {{"type": "GO_BACK", "description": "..."}}
  - {{"type": "WAIT", "description": "..."}}
  - {{"type": "TASK_COMPLETE", "answer": "<最终答案文本>"}}

严格规则：
- 坐标(x, y)使用0-1000的归一化尺度，不是原始像素值：
  x=0表示截图最左边，x=1000表示截图最右边；
  y=0表示截图最上边，y=1000表示截图最下边。
  是相对当前这张截图的比例位置，不是相对整个页面、也不是绝对像素。
- CLICK和TYPE必须同时给出x和y两个坐标字段，不能只给一个。
- 只输出这一个JSON对象，不要输出markdown代码块围栏（不要用```包裹），
  不要输出JSON以外的任何文字。
- 如果已经完成任务或者确信无法再取得进展，输出TASK_COMPLETE，answer字段给出
  你能给出的最合理答案，不要在没有任何提交的情况下让任务空转到步数耗尽。
"""

# 每种action类型的必填字段，用于在解析成功但字段缺失时提前拦截，
# 避免execute_action/action_dict_to_v2_string里KeyError直接把整个episode打断。
# "type"本身不用列在这里，单独校验。
_REQUIRED_ACTION_FIELDS: dict[str, tuple[str, ...]] = {
    "CLICK": ("x", "y"),
    "TYPE": ("x", "y", "text"),
    "SCROLL": (),
    "PRESS_KEY": ("key",),
    "GO_BACK": (),
    "WAIT": (),
    "TASK_COMPLETE": (),
}


def _try_repair_json(raw_text: str) -> str:
    """在标准json.loads()解析失败后，尝试修复几种实测观察到的、可确定性
    还原模型真实意图的畸形JSON模式，返回修复后的字符串（修复不了则原样返回，
    交给上层再次json.loads()去触发正常的"解析失败退化WAIT"路径）。

    这不是"防失败工程"——这里不掩盖agent真实决策能力的不足，只是把模型
    已经清楚表达出来的坐标点还原成合法JSON，跟_validate_action_fields()
    处理"x输出成list"是同一性质的解析层健壮性修复。

    目前已知模式：
    - "x": 200, 345, "y": 345  -> "x": 200, "y": 345
      （模型把y的值提前多写了一遍到x后面，逗号分隔多出一个孤立数字，
      实测在Qwen3-VL上出现过连续多步稳定复现同一条畸形输出的情况，
      不修复的话会让整个episode从某一步开始持续空转到步数耗尽）
    """
    # x字段值后面跟着一个孤立数字、再跟"y"字段：把中间孤立数字删掉
    repaired = re.sub(
        r'("x"\s*:\s*-?\d+(?:\.\d+)?)\s*,\s*-?\d+(?:\.\d+)?\s*,\s*("y"\s*:)',
        r'\1, \2',
        raw_text,
    )
    return repaired


def _try_repair_json(raw_text: str) -> str:
    """在标准json.loads()解析失败后，尝试修复"坐标元组"这一类畸形JSON模式，
    返回修复后的字符串（修复不了则原样返回，交给上层再走_loose_extract_action()
    这道更彻底的兜底，或者最终判定为真正的解析失败）。

    这不是"防失败工程"——不掩盖agent真实决策能力的不足，只是把模型已经
    清楚表达出来的坐标点还原成合法JSON，跟_validate_action_fields()处理
    "x输出成list"是同一性质的解析层健壮性修复。

    实测观察到的根因：模型有时候脑子里想的是Python风格的(x, y)元组，
    只把它硬塞进JSON里，具体又分两种变体，用同一条正则统一处理
    （用零宽断言偷看两个数字后面紧跟的key名，不消耗它，根据key名是不是
    "y"来判断具体是哪种变体）：
    - 变体1（"y"值被多写了一遍）: "x": 200, 345, "y": 345
      -> "x": 200, "y": 345
    - 变体2（"y"这个key整个缺失，第二个数字其实就是y的值）:
      "x": 965, 197, "description": "..."
      -> "x": 965, "y": 197, "description": "..."
    """
    pattern = re.compile(
        r'"x"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*'
        r'(?="([a-zA-Z_][a-zA-Z0-9_]*)"\s*:)'
    )

    def _repl(m: re.Match) -> str:
        x_val, second_num, next_key = m.group(1), m.group(2), m.group(3)
        if next_key == "y":
            return f'"x": {x_val}, '
        return f'"x": {x_val}, "y": {second_num}, '

    return pattern.sub(_repl, raw_text)


def _grab_num(text: str, field: str) -> float | None:
    m = re.search(rf'"{field}"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    return float(m.group(1)) if m else None


def _grab_str(text: str, field: str) -> str | None:
    m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    return m.group(1) if m else None


def _loose_extract_action(raw_text: str) -> dict | None:
    """最后一道防线：当json.loads()连_try_repair_json()修复后仍然失败时
    （比如漏了个逗号、括号没配对、转义没写对，而不是我们已知的坐标元组
    模式），不再要求整段输出必须是合法JSON，而是直接用针对性正则从原始
    文本里逐字段摘取type/x/y/text/key/direction/amount/answer/description。

    只要能识别出已知的type、并且该动作类型的必填字段全部摘取到，就认为
    提取成功；只要有一个必填字段摘不出来就返回None，交给上层走正常的
    "解析失败退化WAIT"路径——不会因为"这个动作反正摘不全"就编造缺失值，
    跟_try_repair_json()一样，这仍然是在还原模型已经表达清楚的意图，
    不是在掩盖agent真实能力不足导致的失败。
    """
    type_match = re.search(r'"type"\s*:\s*"([A-Za-z_]+)"', raw_text)
    if not type_match:
        return None
    t = type_match.group(1).upper()
    if t not in _REQUIRED_ACTION_FIELDS:
        return None

    action: dict = {"type": t, "description": _grab_str(raw_text, "description") or ""}

    if t in ("CLICK", "TYPE"):
        x, y = _grab_num(raw_text, "x"), _grab_num(raw_text, "y")
        if x is None or y is None:
            return None
        action["x"], action["y"] = x, y
        if t == "TYPE":
            text_val = _grab_str(raw_text, "text")
            if text_val is None:
                return None
            action["text"] = text_val
    elif t == "SCROLL":
        direction = _grab_str(raw_text, "direction")
        amount = _grab_num(raw_text, "amount")
        if direction is not None:
            action["direction"] = direction
        if amount is not None:
            action["amount"] = amount
    elif t == "PRESS_KEY":
        key_val = _grab_str(raw_text, "key")
        if key_val is None:
            return None
        action["key"] = key_val
    elif t == "TASK_COMPLETE":
        action["answer"] = _grab_str(raw_text, "answer") or ""
    # GO_BACK / WAIT 没有额外必填字段，走到这里直接算提取成功

    return action


def _flatten_string_action(parsed: dict, action_type: str) -> dict | None:
    """还原"action被压扁成裸字符串"这个畸形模式：模型本该输出`"action":
    {"type": X, ...参数}`，实际写成了`"action": X`，参数全都跑到了跟
    "action"同级的顶层字段里（比如`{"thought": "...", "action":
    "TASK_COMPLETE", "answer": "..."}`）。这仍然是合法JSON、能被
    json.loads()解析，只是action的值类型不对——直接把顶层其它字段
    （去掉thought/action两个key）当成参数拼回一个正常的action dict，
    不用正则，比DSML那层更简单可靠。

    只在action_type是已知合法类型时才处理，否则返回None交给下面的
    兜底继续尝试——跟其它几层兜底一样，不编造缺失字段。
    """
    t = action_type.upper()
    if t not in _REQUIRED_ACTION_FIELDS:
        return None
    action = {k: v for k, v in parsed.items() if k not in ("thought", "action")}
    action["type"] = t
    action.setdefault("description", "")
    return action


# DeepSeek-V4.1特有的畸形输出模式：偶尔（实测约1/5步的概率）不听指令输出
# JSON，而是切回自己的原生工具调用格式DSML。标准DSML写法是`<DSML invoke
# name="X">`后面嵌套独立的`<DSML parameter name="x">928</parameter>`子标签，
# 但实测中模型输出很不稳定，同一个session里就见过至少三种变体：
# 1. 参数摊平写成invoke标签自己的属性：`invoke name="X" x="928" y="90"
#    description="...">`
# 2. 标准嵌套写法：`invoke name="X">` + `parameter name="x">928</parameter>`
# 3. 半路又换成了临时拼的伪XML标签：`invoke name="X"><x>928</x><y>90</y>
#    <description>...</description>`
# 三种在同一个字段上可能混用（比如x/y摊平写、description又走嵌套parameter
# 标签）。DeepSeek官方vendor过来的deepseek_encoding.parse_message_from_
# completion_text()解析不了这类输出（已验证：它要求完整原始token序列，带
# EOS结束符等控制token，我们从OpenAI兼容API拿到的content字段已经被剥掉了
# ——直接喂进去会报"缺少EOS token"）。这里用三条正则分别提取三种变体、
# 结果合并（先出现的变体优先），在本轮session实测抓到的全部54条真实畸形
# 样本上跑通，全部正确还原出type+参数（第一版只处理了变体1，只覆盖43/54，
# 加上变体2/3后补齐）。
_DSML_INVOKE_HEAD = re.compile(r'invoke\s+name="(\w+)"([^>]*?)(?:/>|>)', re.DOTALL)
_DSML_FLAT_ATTR_PATTERN = re.compile(r'(\w+)="([^"]*)"')
_DSML_NESTED_PARAM_PATTERN = re.compile(r'parameter\s+name="(\w+)"[^>]*>(.*?)<', re.DOTALL)
_DSML_ADHOC_TAG_FIELDS = ("x", "y", "text", "description", "direction", "amount", "key", "answer")
_DSML_ADHOC_TAG_PATTERN = re.compile(
    r'<(' + '|'.join(_DSML_ADHOC_TAG_FIELDS) + r')>(.*?)</\1>', re.DOTALL,
)


def _dsml_extract_action(raw_text: str) -> dict | None:
    """从DeepSeek原生DSML工具调用格式里摘取动作，是_loose_extract_action()
    的姊妹兜底——同样的"只还原模型已经表达清楚的意图，摘不全就返回None"
    原则，只是针对DSML的三种畸形变体而不是JSON的`"key": value`写法。
    """
    if "DSML" not in raw_text or "invoke" not in raw_text:
        return None
    m = _DSML_INVOKE_HEAD.search(raw_text)
    if not m:
        return None
    t = m.group(1).upper()
    if t not in _REQUIRED_ACTION_FIELDS:
        return None

    attrs: dict[str, str] = {}
    # 变体1：参数摊平写在invoke标签自己的属性里
    for k, v in _DSML_FLAT_ATTR_PATTERN.findall(m.group(2)):
        attrs.setdefault(k, v)
    rest = raw_text[m.end():]
    # 变体2：标准嵌套<parameter name="x">值</parameter>子标签
    for k, v in _DSML_NESTED_PARAM_PATTERN.findall(rest):
        attrs.setdefault(k, v)
    # 变体3：临时拼的伪XML标签<x>值</x>
    for k, v in _DSML_ADHOC_TAG_PATTERN.findall(rest):
        attrs.setdefault(k, v)

    action: dict = {"type": t, "description": attrs.get("description", "")}
    if t in ("CLICK", "TYPE"):
        if "x" not in attrs or "y" not in attrs:
            return None
        try:
            action["x"], action["y"] = float(attrs["x"]), float(attrs["y"])
        except ValueError:
            return None
        if t == "TYPE":
            if "text" not in attrs:
                return None
            action["text"] = attrs["text"]
    elif t == "SCROLL":
        if "direction" in attrs:
            action["direction"] = attrs["direction"]
        if "amount" in attrs:
            try:
                action["amount"] = float(attrs["amount"])
            except ValueError:
                pass
    elif t == "PRESS_KEY":
        if "key" not in attrs:
            return None
        action["key"] = attrs["key"]
    elif t == "TASK_COMPLETE":
        action["answer"] = attrs.get("answer", "")
    # GO_BACK / WAIT 没有额外必填字段，走到这里直接算提取成功

    return action


def _validate_action_fields(action: dict, raw_output: str) -> None:
    """校验action dict字段完整性和基本类型。字段缺失或类型不对时抛ValueError，
    跟JSON解析失败走同一条容错路径（run_task里当作一次WAIT处理，不中断episode）。

    实测发现视觉模型偶尔会把x/y坐标输出成list而不是单个数字（比如
    "x": [948, 58]，把两个坐标塞进了一个字段里），这种情况字段"存在"，
    能通过缺字段检查，但类型不对，会在execute_action里让Playwright调用
    直接报错（"expected number, got object"）——虽然那边try/except能兜住
    不崩溃，但会浪费一次LLM调用和一步预算。提前在这里做类型检查更干净。
    """
    t = action.get("type", "")
    t_upper = t.upper() if isinstance(t, str) else ""
    if t_upper not in _REQUIRED_ACTION_FIELDS:
        raise ValueError(f"模型输出了未知action类型{t!r}，原始输出：{raw_output!r}")
    missing = [f for f in _REQUIRED_ACTION_FIELDS[t_upper] if f not in action]
    if missing:
        raise ValueError(
            f"action类型{t_upper}缺少必填字段{missing}，原始输出：{raw_output!r}"
        )
    if t_upper in ("CLICK", "TYPE"):
        for coord_field in ("x", "y"):
            if not isinstance(action[coord_field], (int, float)) or isinstance(action[coord_field], bool):
                raise ValueError(
                    f"action类型{t_upper}的'{coord_field}'字段类型不对"
                    f"（期望数字，实际是{action[coord_field]!r}），原始输出：{raw_output!r}"
                )


class VisionAgent:
    """截图驱动的Online-Mind2Web agent。

    职责边界跟baseline.py里的BaselineAgent一致：只负责"看当前观察 -> 出一个动作"，
    不做周期检测/黑名单拦截这类"防失败工程"——原因跟baseline.py注释里写的一样，
    这是个baseline，应该保留它在benchmark上自然会有的失败模式。
    """

    def __init__(self):
        # run_task()构造完实例后立刻设置成当前任务的task_id，仅用于给
        # _log_token_calibration()的每条记录打标——原来token_calibration.jsonl
        # 是所有任务共用的一份追加日志，没有task_id字段的话，消融实验想按
        # task_id统计baseline vs raps各自的token消耗完全做不到（日志里
        # 几十个任务的调用记录全混在一起，没法归属）。跟raps那边
        # （mind2web_agent_raps.py）同一处修复，两边格式对齐才能直接比较。
        self.task_id: str | None = None
        self.last_thought: str = ""

    def _estimate_tokens(self, messages: list[dict]) -> tuple[int, int]:
        """粗略估算：文本部分按2字符/token算，每张图按固定1000token估算
        （视觉模型的图片token开销跟分辨率/patch数有关，1000只是保守占位值，
        待用真实response.usage校准）。

        返回(估算总token数, 本次请求包含的图片张数)——图片数单独返回，
        方便_log_token_calibration()按"每张图实际花了多少token"反推校准值。
        """
        total = 0
        image_count = 0
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
                        image_count += 1
        return total, image_count

    def _log_token_calibration(self, estimated_tokens: int, image_count: int,
                                usage, step: int | None, mode: str | None,
                                text_tokens_real: int | None = None) -> None:
        """把每次LLM调用的"估算值 vs 真实response.usage"都记下来，
        跑几个真实任务之后可以直接看这个文件，统计真实的
        每张图token开销、prompt token/字符比例，回头校准
        _estimate_tokens()里的占位常量（目前是"每张图1000token"、
        "文本2字符/token"，都是没验证过的猜测值）。

        step/mode是消融实验要用的、跟raps那边（mind2web_agent_raps.py）
        对齐的字段结构——baseline每步都是真实截图感知，没有复用模式，
        mode固定传"感知"，纯粹是为了两边日志schema一致，方便统一脚本
        处理，不代表baseline这边有什么模式区分的概念。

        text_tokens_real是用真实tokenizer精确算出的、这次调用去掉图片后
        的文本token数（见_real_text_tokens）。image_tokens_estimated=
        actual_prompt_tokens-text_tokens_real，是"视觉token消耗"这个
        指标的真实来源——SiliconFlow的usage不返回模态拆分，只能靠这个
        减法反推；复用模式（image_count=0）下这个值应该接近0，可以当
        方法本身是否可靠的一个持续性sanity check。

        cached_prompt_tokens/estimated_cost_yuan_peak是2026-09-16新加的
        两个指标（命中缓存token数、实际费用）。cached_prompt_tokens直接
        读usage.prompt_tokens_details.cached_tokens——这是七牛云在真实
        response里返回的字段，不是估算值（真机测试过：同一段长prompt连续
        调用会显示非零cached_tokens，命中率能到90%左右，命中前缀不受结尾
        变化内容影响，见对话记录里的验证）。estimated_cost_yuan_peak用
        PRICE_YUAN_PER_K的高峰单价乘真实token数算出，是这次调用花费的
        保守估算（上限），因为不知道具体调用发生在高峰还是空闲时段。
        """
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        actual_prompt_tokens = getattr(usage, "prompt_tokens", None)
        actual_completion_tokens = getattr(usage, "completion_tokens", None)
        image_tokens_estimated = (
            actual_prompt_tokens - text_tokens_real
            if actual_prompt_tokens is not None and text_tokens_real is not None
            else None
        )
        prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
        cached_prompt_tokens = (
            getattr(prompt_tokens_details, "cached_tokens", None)
            if prompt_tokens_details is not None else None
        )
        if actual_prompt_tokens is not None and actual_completion_tokens is not None:
            cached = cached_prompt_tokens or 0
            noncached = actual_prompt_tokens - cached
            estimated_cost_yuan_peak = (
                cached * PRICE_YUAN_PER_K["cached_input"]
                + noncached * PRICE_YUAN_PER_K["noncached_input"]
                + actual_completion_tokens * PRICE_YUAN_PER_K["output"]
            ) / 1000
        else:
            estimated_cost_yuan_peak = None
        record = {
            "timestamp": datetime.now().isoformat(),
            "task_id": self.task_id,
            "step": step,
            "mode": mode,
            "estimated_tokens": estimated_tokens,
            "image_count": image_count,
            "actual_prompt_tokens": actual_prompt_tokens,
            "actual_completion_tokens": actual_completion_tokens,
            "actual_total_tokens": getattr(usage, "total_tokens", None),
            "text_tokens_real": text_tokens_real,
            "image_tokens_estimated": image_tokens_estimated,
            "cached_prompt_tokens": cached_prompt_tokens,
            "estimated_cost_yuan_peak": estimated_cost_yuan_peak,
        }
        with open(LOG_DIR / "token_calibration.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _wait_for_token_budget(self, estimated_tokens: int) -> None:
        now = time.monotonic()
        while _token_usage_log and now - _token_usage_log[0][0] > 60:
            _token_usage_log.pop(0)
        used = sum(t for _, t in _token_usage_log)
        budget = TPM_LIMIT * TPM_SAFETY_MARGIN
        if used + estimated_tokens > budget:
            if _token_usage_log:
                wait_seconds = 60 - (now - _token_usage_log[0][0]) + 0.5
                if wait_seconds > 0:
                    print(f"  ⏳ 预计本次调用约{estimated_tokens}token，"
                          f"过去60秒已用约{used}token，等待{wait_seconds:.1f}秒...")
                    time.sleep(wait_seconds)

    def _wait_for_request_interval(self) -> None:
        """真实瓶颈是RPM不是TPM（见MIN_REQUEST_INTERVAL_SECONDS定义处的说明），
        这里强制两次真实LLM调用之间至少间隔MIN_REQUEST_INTERVAL_SECONDS秒。
        """
        now = time.monotonic()
        elapsed = now - _last_llm_call_time[0]
        if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
            wait_seconds = MIN_REQUEST_INTERVAL_SECONDS - elapsed
            print(f"  ⏳ 距上次调用仅{elapsed:.1f}秒，为避开RPM限流等待{wait_seconds:.1f}秒...")
            time.sleep(wait_seconds)
        _last_llm_call_time[0] = time.monotonic()

    def _call_llm_with_retry(self, messages: list[dict], max_retries: int = 5,
                              step: int | None = None, mode: str | None = None) -> str:
        estimated_tokens, image_count = self._estimate_tokens(messages)
        text_tokens_real = _real_text_tokens(messages)
        self._wait_for_token_budget(estimated_tokens)
        self._wait_for_request_interval()
        delay = 5.0
        for attempt in range(max_retries):
            try:
                # extra_body关掉思考模式（2026-09-15验证过）：DeepSeek-V4.1-
                # Flash默认会走思考模式，哪怕没有请求它——全量943次raps真实
                # 调用里completion_tokens从没低于过149、中位数1211，正常
                # GUI动作JSON回复通常几十token就够了，这个地板高到不正常，
                # 印证了默认在思考。真机对照测过同一个GUI动作提示词：默认
                # 2.6秒/342token（184思考token），关掉后1.3秒/141token（0
                # 思考token），输出JSON质量看起来相当。这个GUI点击/输入类
                # 任务不需要长链条推理，思考模式只是纯粹的延迟和token开销，
                # 关掉换取更快更省。试过"chat_template_kwargs":{"thinking":
                # False}和"enable_thinking": False两种常见写法都不生效，
                # 只有这个"thinking": {"type": "disabled"}是真的管用的。
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                # 用真实usage.total_tokens（如果API返回了的话）记进限速滑动窗口，
                # 比用估算值更准；拿不到usage时退回估算值，不影响原有限速逻辑。
                actual_total = getattr(response, "usage", None)
                tokens_for_rate_limit = (
                    actual_total.total_tokens
                    if actual_total is not None and getattr(actual_total, "total_tokens", None)
                    else estimated_tokens
                )
                _token_usage_log.append((time.monotonic(), tokens_for_rate_limit))
                if actual_total is not None:
                    self._log_token_calibration(estimated_tokens, image_count, actual_total,
                                                 step, mode, text_tokens_real)
                return response.choices[0].message.content.strip()
            # 原来这里只重试字符串里带"429"/"rate limit"的错误——真机批量跑
            # 撞到过SiliconFlow返回500"Unknown error"这类瞬时性服务端故障
            # （不含"429"字样，会被误判成"不该重试"直接原样抛出、崩掉整个
            # 任务）。跟raps那边（mind2web_agent_raps.py）同一处修复：改成
            # 按openai SDK给出的具体异常类型判断，覆盖429限流/5xx服务端
            # 错误/网络层超时或断连这几类"大概率跟本次请求内容无关、重试
            # 一下就好"的基础设施抖动；其余错误（400参数错误、401鉴权
            # 失败）大概率重试也没用，直接抛出更快暴露问题。
            except (RateLimitError, InternalServerError,
                    APIConnectionError, APITimeoutError) as e:
                if attempt == max_retries - 1:
                    raise
                print(f"  ⏳ 触发可重试错误（{type(e).__name__}），"
                      f"{delay:.0f}秒后重试（第{attempt + 1}/{max_retries}次）...")
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError("不应该到达这里")

    def build_messages(self, task_instruction: str, screenshot_b64: str, step: int,
                        max_steps: int, action_history: list[str],
                        force_final: bool = False) -> list[dict]:
        # 2026-09-18：user文本顺序改成task_instruction+history_txt在前、
        # budget_txt在后——跟raps那边（mind2web_agent_raps_deepseek.py）
        # 同一处改动、同一个理由，见那边_build_history_text/_build_budget_text
        # 的docstring（这是通用的prompt缓存效率问题，跟raps的感知/复用
        # 机制毫无关系，按项目一贯原则两边同步修）：缓存是从prompt最开头
        # 精确前缀匹配、一旦某个token不同后面全部作废，不能"跳过中间
        # 不同的地方"。budget_txt里的"当前第X步"数字每次调用必然不同，
        # 原来排在action_history前面，导致action_history_log这部分——
        # 本来是只增不改、上一次调用的内容必然是这一次的前缀、100%保证
        # 稳定命中的——因为排在budget_txt后面，从来没有机会被缓存命中过。
        # 挪到budget_txt前面后，实测（同样5-15个任务）baseline缓存命中率
        # 会有实质提升。
        user_text = f"任务目标：{task_instruction}"
        if action_history:
            # 传完整历史而不是只传"上一步"——单帧+单步反馈会让模型每一步
            # 都是"失忆"地重新判断，没有能力意识到"我已经连续点了好几次
            # 同一个位置了"。这不是在给agent的决策能力打补丁、掩盖它的
            # 失败（那样才是防失败工程），而是在补全observation本身该有
            # 的基本构成——几乎所有标准GUI/web agent的prompt设计里，动作
            # 历史都是跟任务目标、当前画面平级的基础输入，不是可选项。
            # 纯文字历史，token成本相对一整张截图可以忽略不计，30步内
            # 累计下来也不值得做截断优化，不加没必要的复杂度。
            history_txt = "\n".join(action_history)
            user_text += f"\n\n已执行的动作历史（从第1步到现在）：\n{history_txt}"

        remaining = max_steps - step
        budget_txt = f"\n\n当前第{step + 1}步，总共最多{max_steps}步（剩余{remaining}步）。"
        if force_final or remaining <= URGENT_REMAINING_STEPS:
            budget_txt += ("\n⚠️ 步数即将耗尽（或已到最后一步）！必须输出TASK_COMPLETE，"
                            "基于目前已掌握的信息给出你能给出的最合理答案。")
        user_text += budget_txt

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    },
                ],
            },
        ]

    def get_action(self, task_instruction: str, screenshot_b64: str, step: int,
                    max_steps: int, action_history: list[str],
                    force_final: bool = False) -> dict:
        messages = self.build_messages(
            task_instruction, screenshot_b64, step, max_steps,
            action_history, force_final=force_final,
        )
        raw_output = self._call_llm_with_retry(messages, step=step, mode="感知")

        # 跟baseline.py同样的输出格式归一化：模型偶尔会用```json包裹，先剥掉
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip())
        # 关闭思考模式后新出现的高频畸形（思考开时5次，关掉后实测43次）：
        # 模型习惯性地把英文单引号转义成\'（Python字符串写法），但这在JSON
        # 里是不合法的转义序列，会让json.loads()直接报"Invalid \escape"，
        # 而且是在最外层就失败，后面_try_repair_json/_loose_extract_action
        # 这些兜底都是在"整体大致像JSON"的假设上做局部修复，对这种最外层
        # 解析失败一样束手无策。JSON字符串里单引号本来就不需要转义，直接
        # 把\'替换成'必然更接近模型的真实意图，不会误伤本来就合法的JSON
        # （合法JSON里不会出现这个序列，替换是安全的空操作）。
        cleaned = cleaned.replace("\\'", "'")

        action: dict | None = None
        thought = ""

        # 第一层：标准解析
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = None
            # 第二层：修复已知的"坐标元组"畸形模式后重试一次
            repaired = _try_repair_json(cleaned)
            if repaired != cleaned:
                try:
                    parsed = json.loads(repaired)
                except json.JSONDecodeError:
                    parsed = None

        if parsed is not None:
            thought = parsed.get("thought", "")
            candidate = parsed.get("action")
            if isinstance(candidate, dict):
                action = candidate
            elif isinstance(candidate, str):
                # 关闭思考模式后新出现的畸形模式（思考开的时候几乎不出现，
                # 关掉之后实测跳到了79次/6个任务）：模型把"action": {"type":
                # X, ...参数}压扁成了"action": X，参数字段全都跑到跟action
                # 同级。这是合法JSON，能被json.loads()解析，只是action的
                # 值不是我们要的dict，是个裸字符串——用_flatten_string_
                # action()把同级字段捞回来拼成正常的action dict。真机
                # 实测这个畸形几乎总是卡在TASK_COMPLETE这一步（模型已经
                # 想好了答案，收尾格式却写错了），不修的话会让整个episode
                # 从这一步开始每步都解析失败，一直空转到步数耗尽——哪怕
                # 答案早就在parsed里躺着。
                action = _flatten_string_action(parsed, candidate)

        if action is None:
            # 第三层兜底：前两层都救不回来（比如漏逗号、括号不配对这类
            # 我们没见过的畸形），改用逐字段正则摘取，不再要求整体合法JSON。
            # 摘不出必填字段就还是保持None，走下面的正常报错路径。
            action = _loose_extract_action(cleaned)
            if action is not None and not thought:
                thought = _grab_str(cleaned, "thought") or ""

        if action is None:
            # 第四层兜底（DeepSeek特有）：模型没听指令输出JSON，切回了自己
            # 原生的DSML工具调用格式，见_dsml_extract_action()注释。
            action = _dsml_extract_action(cleaned)
            if action is not None and not thought:
                # DSML输出通常在<DSML calls>标签之前还留了一段自然语言的
                # thought（比如"The product menu is still open. Let me
                # close it..."），跟JSON格式里的thought字段不是一回事，
                # 直接取标签前的文本整体当thought，没有就留空。
                idx = cleaned.find("DSML")
                thought = cleaned[:idx].strip() if idx > 0 else ""

        if action is None:
            raise ValueError(f"模型输出不是合法JSON，原始输出：{raw_output!r}")

        self.last_thought = thought
        _validate_action_fields(action, raw_output)
        return action



def action_dict_to_v2_string(action: dict) -> tuple[str, str]:
    """把内部action dict转换成Online-Mind2Web v2 schema的Grammar A字符串。

    返回 (action_string, status_placeholder)，status要等实际执行完才知道
    SUCCESS/FAILED，这里先返回占位，调用方执行完动作后再拼回去。
    """
    t = action.get("type", "").upper()
    desc = action.get("description", "")
    if t == "CLICK":
        return f"CLICK coords({action['x']}, {action['y']}) -> {desc}", "STATUS"
    elif t == "TYPE":
        return f"TYPE coords({action['x']}, {action['y']}) -> {desc}", "STATUS"
    elif t == "SCROLL":
        return f"SCROLL page -> {desc}", "STATUS"
    elif t == "PRESS_KEY":
        return f"PRESS_KEY page -> {desc}", "STATUS"
    elif t == "GO_BACK":
        return "page -> GO_BACK -> " + (desc or "返回上一页"), "STATUS"
    elif t == "WAIT":
        return f"WAIT page -> {desc}", None  # WAIT不需要status
    elif t == "TASK_COMPLETE":
        return f"TASK_COMPLETE -> ANSWER: {action.get('answer', '')}", None
    else:
        raise ValueError(f"未知action类型: {t}")


def _cdp_screenshot(page) -> bytes:
    """绕过Playwright高层page.screenshot()的"稳定性等待"（含字体加载检查），
    直接走Chrome DevTools Protocol截图。CDP的Page.captureScreenshot只是
    简单地截取当前渲染出来的画面，不会等待任何"稳定"信号，所以不会卡在
    "waiting for fonts to load"——代价是可能截到一个还没完全渲染稳定的
    画面（比如动画正播放到一半），但对agent来说"看到当前真实状态"本身
    就是目标，不需要等到画面绝对静止。

    用作page.screenshot()超时之后的兜底，而不是直接替换掉它——
    page.screenshot()的稳定性等待在大多数正常网站上是有用的（能避免
    截到过渡态画面），只在少数字体资源有问题的网站上才会异常卡死。
    """
    cdp = page.context.new_cdp_session(page)
    try:
        result = cdp.send("Page.captureScreenshot", {"format": "png"})
    finally:
        cdp.detach()
    return base64.b64decode(result["data"])


class _ActivePageTracker:
    """跟踪当前应该被agent感知的"激活"标签页。

    点击有时会触发新标签页（target="_blank"链接、window.open()），Playwright
    不会自动把已有的page引用切换过去——如果不显式监听，代码会一直对着最早
    创建的那个标签页截图/执行动作，导致新标签页对agent完全不可见：截图看
    起来"什么都没变"，模型判断点击没生效，于是反复重试同一个点击，每次都
    真的再开一个新窗口，越堆越多（实测在批量跑的结果里观察到同一页面被
    重复打开多次的现象，根因就是这个）。

    这是环境可观测性缺陷，不是agent能力不足——agent的动作确实生效了，只是
    harness没把"生效后的新状态"如实呈现给它。在harness层面统一修，
    baseline和RAPS都会同样受益，跟"两者必须在同一harness下实现、
    唯一变量是感知决策逻辑"这条原则一致，不是给某一方开小灶打补丁。

    策略：始终把"最近打开的那个页面"当作激活页面（跟大多数人类用户的直觉
    一致——新标签页打开后注意力自然转移过去）。如果激活页面被关闭（比如
    某些网站的新标签页只是临时用来做统计埋点，秒开秒关），回退到context
    里还开着的最后一个页面。
    """

    def __init__(self, context, initial_page):
        self._context = context
        self._current = initial_page
        context.on("page", self._on_new_page)
        initial_page.on("close", self._on_page_closed)

    def _on_new_page(self, new_page) -> None:
        try:
            new_page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            pass
        self._current = new_page
        new_page.on("close", self._on_page_closed)

    def _on_page_closed(self, closed_page) -> None:
        if closed_page is self._current:
            remaining = [p for p in self._context.pages if not p.is_closed()]
            self._current = remaining[-1] if remaining else closed_page

    @property
    def page(self):
        return self._current


def execute_action(page, action: dict) -> tuple[bool, str]:
    """在Playwright page上执行动作，返回(是否成功, 人类可读的结果描述)。

    成功/失败判定比baseline.py里简单粗暴：只要Playwright调用没抛异常就算SUCCESS。
    这跟WebArena那边"页面是否变化"的判定标准不一样——截图驱动的场景下，
    "页面是否变化"需要额外截图比对，先用最简单的版本跑通，
    如果之后发现FAILED比例明显失真，再考虑加截图diff判定。
    """
    t = action.get("type", "").upper()
    try:
        if t == "CLICK":
            page.mouse.click(action["x"], action["y"])
            page.wait_for_timeout(500)  # 给页面一点反应时间，纯经验值，非WebArena那个500ms超时bug的同类问题
            return True, "点击执行完成"
        elif t == "TYPE":
            page.mouse.click(action["x"], action["y"])
            page.wait_for_timeout(200)
            # 打字前先全选一次已有内容（Ctrl+A），让接下来的keyboard.type()
            # 覆盖掉它，而不是追加在后面。这是纯粹的执行层bug修复，跟
            # 感知/复用机制无关——点击聚焦的输入框如果被自动补全/地理定位
            # 塞进了默认建议（截图上可能显示为空，但实际DOM里已经有内容），
            # 不清空直接打字会把新文字拼接在看不见的默认值后面，产出语法
            # 正确但语义错误的输入（真机跑raps时踩到过"Beverly Hills,
            # CA90011"这类拼接乱码）。跟raps那边（mind2web_agent_raps.py）
            # 同一处修复，两边都该有，不属于消融实验要控制的变量。
            page.keyboard.press("Control+A")
            page.wait_for_timeout(50)
            page.keyboard.type(action["text"])
            return True, "输入执行完成"
        elif t == "SCROLL":
            dy = action.get("amount", 500)
            if action.get("direction") == "up":
                dy = -dy
            page.mouse.wheel(0, dy)
            page.wait_for_timeout(300)
            return True, "滚动执行完成"
        elif t == "PRESS_KEY":
            page.keyboard.press(action["key"])
            page.wait_for_timeout(300)
            return True, "按键执行完成"
        elif t == "GO_BACK":
            page.go_back()
            page.wait_for_timeout(500)
            return True, "返回执行完成"
        elif t == "WAIT":
            page.wait_for_timeout(1000)
            return True, "等待完成"
        elif t == "TASK_COMPLETE":
            return True, "任务标记完成"
        else:
            return False, f"未知动作类型: {t}"
    except Exception as e:
        return False, f"执行异常: {e}"


def load_task(task_id: str | None = None) -> dict:
    """加载一条Online-Mind2Web任务记录。

    真实字段名已用本地拉取的数据集核实（HuggingFace数据集卡片README字段名
    和实际数据不一致，以下以本地print()验证过的实际结构为准）：
    - "task_id" (str)
    - "confirmed_task" (str)  —— 注意：数据集卡片README写的是"task_description"，
      实际字段名是"confirmed_task"（跟原版Mind2Web一致），README没更新，别被误导
    - "website" (str)
    - "reference_length" (int)
    - "level" (str)：难度分级（如"medium"），README未提及，一并保留

    该数据集是gated的，需要先在 https://huggingface.co/datasets/osunlp/Online-Mind2Web
    页面登录并同意协议，本地拉取前还需要 `huggingface-cli login`，否则会报权限错误。

    数据集只有一个split，实际名字是"test"（不是常见的"train"），已用
    list(ds.keys())[0]动态获取，不用硬编码。

    task_id=None时返回手写示例任务（跑通端到端流程用，不是真实数据）；
    传入具体task_id时从真实数据集里查找。
    """
    if task_id is None:
        return {
            "task_id": "example_task_001",
            "website": "https://www.example.com",
            "confirmed_task": "示例任务：请描述你在这个页面上看到了什么",
            "reference_length": 3,
            "level": None,
        }

    from datasets import load_dataset

    ds = load_dataset("osunlp/Online-Mind2Web")
    split_name = list(ds.keys())[0]
    for record in ds[split_name]:
        if record["task_id"] == task_id:
            return {
                "task_id": record["task_id"],
                "website": record["website"],
                "confirmed_task": record["confirmed_task"],
                "reference_length": record["reference_length"],
                "level": record.get("level"),
            }
    raise ValueError(f"未在数据集中找到task_id={task_id!r}")


def load_all_tasks() -> list[dict]:
    """一次性加载Online-Mind2Web全部任务（300条），返回字段结构跟
    load_task()单条查询保持一致。批量跑（run_batch.py）用这个而不是
    循环调用load_task(task_id=...)——后者每次都要重新触发一次
    datasets.load_dataset()，300个任务循环调用就是300次重复加载，
    这个函数只加载一次。
    """
    from datasets import load_dataset

    ds = load_dataset("osunlp/Online-Mind2Web")
    split_name = list(ds.keys())[0]
    return [
        {
            "task_id": record["task_id"],
            "website": record["website"],
            "confirmed_task": record["confirmed_task"],
            "reference_length": record["reference_length"],
            "level": record.get("level"),
        }
        for record in ds[split_name]
    ]


def run_task(task: dict, max_steps: int = MAX_STEPS,
             submission_dir: Path = SUBMISSION_DIR, verbose: bool = True) -> dict:
    """跑完一个Online-Mind2Web任务，产出符合v2 schema的result.json + trajectory截图。

    对页面加载/截图这类Playwright调用做了异常隔离：单个任务遇到网络超时、
    页面卡死等环境层问题时，不会让整个进程崩溃——会把已收集到的trajectory
    正常写出，并在result.json里的"environment_error"字段记录出错原因，
    方便区分"agent能力不足的失败"和"网站本身加载失败/超时"这两类性质完全
    不同的失败。这个隔离是run_batch.py能安全批量跑的前提，单任务模式下
    先把这层加上。
    """
    task_id = task["task_id"]
    task_dir = submission_dir / task_id
    trajectory_dir = task_dir / "trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    agent = VisionAgent()
    agent.task_id = task_id
    action_history_v2: list[dict] = []
    agent_final_answer = None
    action_history_log: list[str] = _CompressingHistory(task_id)
    env_error: str | None = None
    # 循环外初始化，语义是"最后已知页面URL"（不是"最后一次动作执行结果
    # 文本"——那是exec_result该管的事）。这是跟raps那边（
    # mind2web_agent_raps.py）同一个bug、同一处修复：这个变量原来叫
    # last_action_result，只在循环内至少完整跑完一步（走到execute_action
    # 那一行）才会被赋值；如果第0步的LLM调用就直接抛出未被重试逻辑捕获的
    # 异常（比如AuthenticationError这类不在RateLimitError/
    # InternalServerError/APIConnectionError/APITimeoutError重试白名单里的
    # 错误类型），下面步数耗尽兜底逻辑里引用这个变量会直接UnboundLocalError
    # 崩溃——这本该是"环境层失败不让整个任务崩溃"的保护，却被这个bug自己
    # 绕过了。真机批量跑baseline时40个任务里有2个就是这么崩的。
    last_known_url = ""

    with sync_playwright() as p:
        # HEADLESS原样保留True时，Playwright的headless Chromium带有明显的
        # 自动化指纹（navigator.webdriver=true、旧版UA字符串带"HeadlessChrome"
        # 标识），很多网站的bot防护（Akamai/Cloudflare等）会直接在服务器层
        # 拒绝这类请求（返回边缘节点的Access Denied页，不是真实页面内容）。
        # 之前跑Trader Joe's官网撞到的就是这种情况（errors.edgesuite.net是
        # Akamai的错误域名）。缓解措施：
        # 1. 加--disable-blink-features=AutomationControlled参数
        # 2. 用add_init_script在页面JS执行前覆盖navigator.webdriver
        # 3. 显式指定不带HeadlessChrome标识的UA字符串
        # 这些只能降低被拦概率，不保证100%绕过所有bot防护——如果某个网站
        # 防护更严格（比如带JS challenge/验证码），这些措施可能不够，
        # 到时候只能把该任务记为环境层失败，不用继续硬啃。
        browser = p.chromium.launch(
            headless=HEADLESS,
            args=["--disable-blink-features=AutomationControlled"],
        )
        # 改用browser.new_context()+context.new_page()而不是原来的
        # browser.new_page()快捷方式——原因有两个：
        # 1. 只有在context层面才能监听context.on("page", ...)捕获新开的
        #    标签页（见下面的_ActivePageTracker）
        # 2. add_init_script()挂在context上会自动对该context里未来新开的
        #    每个标签页生效，挂在单个page上则只对那一个页面生效——原来的
        #    写法下，如果点击真开出了新标签页，新标签页是完全裸奔的
        #    （没有navigator.webdriver覆盖、没有字体检查绕过），等于给
        #    反爬检测和截图卡死这两个已修复的问题都开了后门
        context = browser.new_context(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        # 修复screenshot()卡在"waiting for fonts to load"的已知Playwright问题
        # （microsoft/playwright仓库里有多个issue报告过这个问题，#35200/#35972/
        # #32144等，多版本多平台都出现过，不是本地环境特有的）：Playwright的
        # screenshot()内部会轮询document.fonts.ready这个Promise来判断字体是否
        # 加载完，某些网站这个Promise永远不resolve，导致screenshot()一直卡到
        # 超时。这里在每次页面加载前就把document.fonts.ready强制覆盖成一个
        # 立即resolve的Promise，让Playwright内部检查瞬间通过，不再傻等。
        # 官方issue #35200评论区给出的正是这个绕过方案。
        context.add_init_script(
            "Object.defineProperty(document, 'fonts', "
            "{value: {ready: Promise.resolve()}})"
        )
        page = context.new_page()
        active = _ActivePageTracker(context, page)

        if verbose:
            print(f"===== 任务 {task_id} =====\n目标：{task['confirmed_task']}\n起始网站：{task['website']}\n")

        # 官方要求：必须从benchmark指定的起始网站出发，不能让agent自己搜索跳转，
        # 否则不同run之间难度不可比。
        # wait_until用"domcontentloaded"而不是Playwright默认的"load"——
        # "load"要等页面上所有资源（图片/iframe/第三方脚本/埋点/客服插件等）
        # 全部加载完才触发，很多真实网站这个事件很晚才触发甚至不触发，
        # 会直接把goto()拖到超时。domcontentloaded更快更可靠，跟原本紧接着
        # 那句wait_for_load_state("domcontentloaded")保持一致（原来那句其实
        # 是在"load"已经等到之后再多等一次domcontentloaded，完全是重复劳动，
        # 真正该等的反而没在goto()里生效）。
        try:
            active.page.goto(task["website"], wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            env_error = f"页面加载失败（page.goto超时或异常）: {e}"
            if verbose:
                print(f"  ⚠️ {env_error}")

        if env_error is None:
            for step in range(max_steps):
                is_last_step = step == max_steps - 1
                # 每一步都重新取一次当前激活页面，而不是复用循环外的固定
                # 引用——上一步的动作可能刚好切换了激活标签页（见
                # _ActivePageTracker），本步的截图/执行/URL记录都要基于
                # "此刻"真正激活的那个页面，不然还是等于没修
                cur_page = active.page
                try:
                    # 显式设timeout（默认30秒），遇到字体资源加载异常/页面卡死
                    # 这类问题时更快失败退出，不用死等到默认值；出错时不让
                    # 整个进程崩溃，而是记录环境错误、结束这个任务的循环，
                    # 已经收集到的前面几步截图和记录仍然正常保留、写入结果。
                    screenshot_bytes = cur_page.screenshot(timeout=15000)
                except Exception as e:
                    # page.screenshot()卡死（通常是字体加载检查这个已知问题，
                    # add_init_script覆盖document.fonts.ready只能缓解一部分
                    # 网站，不保证100%生效）——先不急着放弃整个任务，
                    # 用CDP直接截图兜底一次，这条路径不会触发同样的等待逻辑。
                    if verbose:
                        print(f"  ⚠️ 第{step}步page.screenshot()超时（{e.__class__.__name__}），"
                              f"尝试CDP兜底截图...")
                    try:
                        screenshot_bytes = _cdp_screenshot(cur_page)
                    except Exception as e2:
                        env_error = (
                            f"第{step}步截图失败：page.screenshot和CDP兜底都失败了。"
                            f"page.screenshot错误: {e}；CDP错误: {e2}"
                        )
                        if verbose:
                            print(f"  ⚠️ {env_error}")
                        break
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

                screenshot_filename = f"{step:04d}.png"
                (trajectory_dir / screenshot_filename).write_bytes(screenshot_bytes)

                llm_start = time.monotonic()
                try:
                    action = agent.get_action(
                        task["confirmed_task"], screenshot_b64, step, max_steps,
                        action_history_log, force_final=is_last_step,
                    )
                except ValueError as e:
                    if verbose:
                        print(f"  ⚠️ 第{step}步模型输出解析失败: {e}")
                    # 解析失败当作一次FAILED的WAIT处理，不中断整个episode
                    action = {"type": "WAIT", "description": f"模型输出解析失败: {e}"}
                except Exception as e:
                    # _call_llm_with_retry内部已经对429/5xx/超时/连接错误做过
                    # 多轮重试，能走到这里说明重试也没能恢复——这是真正的
                    # 基础设施故障（比如API持续不可用），不是模型输出格式
                    # 问题。原来这里没有这一层，导致重试耗尽后异常直接冒泡
                    # 出run_task()、被run_batch外层当成"未预期崩溃"：已经
                    # 收集到的前面几步截图/记录全部丢光，还会污染批次汇总
                    # 统计（本该算"环境层失败"，却被记成"crashed"）。跟raps
                    # 那边（mind2web_agent_raps.py）同一处修复：记env_error、
                    # 结束这个任务的循环，已收集到的前面几步照常写入结果。
                    env_error = str(e)
                    if verbose:
                        print(f"  ⚠️ 第{step}步LLM调用重试耗尽，判定为环境层故障: {env_error}")
                    break
                else:
                    # 把模型输出的0-1000归一化坐标换算成截图实际像素坐标。
                    # 根因：Qwen3-VL系列模型的grounding原生输出习惯就是0-1000
                    # 归一化尺度（多个官方文档/社区实测确认过这一点），之前
                    # prompt里要求它直接输出像素值，模型大概率仍按自己训练时
                    # 的习惯给归一化值，导致坐标系不匹配、点击位置系统性偏移
                    # （实测证据：Trader Joe's弹窗任务里模型连续多步稳定输出
                    # (949,58)附近的值，换算成1280x800下的归一化坐标后落在
                    # (1215,46)——正好是弹窗右上角关闭按钮的合理位置，说明
                    # 模型其实"看对了"，只是输出的数值单位不对）。
                    if action.get("type", "").upper() in ("CLICK", "TYPE"):
                        # clamp到0-1000防止模型偶尔越界输出（比如1000.5、-3这种），
                        # 换算后确保落在实际viewport范围内，不会点到画面外
                        norm_x = max(0, min(1000, action["x"]))
                        norm_y = max(0, min(1000, action["y"]))
                        action["x"] = round(norm_x / 1000 * VIEWPORT_WIDTH)
                        action["y"] = round(norm_y / 1000 * VIEWPORT_HEIGHT)
                llm_elapsed = time.monotonic() - llm_start

                action_str, status_placeholder = action_dict_to_v2_string(action)
                url_before = cur_page.url
                last_known_url = url_before

                if verbose:
                    print(f"[Step {step}] {action_str}  (LLM耗时{llm_elapsed:.1f}s)")

                is_terminal = action.get("type", "").upper() == "TASK_COMPLETE"
                if is_terminal:
                    agent_final_answer = action.get("answer", "")
                    success, exec_result = True, "任务标记完成"
                else:
                    success, exec_result = execute_action(cur_page, action)

                if status_placeholder == "STATUS":
                    status = "SUCCESS" if success else "FAILED"
                    action_str = f"{action_str} | {status}"

                # 累积进完整动作历史，供下一步get_action()使用——注意这里用
                # 拼完SUCCESS/FAILED之后的最终action_str，模型看到的历史里
                # 会明确知道"上次点这个位置到底成没成功"，而不只是"点过"
                action_history_log.append(f"Step {step + 1}: {action_str}")

                step_record = {
                    "step": step,
                    "screenshot": screenshot_filename,
                    "action": action_str,
                    "thought": agent.last_thought or None,
                    "url": url_before,
                }
                action_history_v2.append(step_record)

                if not success and verbose:
                    print(f"  ⚠️ 动作执行失败: {exec_result}")

                if is_terminal:
                    break

        browser.close()

    # 如果耗尽步数都没有TASK_COMPLETE（包括因env_error提前终止的情况），
    # 官方schema要求最后一步的action必须以TASK_COMPLETE开头——补一个强制
    # 收尾的终止步，跟baseline.py里force_final那套"步数耗尽前必须给出最
    # 合理答案"的要求是同一个精神。区分一下措辞：env_error不是None时说明
    # 是环境层问题提前中断，不是agent真的没给出答案，答案文本上标注清楚，
    # 方便之后统计"环境失败"和"agent能力失败"两类占比时不会混在一起。
    if agent_final_answer is None:
        if env_error is not None:
            agent_final_answer = f"（任务因环境错误提前终止，未能给出答案：{env_error}）"
        else:
            agent_final_answer = "（步数耗尽，未能给出明确答案）"
        action_history_v2.append({
            "step": len(action_history_v2),
            "screenshot": action_history_v2[-1]["screenshot"] if action_history_v2 else "0000.png",
            "action": f"TASK_COMPLETE -> ANSWER: {agent_final_answer}",
            "thought": "步数耗尽或环境错误，兜底提交",
            "url": last_known_url,
        })
        # 注意：这条兜底步复用了最后一张已有的截图文件名，没有新截图——
        # 如果WebJudge校验要求screenshot严格对应新文件会报错，到时候再改成
        # "复制最后一张截图重新编号"而不是复用文件名。先标记成TODO。

    result = {
        "schema_version": "online-mind2web-v2",
        "task": task["confirmed_task"],
        "task_id": task_id,
        "agent_final_answer": agent_final_answer,
        "reference_length": task.get("reference_length"),
        "action_history": action_history_v2,
        "environment_error": env_error,
    }

    result_path = task_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n任务结束。共{len(action_history_v2)}步，结果写入：{result_path.resolve()}")

    return result


if __name__ == "__main__":
    # 先跑手写的示例任务，验证端到端流程（截图→模型出坐标→Playwright执行→
    # 写出v2 schema的result.json）能跑通，再接入真实的Online-Mind2Web数据集
    task = load_task()
    run_task(task)
