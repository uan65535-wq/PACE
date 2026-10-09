"""DeepSeek同期成对实验的WebJudge评测（2026-09-26）：按tools/queue_paired.json的顺序，从第一个成对任务开始，
同一个任务先评baseline、再评chunk，两边都评完才进入下一个任务。评测逻辑直接复用run_webjudge.py
（硅基流动Qwen3-VL-32B判卷、断点续跑：输出jsonl里已有该task_id就跳过）。
- 只评两边都有result.json、且两边都没有environment_error的任务：环境错误的结果之后可能被清理重跑
  （tools/clean_env429.py），现在评了会变成过期分数，所以先跳过，最后统一处理。
- --follow：评完当前已就绪的任务后不退出，每10分钟检查一次有没有新跑完的成对任务（可以和实验同时跑）。
输出：runs/baseline_deepseek_paired/webjudge_results.jsonl、runs/chunk_deepseek_v3/webjudge_results.jsonl
用法：python tools/paired_webjudge.py [--follow]
"""
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import run_webjudge as rw  # noqa: E402
import threading  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
import webjudge.webjudge as ww  # noqa: E402

# ---- 加速：同一边轨迹的逐张截图打分并发执行（2026-09-26）----
# evaluate_task里逐张调用judge_image是官方算法里彼此独立的调用（官方实现本来就是asyncio.gather并发）。
# 这里在第一次调用judge_image时，把这条轨迹的全部截图一次性并发打分、按原顺序缓存，之后evaluate_task
# 按顺序取缓存——提示词、结果、判定逻辑完全不变，只改调用的并发度。共用的webjudge/webjudge.py不改。
WORKERS = 6
_current_paths: list = []
_cache: dict = {}
_orig_judge_image = ww.judge_image
_orig_wait = ww._wait_for_token_budget
_lock = threading.Lock()


def _locked_wait(estimated_tokens):
    with _lock:  # 限速滑动窗口是模块级list，多线程下pop/求和要串行
        _orig_wait(estimated_tokens)


def _parallel_judge_image(task, image_path, key_points):
    key = (task, str(image_path), key_points)
    if key not in _cache:
        paths = _current_paths if any(str(p) == str(image_path) for p in _current_paths) else [image_path]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(lambda p: _orig_judge_image(task, p, key_points), paths))
        for p, r in zip(paths, results):
            _cache[(task, str(p), key_points)] = r
    return _cache[key]


# 限速：webjudge.py沿用agent当年的保守值（40000×0.75 TPM），并发后成了瓶颈、硅基流动未出现过429；
# 只对评测放宽到60000×0.75。真撞到平台限流时，_call_llm_with_retry的429重试退避会兜住。
ww.TPM_LIMIT = 60000
ww._wait_for_token_budget = _locked_wait
ww.judge_image = _parallel_judge_image

# 2026-09-27用户要求：只评主实验的100对（队列前100个），评完就停，不评LLMLingua叠加实验
MAIN_N = 100
QUEUE = json.load(open(ROOT / "tools" / "queue_paired.json"))[:MAIN_N]
# 连续这么多次评测出错（多半是硅基流动额度用完）就停WebJudge；主线实验在别的进程，不受影响
MAX_CONSECUTIVE_ERRORS = 5
_consecutive_errors = 0
ARMS = [("baseline", ROOT / "runs" / "baseline_deepseek_paired"), ("chunk", ROOT / "runs" / "chunk_deepseek_v3")]
POLL_SECONDS = 600


def ready(t):
    """两边都有result.json且都没有环境错误。"""
    for _, d in ARMS:
        p = d / "submissions" / t / "result.json"
        if not p.exists():
            return False
        if json.load(open(p, encoding="utf-8")).get("environment_error"):
            return False
    return True


def judged(t, d):
    return rw._has_existing_score(t, d / "webjudge_results.jsonl")


def one_pass():
    """评一遍当前所有就绪、还没评完的任务；返回本轮评了几个任务。"""
    global _consecutive_errors
    n = 0
    for t in QUEUE:
        if not ready(t) or all(judged(t, d) for _, d in ARMS):
            continue
        for name, d in ARMS:
            if judged(t, d):
                continue
            result = json.load(open(d / "submissions" / t / "result.json", encoding="utf-8"))
            task = {"task_id": t, "confirmed_task": result["task"]}
            _current_paths[:] = [d / "submissions" / t / "trajectory" / s["screenshot"] for s in result["action_history"]]
            _cache.clear()
            print(f"\n##### {t[:12]} {name}", flush=True)
            s = rw.run_webjudge_batch([task], d / "submissions", d / "webjudge_results.jsonl", 3)
            if s["judge_error"]:
                _consecutive_errors += 1
                print(f"⚠️ {t} {name} 评测出错（连续第{_consecutive_errors}次），本轮先跳过这个任务的另一边，下轮重试", flush=True)
                if _consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print("连续评测出错，可能是硅基流动额度用完，停止WebJudge", flush=True)
                    sys.exit(3)
                break
            _consecutive_errors = 0
        n += 1
    return n


if __name__ == "__main__":
    follow = "--follow" in sys.argv
    while True:
        n = one_pass()
        done = sum(ready(t) and all(judged(t, d) for _, d in ARMS) for t in QUEUE)
        print(f"\n{time.strftime('%m-%d %H:%M')} 本轮评测{n}个任务；两边都已评完的成对任务共{done}个", flush=True)
        main_done = all((d / "submissions" / t / "result.json").exists() for t in QUEUE for _, d in ARMS)
        if main_done and n == 0:
            print("主实验100对都已跑完并评完，WebJudge停止", flush=True)
            break
        if not follow:
            break
        time.sleep(POLL_SECONDS)
