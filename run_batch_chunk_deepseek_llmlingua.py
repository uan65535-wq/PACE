"""【NOT OFFICIAL】LLMLingua-2叠加实验批跑脚本（由tools/make_lingua_deepseek.py生成），复制自run_batch_chunk_deepseek.py，只换import为mind2web_agent_chunk_deepseek_llmlingua，输出runs/chunk_deepseek_llmlingua/。"""
"""批量跑Online-Mind2Web任务（chunk实验分支，mind2web_agent_chunk_deepseek.py，
【实验分支 / NOT OFFICIAL】）。复制自run_batch_mind2web_deepseek.py，只换了import
源；输出目录runs/chunk_deepseek/，跟baseline一样用固定的submissions/{task_id}/路径。

用法举例：
    python run_batch_chunk_deepseek.py --task-ids <id1>,<id2>
    python run_batch_chunk_deepseek.py --n 10 --no-resume
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from mind2web_agent_chunk_deepseek_llmlingua import (
    LOG_DIR,
    SUBMISSION_DIR,
    load_all_tasks,
    run_task,
)


def _classify_result(result: dict | None, crashed: bool) -> str:
    if crashed or result is None:
        return "crashed"
    if result.get("environment_error"):
        return "env_error"
    last_action = result["action_history"][-1]["action"] if result["action_history"] else ""
    if "步数耗尽，未能给出明确答案" in last_action:
        return "step_limit"
    return "completed"


def _has_existing_result(task_id: str, submission_dir: Path) -> bool:
    return (submission_dir / task_id / "result.json").exists()


def run_batch(tasks: list[dict], resume: bool = True,
              submission_dir: Path = SUBMISSION_DIR) -> dict:
    summary: dict[str, list[str]] = {
        "completed": [], "step_limit": [], "env_error": [], "crashed": [], "skipped": [],
    }

    total = len(tasks)
    for i, task in enumerate(tasks, start=1):
        task_id = task["task_id"]

        if resume and _has_existing_result(task_id, submission_dir):
            print(f"[{i}/{total}] {task_id} 已有result.json，跳过（--no-resume可强制重跑）")
            summary["skipped"].append(task_id)
            continue

        print(f"\n{'=' * 60}\n[{i}/{total}] 开始任务 {task_id}\n{'=' * 60}")
        t0 = time.monotonic()
        result = None
        crashed = False
        try:
            result = run_task(task, submission_dir=submission_dir)
        except Exception:
            crashed = True
            print(f"⚠️ 任务{task_id}发生未捕获异常（这不应该发生，需要回头看）：")
            traceback.print_exc()
        elapsed = time.monotonic() - t0

        category = _classify_result(result, crashed)
        summary[category].append(task_id)
        print(f"[{i}/{total}] {task_id} 结束，分类={category}，耗时{elapsed:.1f}s")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=None,
                         help="跑前N个任务（按筛选后的顺序）。不指定则跑筛选出的全部任务")
    parser.add_argument("--task-ids", type=str, default=None,
                         help="逗号分隔的指定task_id列表，优先级高于--n/--level")
    parser.add_argument("--level", type=str, default=None,
                         help="只跑指定难度档位的任务（值以数据集实际level字段为准，比如medium）")
    parser.add_argument("--no-resume", action="store_true",
                         help="默认跳过已有result.json的任务；加这个参数强制全部重跑"
                              "（会覆盖掉已有的result.json/trajectory）")
    args = parser.parse_args()

    print("正在加载Online-Mind2Web数据集（第一次跑可能需要huggingface-cli login过的权限）...")
    all_tasks = load_all_tasks()
    print(f"共加载{len(all_tasks)}条任务。")

    if args.task_ids is not None:
        # 2026-09-27教训：空字符串--task-ids ""在旧代码里被当成"没传"，会误跑全部300个任务
        # （autoround.sh管道里next-baseline偶尔会返回空字符串）。现在空字符串直接报错退出。
        wanted_ids = set(x.strip() for x in args.task_ids.split(",") if x.strip())
        if not wanted_ids:
            sys.exit("⚠️ --task-ids是空字符串，没有可跑的任务，主动退出（避免误跑全部任务）。")
        tasks = [t for t in all_tasks if t["task_id"] in wanted_ids]
        missing = wanted_ids - {t["task_id"] for t in tasks}
        if missing:
            print(f"⚠️ 以下task_id在数据集里没找到，已忽略：{missing}")
    else:
        tasks = all_tasks
        if args.level:
            tasks = [t for t in tasks if t.get("level") == args.level]
        if args.n is not None:
            tasks = tasks[:args.n]

    if not tasks:
        print("筛选后没有任何任务，检查--level/--task-ids参数是否正确。")
        sys.exit(1)

    print(f"本次将跑{len(tasks)}个任务，resume={'开启' if not args.no_resume else '关闭（强制重跑）'}\n")

    summary = run_batch(tasks, resume=not args.no_resume)

    print(f"\n{'=' * 60}\n批次结束\n{'=' * 60}")
    for category in ("completed", "step_limit", "env_error", "crashed", "skipped"):
        print(f"  {category}: {len(summary[category])}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LOG_DIR / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已写入：{summary_path.resolve()}")


if __name__ == "__main__":
    main()
