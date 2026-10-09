"""Run WebJudge (webjudge/webjudge.py) on the result.json files produced by either agent.

Both agents write the same Online-Mind2Web v2 result schema, so one evaluator serves both.
Per-task failures are isolated, and tasks already present in the output jsonl are skipped
(resume by default). Both submission layouts are accepted:
  submissions_dir/{task_id}/result.json and submissions_dir/{task_id}_{timestamp}/result.json
(the newest one is used).

Examples:
    python run_webjudge.py --submissions-dir runs/chunk_deepseek_v3/submissions
    python run_webjudge.py --submissions-dir runs/baseline_deepseek_paired/submissions --task-ids id1,id2
    python run_webjudge.py --submissions-dir runs/chunk_deepseek_v3/submissions --no-resume

Results are appended to --output (default: webjudge_results.jsonl next to submissions_dir),
one full record per task (predicted_label, response, key_points, image_judge_record).
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from mind2web_agent_deepseek import load_all_tasks
from webjudge.webjudge import evaluate_task


def _resolve_submission_dir(task_id: str, submissions_dir: Path) -> Path | None:
    """兼容baseline的固定目录和raps的带时间戳目录，返回带有效result.json的
    那个目录；raps那边如果同一个task_id被重跑过好几次（比如手动测试留下的
    多个时间戳目录），取其中修改时间最新的一份，不是跑最多步的那份——
    "最新"更符合"这是最后一次真正想保留的结果"这个直觉，也是run_batch_raps.py
    resume逻辑本来就默认认可的语义（它只关心"有没有"，不区分新旧，但选
    最新的对齐"如果手动重跑过，应该看最后一次"这个使用习惯）。
    """
    exact = submissions_dir / task_id
    if (exact / "result.json").exists():
        return exact
    candidates = [
        d for d in submissions_dir.glob(f"{task_id}_*")
        if (d / "result.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _has_existing_score(task_id: str, output_path: Path) -> bool:
    if not output_path.exists():
        return False
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("task_id") == task_id:
                return True
    return False


def run_webjudge_batch(tasks: list[dict], submissions_dir: Path, output_path: Path,
                        score_threshold: int, resume: bool = True) -> dict:
    summary: dict[str, list[str]] = {
        "success": [], "failure": [], "env_error_skip": [],
        "no_submission": [], "judge_error": [], "skipped": [],
    }
    total = len(tasks)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, task in enumerate(tasks, start=1):
        task_id = task["task_id"]

        if resume and _has_existing_score(task_id, output_path):
            print(f"[{i}/{total}] {task_id} 已有评测结果，跳过（--no-resume可强制重新评测）")
            summary["skipped"].append(task_id)
            continue

        task_dir = _resolve_submission_dir(task_id, submissions_dir)
        if task_dir is None:
            print(f"[{i}/{total}] {task_id} 在{submissions_dir}下找不到result.json，跳过"
                  f"（还没跑过agent，不是评测脚本负责的事）")
            summary["no_submission"].append(task_id)
            continue

        with open(task_dir / "result.json", "r", encoding="utf-8") as f:
            result = json.load(f)

        task_description = result.get("task", task["confirmed_task"])

        # 环境层失败（页面加载失败/反爬拦截导致agent自己都进不去网站这类，
        # result.json里environment_error字段非空）本来就不可能完成任务，
        # 直接判failure，不用再花一次完整的WebJudge调用（通常是1次key
        # point识别 + N次截图打分 + 1次终判）去确认这个显而易见的结论。
        if result.get("environment_error"):
            print(f"[{i}/{total}] {task_id} 是environment_error"
                  f"（{result['environment_error']!r}），直接判failure，不调用WebJudge")
            record = {
                "task_id": task_id,
                "task": task_description,
                "predicted_label": 0,
                "response": None,
                "key_points": None,
                "image_judge_record": None,
                "skipped_reason": "environment_error",
            }
            summary["env_error_skip"].append(task_id)
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            continue

        steps = result.get("action_history", [])
        action_history = [step["action"] for step in steps]
        screenshot_dir = task_dir / "trajectory"
        screenshot_paths = [screenshot_dir / step["screenshot"] for step in steps]

        print(f"\n{'=' * 60}\n[{i}/{total}] 开始评测 {task_id}"
              f"（{len(screenshot_paths)}张截图）\n{'=' * 60}")
        t0 = time.monotonic()
        try:
            eval_result = evaluate_task(
                task_description, action_history, screenshot_paths,
                score_threshold=score_threshold,
            )
        except Exception:
            print(f"⚠️ 任务{task_id}评测过程中发生未捕获异常（不影响其它任务的评测）：")
            traceback.print_exc()
            summary["judge_error"].append(task_id)
            continue
        elapsed = time.monotonic() - t0

        category = "success" if eval_result["predicted_label"] == 1 else "failure"
        summary[category].append(task_id)

        record = {"task_id": task_id, "task": task_description, **eval_result}
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[{i}/{total}] {task_id} 评测结束，判定={category}，耗时{elapsed:.1f}s")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submissions-dir", type=str, required=True,
                         help="要评测的submissions目录，比如raps/submissions或submissions")
    parser.add_argument("--output", type=str, default=None,
                         help="评测结果输出的jsonl路径，不指定则默认写在"
                              "submissions-dir同级的webjudge_results.jsonl")
    parser.add_argument("--score-threshold", type=int, default=3,
                         help="截图打分阈值（1-5分），达到这个分数才会被采纳进最终判决的"
                              "关键截图集合，默认3（跟官方eval.sh默认值一致）")
    parser.add_argument("--n", type=int, default=None,
                         help="评测前N个任务（按数据集原始顺序）")
    parser.add_argument("--task-ids", type=str, default=None,
                         help="逗号分隔的指定task_id列表，优先级高于--n")
    parser.add_argument("--no-resume", action="store_true",
                         help="忽略已有评测结果，强制重新评测")
    args = parser.parse_args()

    submissions_dir = Path(args.submissions_dir)
    if not submissions_dir.exists():
        print(f"目录不存在：{submissions_dir.resolve()}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else submissions_dir.parent / "webjudge_results.jsonl"

    print("正在加载Online-Mind2Web数据集...")
    all_tasks = load_all_tasks()
    print(f"共加载{len(all_tasks)}条任务。")

    if args.task_ids:
        wanted_ids = set(x.strip() for x in args.task_ids.split(","))
        tasks = [t for t in all_tasks if t["task_id"] in wanted_ids]
        missing = wanted_ids - {t["task_id"] for t in tasks}
        if missing:
            print(f"⚠️ 以下task_id在数据集里没找到，已忽略：{missing}")
    else:
        tasks = all_tasks
        if args.n is not None:
            tasks = tasks[:args.n]

    if not tasks:
        print("筛选后没有任何任务，检查--task-ids参数是否正确。")
        sys.exit(1)

    print(f"本次将尝试评测{len(tasks)}个任务（实际数量取决于哪些已经有result.json），"
          f"resume={'开启' if not args.no_resume else '关闭（强制重新评测）'}，"
          f"输出到{output_path.resolve()}\n")

    summary = run_webjudge_batch(
        tasks, submissions_dir, output_path, args.score_threshold,
        resume=not args.no_resume,
    )

    print(f"\n{'=' * 60}\n评测批次结束\n{'=' * 60}")
    for category in ("success", "failure", "env_error_skip", "no_submission", "judge_error", "skipped"):
        print(f"  {category}: {len(summary[category])}")

    n_success = len(summary["success"])
    n_failure = len(summary["failure"])
    n_env = len(summary["env_error_skip"])
    scored_all = n_success + n_failure + n_env
    scored_reachable = n_success + n_failure  # 只算真的进了网站互动过的任务
    if scored_all:
        print(f"\n成功率（含环境层失败算failure）: "
              f"{n_success / scored_all * 100:.1f}%（{n_success}/{scored_all}）")
    if scored_reachable:
        print(f"成功率（排除环境层失败，只算真的进了网站互动的任务）: "
              f"{n_success / scored_reachable * 100:.1f}%（{n_success}/{scored_reachable}）")


if __name__ == "__main__":
    main()
