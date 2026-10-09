"""同期成对baseline的批跑脚本（2026-09-25新建）。

agent代码直接import正式版mind2web_agent_deepseek.py，一行不改（模型、temperature、thinking关闭、
MAX_STEPS、不等页面稳定都跟chunk v3.1一致，chunk跟它的唯一区别就是分块机制）；这里只把输出
重定向到runs/baseline_deepseek_paired/，不跟09-18跑的runs/baseline_deepseek_cache_v2/混在一起
（那批比baseline代码09-22的最后一次修改还早）。任务队列见tools/queue_paired.json，
配套检查脚本tools/paired_batch_check.py。

用法：python run_batch_baseline_deepseek_paired.py --task-ids <id1>,<id2>
"""

import sys

import mind2web_agent_deepseek as agent
import run_batch_mind2web_deepseek as runner

_OUT = agent._SCRIPT_DIR / "runs" / "baseline_deepseek_paired"
agent.LOG_DIR = _OUT / "logs"            # _log_token_calibration()运行时读这个全局变量
agent.SUBMISSION_DIR = _OUT / "submissions"
runner.LOG_DIR = agent.LOG_DIR           # batch_summary写这里
runner.SUBMISSION_DIR = agent.SUBMISSION_DIR

# run_batch/run_task的submission_dir默认值在定义时就绑定了旧目录，包一层显式传新目录
_orig_run_batch = runner.run_batch
runner.run_batch = lambda tasks, resume=True: _orig_run_batch(tasks, resume=resume, submission_dir=agent.SUBMISSION_DIR)

if __name__ == "__main__":
    if "--task-ids" not in sys.argv:
        sys.exit("成对实验只允许用--task-ids指定任务（按tools/queue_paired.json的顺序取）")
    runner.main()
