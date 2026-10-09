"""LLMLingua-2叠加实验的四组对比（2026-09-27）：baseline / chunk / baseline+LLMLingua / chunk+LLMLingua。
只用四组都可比的任务（队列前100个；任一组环境错误、≤5步拒绝访问/空白、人工排除名单都剔除）。
费用按论文公式(2)：所有输入token按非缓存价0.002元/千，输出0.008元/千。
另外核对LLMLingua在真实API token上的实际压缩效果（文本token/调用），不看它自报的origin/compressed_tokens。
用法：python tools/lingua_numbers.py
"""
import json, os, re, collections, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
Q = json.load(open("tools/queue_paired.json"))[:100]
ARMS = {"baseline": "baseline_deepseek_paired", "chunk": "chunk_deepseek_v3",
        "baseline+L": "baseline_deepseek_llmlingua", "chunk+L": "chunk_deepseek_llmlingua"}
MANUAL_EXCLUDE = {"871e7771cecb989972f138ecc373107b", "f389398d2eeb29e5571e00439c57eb76"}
BLOCK_RE = re.compile(r"blocked|denied|无法访问|拒绝访问|Access Denied|不可用|not available|无法加载|空白", re.I)
res = lambda d, t: json.load(open(f"runs/{d}/submissions/{t}/result.json", encoding="utf-8"))
blk = lambda r: len(r["action_history"]) <= 5 and bool(BLOCK_RE.search(r.get("agent_final_answer") or ""))
cost = lambda x: (x["actual_prompt_tokens"] * 0.002 + x["actual_completion_tokens"] * 0.008) / 1000


def logs(d):
    g = collections.defaultdict(list)
    for l in open(f"runs/{d}/logs/token_calibration.jsonl", encoding="utf-8"):
        x = json.loads(l); g[x["task_id"]].append(x)
    return g


ok = [t for t in Q if t not in MANUAL_EXCLUDE and all(os.path.exists(f"runs/{d}/submissions/{t}/result.json") for d in ARMS.values())
      and not any(res(d, t)["environment_error"] or blk(res(d, t)) for d in ARMS.values())]
print(f"四组都可比的任务：{len(ok)}")
L = {k: logs(d) for k, d in ARMS.items()}
S = {}
for k, d in ARMS.items():
    T = collections.Counter()
    for t in ok:
        T["acts"] += len(res(d, t)["action_history"])
        for x in L[k][t]:
            T["calls"] += 1; T["in"] += x["actual_prompt_tokens"]; T["out"] += x["actual_completion_tokens"]
            T["txt"] += x["text_tokens_real"] or 0; T["img"] += x["image_tokens_estimated"] or 0; T["cost"] += cost(x)
    S[k] = T
print(f"\n{'组':11s} {'调用':>6s} {'动作':>6s} {'文本tok/调用':>11s} {'输出tok/调用':>11s} {'总token':>10s} {'费用(元)':>9s}")
for k, T in S.items():
    print(f"{k:11s} {T['calls']:6d} {T['acts']:6d} {T['txt']/T['calls']:11.1f} {T['out']/T['calls']:11.1f} {T['in']+T['out']:10,d} {T['cost']:9.4f}")
b = S["baseline"]
print("\n相对 baseline 的比值：")
for k, T in S.items():
    print(f"  {k:11s} 调用 {T['calls']/b['calls']:.3f}  总token {(T['in']+T['out'])/(b['in']+b['out']):.3f}  费用 {T['cost']/b['cost']:.3f}")
print(f"\nLLMLingua 单独的效果（baseline+L / baseline）：文本token/调用 {S['baseline+L']['txt']/S['baseline+L']['calls']:.1f} vs {b['txt']/b['calls']:.1f}"
      f"（{(S['baseline+L']['txt']/S['baseline+L']['calls'])/(b['txt']/b['calls']):.3f}）")
print(f"chunk 在压缩之上还省多少（chunk+L / baseline+L）：调用 {S['chunk+L']['calls']/S['baseline+L']['calls']:.3f}，费用 {S['chunk+L']['cost']/S['baseline+L']['cost']:.3f}")
print(f"叠加是否相乘：chunk×L 预期费用比 {S['chunk']['cost']/b['cost'] * S['baseline+L']['cost']/b['cost']:.3f}，实际 chunk+L {S['chunk+L']['cost']/b['cost']:.3f}")
# 压缩日志：LLMLingua自报的压缩率（仅供对照）
cr = [json.loads(l) for d in ("baseline_deepseek_llmlingua", "chunk_deepseek_llmlingua") for l in open(f"runs/{d}/logs/compression_log.jsonl", encoding="utf-8")]
o = sum(r.get("origin_tokens") or 0 for r in cr); c = sum(r.get("compressed_tokens") or 0 for r in cr)
print(f"LLMLingua自报（它自己的分词器）：{o} -> {c}（{c/o:.3f}），出错 {sum('error' in r for r in cr)} 条")
