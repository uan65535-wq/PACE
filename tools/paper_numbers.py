"""重新计算论文（sn-article-template/pace_paper.tex）第5章用到的全部数字。
数据：DeepSeek同期成对实验，runs/chunk_deepseek_v3/ vs runs/baseline_deepseek_paired/，队列tools/queue_paired.json。
费用按论文公式(2)计算：所有输入token都按非缓存价 p_in=0.002元/千token，输出 p_out=0.008元/千token
（论文不讨论缓存，所以不用token日志里按缓存计价的estimated_cost_yuan_peak）。
排除规则与tools/paired_batch_check.py一致：任一边环境错误；任一边≤5步就因拒绝访问/空白交卷；人工排除名单。
用法：python tools/paper_numbers.py
"""
import json, os, re, collections, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
Q = json.load(open("tools/queue_paired.json"))
C, B = "chunk_deepseek_v3", "baseline_deepseek_paired"
MANUAL_EXCLUDE = {"871e7771cecb989972f138ecc373107b", "f389398d2eeb29e5571e00439c57eb76"}  # 跟paired_batch_check.py保持一致
BLOCK_RE = re.compile(r"blocked|denied|无法访问|拒绝访问|Access Denied|不可用|not available|无法加载|空白", re.I)
P_IN, P_OUT = 0.002, 0.008

has = lambda d, t: os.path.exists(f"runs/{d}/submissions/{t}/result.json")
res = lambda d, t: json.load(open(f"runs/{d}/submissions/{t}/result.json", encoding="utf-8"))
blk = lambda r: len(r["action_history"]) <= 5 and bool(BLOCK_RE.search(r.get("agent_final_answer") or ""))
cost = lambda x: (x["actual_prompt_tokens"] * P_IN + x["actual_completion_tokens"] * P_OUT) / 1000
typ = lambda a: "GO_BACK" if "GO_BACK" in a["action"][:20] else a["action"].split()[0]


def logs(d):
    g = collections.defaultdict(list)
    for l in open(f"runs/{d}/logs/token_calibration.jsonl", encoding="utf-8"):
        x = json.loads(l); g[x["task_id"]].append(x)
    return g


both = [t for t in Q if has(C, t) and has(B, t)]
env = [t for t in both if res(C, t)["environment_error"] or res(B, t)["environment_error"]]
blocked = [t for t in both if t not in env and (blk(res(C, t)) or blk(res(B, t)))]
manual = [t for t in both if t in MANUAL_EXCLUDE and t not in env and t not in blocked]
ok = [t for t in both if t not in env and t not in blocked and t not in MANUAL_EXCLUDE]
print(f"[5.1 排除] 两边都跑完 {len(both)}；环境错误 {len(env)}；拒绝访问/空白 {len(blocked)}；首步访问不对称(人工) {len(manual)}；可比 {len(ok)}")

cl, bl = logs(C), logs(B)
T = collections.Counter(); ratios = []; end = collections.Counter(); sizes = collections.Counter(); reasons = collections.Counter()
seg = 0
for t in ok:
    c, b = res(C, t), res(B, t); s = c["chunk_stats"]
    for side, L in (("b", bl[t]), ("c", cl[t])):
        T[side + "calls"] += len(L)
        for x in L:
            T[side + "img"] += x["image_tokens_estimated"] or 0
            T[side + "txt"] += x["text_tokens_real"] or 0
            T[side + "in"] += x["actual_prompt_tokens"]
            T[side + "out"] += x["actual_completion_tokens"]
            T[side + "cost"] += cost(x)
    T["bacts"] += len(b["action_history"]); T["cacts"] += len(c["action_history"])
    T["fu"] += s["followup_actions_executed"]; T["ab"] += s["guard_aborts"]; T["drop"] += s["actions_dropped"]
    for z in s["planned_chunk_sizes"]:
        sizes[z] += 1; T["planned"] += z - 1
    for x in s["abort_reasons"]:
        reasons[re.sub(r"（原来是.*", "", x)] += 1
    ratios.append(sum(cost(x) for x in cl[t]) / sum(cost(x) for x in bl[t]))
    for side, r in (("b", b), ("c", c)):
        end[side + ("_exhaust" if "步数耗尽" in r["action_history"][-1]["action"] else "_done")] += 1
    H = b["action_history"]  # 理论上限：baseline自己的轨迹 + 公式(19)的边界指示
    for i, a in enumerate(H):
        if typ(a) in ("SCROLL", "WAIT", "GO_BACK", "PRESS_KEY", "TASK_COMPLETE") or i == len(H) - 1 or H[i + 1]["url"] != a["url"]:
            seg += 1

n = len(ok); bc, cc = T["bcalls"], T["ccalls"]
pct = lambda c_, b_: f"{c_ / b_ * 100:.1f}% ({(c_ / b_ - 1) * 100:+.1f}%)"
print("\n[表2 主结果，总计]  指标 / baseline / PACE / 比值")
rows = [("模型调用 J", bc, cc), ("执行动作 A", T["bacts"], T["cacts"]), ("视觉token", T["bimg"], T["cimg"]),
        ("文本token(输入)", T["btxt"], T["ctxt"]), ("输出token", T["bout"], T["cout"]),
        ("总token", T["bin"] + T["bout"], T["cin"] + T["cout"])]
for name, b_, c_ in rows:
    print(f"  {name:14s} {b_:>10,} {c_:>10,}  {pct(c_, b_)}")
print(f"  {'每动作截图 rho':14s} {bc / T['bacts']:>10.3f} {cc / T['cacts']:>10.3f}")
print(f"  {'费用(元)':14s} {T['bcost']:>10.4f} {T['ccost']:>10.4f}  {pct(T['ccost'], T['bcost'])}")
print(f"  自己交卷 b{end['b_done']} c{end['c_done']}；步数耗尽 b{end['b_exhaust']} c{end['c_exhaust']}")
print(f"\n[表3 每任务平均] 调用 {bc/n:.2f}/{cc/n:.2f}；动作 {T['bacts']/n:.2f}/{T['cacts']/n:.2f}；视觉token {T['bimg']/n:,.0f}/{T['cimg']/n:,.0f}；"
      f"总token {(T['bin']+T['bout'])/n:,.0f}/{(T['cin']+T['cout'])/n:,.0f}；费用 {T['bcost']/n:.4f}/{T['ccost']/n:.4f}")
q = st.quantiles(ratios, n=4)
print(f"  逐任务费用比 中位数 {st.median(ratios):.3f}，四分位 {q[0]:.3f}–{q[2]:.3f}，PACE更省 {sum(r < 1 for r in ratios)}/{n}；每调用覆盖动作 A/J={T['cacts']/cc:.2f}")
print(f"\n[5.3 成本分解] 费用比 {T['ccost']/T['bcost']:.3f} = 调用比 {cc/bc:.3f} × 每调用费用比 {(T['ccost']/cc)/(T['bcost']/bc):.3f}")
print("[表4 每次调用] 视觉 {:.1f}/{:.1f}；文本 {:.1f}/{:.1f}；输出 {:.1f}/{:.1f}；费用(10^-3元) {:.3f}/{:.3f}".format(
    T["bimg"]/bc, T["cimg"]/cc, T["btxt"]/bc, T["ctxt"]/cc, T["bout"]/bc, T["cout"]/cc, T["bcost"]/bc*1000, T["ccost"]/cc*1000))
d_in = (T["ctxt"]/cc - T["btxt"]/bc) * P_IN; d_out = (T["cout"]/cc - T["bout"]/bc) * P_OUT
print(f"  每调用增加的费用里 输入文字占 {d_in/(d_in+d_out):.0%}，输出占 {d_out/(d_in+d_out):.0%}；视觉token占baseline每调用输入 {T['bimg']/T['bin']:.0%}")
print(f"\n[表5 分块统计] 计划1/2/3/4/5个动作的调用 {' / '.join(str(sizes[k]) for k in range(1, 6))}（>5的 {sum(v for k, v in sizes.items() if k > 5)}）；"
      f"多动作调用 {sum(v for k, v in sizes.items() if k > 1)}（{sum(v for k, v in sizes.items() if k > 1)/cc:.1%}）")
print(f"  计划后续动作 {T['planned']}；执行 F={T['fu']}；拦截 {T['ab']}；丢弃 {T['drop']}；冗余率 r={T['fu']/T['planned']:.1%}；恒等式 A-F={T['cacts']-T['fu']} vs J={cc}")
print("[表6 拦截原因]")
for k, v in reasons.most_common():
    print(f"  {v:4d} {v/T['ab']:6.1%}  {k}")
print(f"\n[5.5 理论上限] J*={seg} / A={T['bacts']} → rho*={seg/T['bacts']:.3f}；上限可省 {T["bacts"]-seg} 次；PACE比baseline少调用 {bc-cc} 次，在自己的轨迹上靠后续动作省 {T["fu"]} 次")
