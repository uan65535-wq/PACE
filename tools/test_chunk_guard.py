import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mind2web_agent_chunk_deepseek as m
from playwright.sync_api import sync_playwright

C = lambda x, y, d="": {"type": "CLICK", "x": x, "y": y, "description": d}
T = lambda x, y, t: {"type": "TYPE", "x": x, "y": y, "text": t, "description": ""}

# ---- _sanitize_chunk ----
S = m._sanitize_chunk
assert len(S(C(1, 1), [T(2, 2, "a"), C(3, 3)])) == 3
assert len(S(C(1, 1), [{"type": "SCROLL"}, C(3, 3)])) == 2           # SCROLL截在自己之后
assert len(S(C(1, 1), [{"type": "TASK_COMPLETE", "answer": "x"}])) == 1  # TC不能跟在后面
assert len(S({"type": "TASK_COMPLETE", "answer": "x"}, [C(1, 1)])) == 1
assert len(S({"type": "PRESS_KEY", "key": "Enter"}, [C(1, 1)])) == 1
assert len(S(C(1, 1), [{"type": "CLICK", "x": 5}, C(3, 3)])) == 1     # 缺y，截断
assert len(S(C(1, 1), [C(i, i) for i in range(10)])) == (11 if m.CHUNK_MAX_ACTIONS is None else m.CHUNK_MAX_ACTIONS)
assert "{max_actions}" not in m.SYSTEM_PROMPT and "1到10个动作" in m.SYSTEM_PROMPT
print("sanitize ok")

HTML = """<html><body style="margin:0;font:16px sans-serif">
<div style="position:absolute;left:100px;top:100px">
 <input id=city placeholder="City" style="width:300px;height:30px"
   oninput="document.getElementById('sug').style.display='block'">
 <div id=sug style="display:none;position:absolute;top:32px;left:0;width:300px;height:120px;background:#eee;z-index:9">
   <div role=option>Los Angeles, CA</div></div>
</div>
<input id=name placeholder="Name" style="position:absolute;left:100px;top:180px;width:300px;height:30px">
<input id=email placeholder="Email" style="position:absolute;left:100px;top:240px;width:300px;height:30px">
<button style="position:absolute;left:100px;top:300px" onclick="location.hash='done'">Submit</button>
</body></html>"""

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1280, "height": 800})
    page.set_content(HTML)
    # 场景1：填普通表单 name -> email -> submit，全部应通过
    chunk = [T(250, 195, "Alice"), T(250, 255, "a@b.c"), C(130, 310)]
    ctx = m._capture_chunk_context(page, chunk[1:])
    m.execute_action(page, chunk[0])
    for i, a in enumerate(chunk[1:]):
        page.wait_for_timeout(m.CHUNK_SETTLE_MS)
        ok, why = m._chunk_guard(page, ctx, a, i)
        assert ok, why
        m.execute_action(page, a)
    print("scenario1 plain form: all passed")
    # 场景2：先在City里输入（弹出建议盖住Name），下一个动作填Name应被拦下
    page.set_content(HTML)
    chunk = [T(250, 115, "Los"), T(250, 195, "Alice")]
    ctx = m._capture_chunk_context(page, chunk[1:])
    m.execute_action(page, chunk[0])
    page.wait_for_timeout(m.CHUNK_SETTLE_MS)
    ok, why = m._chunk_guard(page, ctx, chunk[1], 0)
    assert not ok, "应该被拦下"
    print("scenario2 autocomplete covers next field -> aborted:", why)
    # 场景3：点Submit真实跳转到另一个URL后，再有动作应被拦下
    page.route("http://t.local/**", lambda r: r.fulfill(content_type="text/html",
        body=HTML.replace("location.hash='done'", "location.href='http://t.local/next'")))
    page.goto("http://t.local/form")
    chunk = [C(130, 310), T(250, 195, "x")]
    ctx = m._capture_chunk_context(page, chunk[1:])
    m.execute_action(page, chunk[0])
    page.wait_for_timeout(m.CHUNK_SETTLE_MS)
    ok, why = m._chunk_guard(page, ctx, chunk[1], 0)
    assert not ok, page.url
    print("scenario3 navigated -> aborted:", why, page.url)
    b.close()
print("ALL OK")

# ===== v3新增测试 =====
from playwright.sync_api import sync_playwright
FORM = """<html><body style="margin:0;font:16px sans-serif">
<form role="search" style="position:absolute;left:50px;top:50px;width:1000px;height:500px;background:#fafafa">
 <input id=where placeholder="Where" style="position:absolute;left:20px;top:20px;width:300px;height:30px">
 <input id=who placeholder="Who" style="position:absolute;left:20px;top:100px;width:300px;height:30px">
 <div id=extra style="position:absolute;left:500px;top:20px"></div>
 <div id=whobox style="position:absolute;left:20px;top:260px;width:300px;height:40px;border:1px solid #999">Add guests</div>
 <button type=button style="position:absolute;left:20px;top:180px" onclick="document.getElementById('extra').innerText='Nearby: Paris, London, Tokyo, Berlin'">Expand</button>
</form>
<div style="position:absolute;left:700px;top:700px;width:100px;height:20px">blank</div>
</body></html>"""
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1280, "height": 800})
    # 场景4：点击后表单里别处多了文字，目标输入框没被挡住——v3应放行（v2会因FORM文字变长误拦）
    page.set_content(FORM)
    chunk = [C(90, 245), C(170, 330)]   # 第二个目标是表单里的普通div（Airbnb的"添加人数"就是这样）
    ctx = m._capture_chunk_context(page, chunk[1:])
    m.execute_action(page, chunk[0])
    page.wait_for_timeout(m.CHUNK_SETTLE_MS)
    ok, why = m._chunk_guard(page, ctx, chunk[1], 0, chunk[0])
    assert ok, why
    print("scenario4 container text changed elsewhere -> passed (no false abort)")
    # 场景5a：TYPE确实输进去了，后面的回车应放行
    page.set_content(FORM)
    chunk = [T(170, 85, "Paris"), {"type": "PRESS_KEY", "key": "Enter", "description": ""}]
    ctx = m._capture_chunk_context(page, chunk[1:])
    m.execute_action(page, chunk[0])
    ok, why = m._chunk_guard(page, ctx, chunk[1], 0, chunk[0])
    assert ok, why
    print("scenario5a TYPE landed -> passed")
    # 场景5b：TYPE点在空白处没输进去，后面的回车应被拦下
    page.set_content(FORM)
    chunk = [T(750, 710, "Paris"), {"type": "PRESS_KEY", "key": "Enter", "description": ""}]
    ctx = m._capture_chunk_context(page, chunk[1:])
    m.execute_action(page, chunk[0])
    ok, why = m._chunk_guard(page, ctx, chunk[1], 0, chunk[0])
    assert not ok, "应该被拦下"
    print("scenario5b TYPE missed -> aborted:", why)
    # 场景5c（v3.1）：输入框在Shadow DOM里，输入成功应放行
    page.set_content("""<html><body><my-search></my-search><script>
      customElements.define('my-search', class extends HTMLElement { constructor(){ super();
        this.attachShadow({mode:'open'}).innerHTML='<input style="position:absolute;left:100px;top:40px;width:400px;height:30px">'; } });
    </script></body></html>""")
    chunk = [T(300, 55, "https://www.discogs.com/submissions"), {"type": "PRESS_KEY", "key": "Enter", "description": ""}]
    ctx = m._capture_chunk_context(page, chunk[1:]); m.execute_action(page, chunk[0])
    ok, why = m._chunk_guard(page, ctx, chunk[1], 0, chunk[0]); assert ok, why
    print("scenario5c shadow-DOM input typed -> passed")
    # 场景5d（v3.1）：输入框上盖着一层透明元素、焦点被脚本移走，按坐标找到输入框，应放行
    page.set_content("""<html><body>
      <input id=q style="position:absolute;left:100px;top:40px;width:400px;height:30px" oninput="setTimeout(()=>document.body.focus(),0)">
      <div style="position:absolute;left:100px;top:40px;width:400px;height:30px;pointer-events:none"></div>
      <button id=other style="position:absolute;left:0;top:0">x</button>
    </body></html>""")
    chunk = [T(300, 55, "black swimsuit"), {"type": "PRESS_KEY", "key": "Enter", "description": ""}]
    ctx = m._capture_chunk_context(page, chunk[1:]); m.execute_action(page, chunk[0])
    page.evaluate("() => document.activeElement.blur()")
    ok, why = m._chunk_guard(page, ctx, chunk[1], 0, chunk[0]); assert ok, why
    print("scenario5d focus lost but input under point has text -> passed")
    # 场景6：页面稳定等待（chunk里已删掉这段代码时跳过）
    if hasattr(m, '_wait_for_page_settle'):
        page.set_content("<html><body><p>static</p></body></html>")
        t_static = m._wait_for_page_settle(page)
        page.set_content("""<html><body><div id=d></div><script>
          let n=0; const h=setInterval(()=>{document.getElementById('d').innerHTML+='<p>item '+(n++)+'</p>'; if(n>12) clearInterval(h);},100);
        </script></body></html>""")
        t_dyn = m._wait_for_page_settle(page)
        assert t_static < 1500 and t_dyn >= 1200, (t_static, t_dyn)
        print(f"scenario6 settle: static {t_static}ms, still-loading {t_dyn}ms")
    b.close()
print("V3 ALL OK")
