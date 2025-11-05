#!/usr/bin/env python3
# run.py
import os
import sys
import time
import json
import logging
import datetime
from typing import Dict, List, Tuple, Optional

import requests
import pytz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/118.0.0.0 Safari/537.36"
)

WAKING_UP_KEYWORDS = [
    "waking up",
    "is waking up",
    "space is loading",
    "loading space",
    "container is starting",
    "building",
    "launching",
    "runtime is starting",
    "runtime starting",
    "starting",
]

ERROR_PAGE_KEYWORDS = [
    "application error",
    "runtime error",
    "traceback (most recent call last)",
    "build failed",
    "build error",
    "error id:",
    "internal server error",
]

DEFAULT_TZ = "Asia/Shanghai"


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name, default)
    if v is None:
        return None
    v = v.strip()
    return v if v else default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def now_str(tz: str = DEFAULT_TZ) -> str:
    return datetime.datetime.now(pytz.timezone(tz)).strftime("%Y-%m-%d %H:%M:%S")


def set_github_output(key: str, value: str) -> None:
    out_path = os.environ.get("GITHUB_OUTPUT")
    if not out_path:
        return
    try:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")
    except Exception as e:
        logging.warning(f"Failed to write GITHUB_OUTPUT: {e}")


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def space_url(username: str, space: str) -> str:
    return f"https://{username}-{space}.hf.space"


def runtime_url(username: str, space: str) -> str:
    return f"https://huggingface.co/api/spaces/{username}/{space}/runtime"


def restart_url(username: str, space: str) -> str:
    return f"https://huggingface.co/api/spaces/{username}/{space}/restart"


def request_get(url: str, headers: Dict[str, str], timeout: int = 45) -> requests.Response:
    return requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)


def request_post(url: str, headers: Dict[str, str], json_body: Optional[dict] = None, timeout: int = 60) -> requests.Response:
    return requests.post(url, headers=headers, json=json_body, timeout=timeout)


def read_runtime(username: str, space: str, token: Optional[str]) -> Tuple[Optional[str], Optional[dict]]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = request_get(runtime_url(username, space), headers, timeout=30)
        if r.status_code >= 400:
            return None, None
        data = r.json()
        stage = str(data.get("stage") or data.get("status") or "").upper()
        return stage, data
    except Exception:
        return None, None


def ping_space(username: str, space: str) -> Tuple[int, str, float]:
    headers = {"User-Agent": USER_AGENT}
    url = space_url(username, space)
    t0 = time.time()
    try:
        r = request_get(url, headers, timeout=60)
        dt = time.time() - t0
        return r.status_code, (r.text or ""), dt
    except Exception as e:
        dt = time.time() - t0
        return 599, str(e), dt


def classify_from_html(status_code: int, html: str) -> str:
    if status_code >= 500:
        return "ERROR"
    low = (html or "").lower()
    if any(k in low for k in ERROR_PAGE_KEYWORDS):
        return "ERROR"
    if any(k in low for k in WAKING_UP_KEYWORDS):
        return "WAKING_UP"
    if 200 <= status_code < 300:
        return "RUNNING"
    if status_code >= 400:
        return "ERROR"
    return "UNKNOWN"


def normalize_stage(stage: Optional[str]) -> Optional[str]:
    if not stage:
        return None
    s = stage.upper()
    if s in {"RUNNING", "RUNNING_BUILDING"}:
        return "RUNNING"
    if s in {"BUILDING", "STARTING", "RUNTIME_STARTING"}:
        return "WAKING_UP"
    if s in {"SLEEPING", "STOPPED"}:
        return "SLEEPING"
    if "ERROR" in s or "FAIL" in s:
        return "ERROR"
    return s


def wait_until_running(
    username: str,
    space: str,
    token: Optional[str],
    max_wait: int,
    poll: int = 10,
) -> Tuple[str, float]:
    t0 = time.time()
    last_state = "UNKNOWN"
    while True:
        stage, _ = read_runtime(username, space, token)
        norm = normalize_stage(stage) if stage else None
        if norm == "RUNNING":
            return "RUNNING", time.time() - t0
        if norm == "ERROR":
            return "ERROR", time.time() - t0
        if time.time() - t0 > max_wait:
            return last_state if last_state != "UNKNOWN" else (norm or "TIMEOUT"), time.time() - t0
        status_code, html, _ = ping_space(username, space)
        last_state = classify_from_html(status_code, html)
        time.sleep(poll)


def restart_space(username: str, space: str, token: str, wait_after: int = 600) -> Tuple[bool, float, str]:
    t0 = time.time()
    headers = {"User-Agent": USER_AGENT, "Authorization": f"Bearer {token}"}
    try:
        r = request_post(restart_url(username, space), headers, timeout=60)
        if r.status_code >= 400:
            return False, time.time() - t0, f"restart http {r.status_code}"
    except Exception as e:
        return False, time.time() - t0, f"restart exception: {e}"
    state, dt = wait_until_running(username, space, token, max_wait=wait_after, poll=15)
    ok = state == "RUNNING"
    return ok, time.time() - t0, state


def generate_html_report(results: List[dict], report_file: str = "docs/index.html") -> str:
    ensure_dir(report_file)
    ts = now_str()
    entry_lines: List[str] = []
    entry_lines.append(f'<div class="log-entry"><span class="timestamp">{ts}</span><br>')
    for r in results:
        icon = "✅" if r.get("success") else "❌"
        cls = "success" if r.get("success") else "failure"
        space = r["space"]
        action = r["action"]
        state = r["state"]
        dur = f'{r["duration"]:.1f}s'
        note = r.get("note") or ""
        entry_lines.append(
            f"{space}: <span class='{cls}'>{icon}</span> [{action} -> {state}] ({dur}) {note}<br>"
        )
    entry_lines.append("</div>")
    new_entry = "".join(entry_lines)

    base = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>HF Spaces 状态</title>
<style>
body{font-family:sans-serif;max-width:900px;margin:24px auto;padding:0 12px;}
.log-entry{margin:12px 0;border:1px solid #ddd;padding:10px;border-radius:8px;background:#fafafa;}
.timestamp{font-weight:bold;color:#333;}
.success{color:#1a7f37;}
.failure{color:#d1242f;}
h1{font-size:20px}
</style></head><body>
<h1>Hugging Face Spaces 状态</h1>
<div id="content"></div>
</body></html>
"""
    content = base
    if os.path.exists(report_file):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            content = base

    marker = '<div id="content">'
    if marker in content:
        pos = content.find(marker) + len(marker)
        updated = content[:pos] + new_entry + content[pos:]
    else:
        updated = base.replace('<div id="content"></div>', f'<div id="content">{new_entry}</div>')

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(updated)
    return ts


def update_readme(results: List[dict], ts: str, readme_file: str = "README.md") -> None:
    lines: List[str] = []
    lines.append("# HF Space 状态报告")
    lines.append("")
    lines.append(f"- 最近更新: {ts}")
    lines.append("- 说明: RUNNING=运行中，WAKING_UP=唤醒中，ERROR=错误触发重建")
    lines.append("")
    lines.append("| Space | 动作 | 状态 | 成功 | 耗时 | 备注 |")
    lines.append("|---|---|---|---:|---:|---|")
    for r in results:
        ok = "✅" if r.get("success") else "❌"
        lines.append(
            f"| {r['space']} | {r['action']} | {r['state']} | {ok} | {r['duration']:.1f}s | {r.get('note','')} |"
        )
    txt = "\n".join(lines) + "\n"

    try:
        if os.path.exists(readme_file):
            with open(readme_file, "r", encoding="utf-8") as f:
                old = f.read()
        else:
            old = ""
        top = txt + "\n---\n\n" + old
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(top)
    except Exception as e:
        logging.warning(f"Update README failed: {e}")


def main() -> int:
    hf_token = env_str("HF_TOKEN")
    username = env_str("USERNAME")
    space_list_str = env_str("SPACE_LIST", "")
    global_timeout_seconds = env_int("GLOBAL_TIMEOUT_SECONDS", 1800)
    wakeup_wait_seconds = env_int("WAKEUP_WAIT_SECONDS", 240)
    between_requests_sleep = env_int("BETWEEN_REQUESTS_SECONDS", 5)

    if not username or not space_list_str:
        logging.error("环境变量 USERNAME 或 SPACE_LIST 缺失。")
        return 2

    spaces = [s.strip() for s in space_list_str.split(",") if s.strip()]
    logging.info(f"将处理用户 {username} 的 Spaces: {spaces}")

    overall_t0 = time.time()
    results: List[dict] = []
    any_failure = False

    for sp in spaces:
        if time.time() - overall_t0 > global_timeout_seconds:
            results.append({
                "space": sp,
                "action": "跳过",
                "state": "TIMEOUT",
                "success": False,
                "duration": 0.0,
                "note": "全局超时",
            })
            any_failure = True
            continue

        logging.info(f"检查 {sp} ...")
        status_code, html, dt_ping = ping_space(username, sp)
        stage, _ = read_runtime(username, sp, hf_token)
        norm_stage = normalize_stage(stage) if stage else None

        if norm_stage == "RUNNING":
            results.append({
                "space": sp,
                "action": "检查",
                "state": "RUNNING",
                "success": True,
                "duration": dt_ping,
                "note": "",
            })
            time.sleep(between_requests_sleep)
            continue

        if norm_stage == "ERROR":
            logging.warning(f"{sp} 处于 ERROR，将尝试重建")
            if not hf_token:
                results.append({
                    "space": sp,
                    "action": "重建",
                    "state": "ERROR",
                    "success": False,
                    "duration": 0.0,
                    "note": "缺少 HF_TOKEN，无法重建",
                })
                any_failure = True
                time.sleep(between_requests_sleep)
                continue
            ok, dt_rebuild, final_state = restart_space(username, sp, hf_token, wait_after=max(480, wakeup_wait_seconds))
            results.append({
                "space": sp,
                "action": "重建",
                "state": final_state,
                "success": ok,
                "duration": dt_rebuild,
                "note": "",
            })
            if not ok:
                any_failure = True
            time.sleep(between_requests_sleep)
            continue

        cls = classify_from_html(status_code, html)
        if cls in ("RUNNING", "WAKING_UP"):
            logging.info(f"{sp} 触发保活/唤醒: {cls}，等待至运行中（最多 {wakeup_wait_seconds}s）")
            final_state, dt_wait = wait_until_running(username, sp, hf_token, max_wait=wakeup_wait_seconds, poll=10)
            state_out = final_state if final_state in ("RUNNING", "ERROR") else "WAKING_UP"
            success = state_out != "ERROR"
            results.append({
                "space": sp,
                "action": "保活",
                "state": state_out,
                "success": success,
                "duration": dt_ping + dt_wait,
                "note": "",
            })
            if not success:
                if hf_token:
                    logging.warning(f"{sp} 唤醒失败，尝试重建")
                    ok, dt_rebuild, final_state2 = restart_space(username, sp, hf_token, wait_after=max(480, wakeup_wait_seconds))
                    results.append({
                        "space": sp,
                        "action": "重建",
                        "state": final_state2,
                        "success": ok,
                        "duration": dt_rebuild,
                        "note": "",
                    })
                    if not ok:
                        any_failure = True
                else:
                    any_failure = True
            time.sleep(between_requests_sleep)
            continue

        if cls == "ERROR":
            logging.warning(f"{sp} 页面判定 ERROR，将尝试重建")
            if not hf_token:
                results.append({
                    "space": sp,
                    "action": "重建",
                    "state": "ERROR",
                    "success": False,
                    "duration": 0.0,
                    "note": "缺少 HF_TOKEN，无法重建",
                })
                any_failure = True
            else:
                ok, dt_rebuild, final_state = restart_space(username, sp, hf_token, wait_after=max(480, wakeup_wait_seconds))
                results.append({
                    "space": sp,
                    "action": "重建",
                    "state": final_state,
                    "success": ok,
                    "duration": dt_rebuild,
                    "note": "",
                })
                if not ok:
                    any_failure = True
            time.sleep(between_requests_sleep)
            continue

        results.append({
            "space": sp,
            "action": "检查",
            "state": norm_stage or cls or "UNKNOWN",
            "success": cls == "RUNNING",
            "duration": dt_ping,
            "note": "",
        })
        if cls != "RUNNING":
            logging.info(f"{sp} 状态不明确: {norm_stage} / {cls}")

        time.sleep(between_requests_sleep)

    ts = generate_html_report(results, report_file="docs/index.html")
    try:
        update_readme(results, ts, readme_file="README.md")
    except Exception as e:
        logging.warning(f"README 更新失败: {e}")

    exit_code = 1 if any_failure else 0
    set_github_output("exit_code", str(exit_code))
    logging.info(f"完成。退出码: {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
