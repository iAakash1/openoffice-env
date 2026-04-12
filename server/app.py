import os
import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenOfficeEnv — Agent Demo</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
  header { background: #1a1d27; border-bottom: 1px solid #2d3148; padding: 18px 32px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.2rem; font-weight: 700; color: #fff; }
  header span { font-size: 0.78rem; background: #2563eb18; color: #60a5fa; border: 1px solid #2563eb33; padding: 2px 10px; border-radius: 20px; }
  .container { max-width: 900px; margin: 0 auto; padding: 32px 20px; }
  .card { background: #1a1d27; border: 1px solid #2d3148; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
  .card h2 { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; margin-bottom: 16px; }
  .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  select { background: #0f1117; border: 1px solid #2d3148; color: #e2e8f0; padding: 10px 14px; border-radius: 8px; font-size: 0.88rem; cursor: pointer; outline: none; }
  select:focus { border-color: #2563eb; }
  button { background: #2563eb; color: #fff; border: none; padding: 10px 24px; border-radius: 8px; font-size: 0.88rem; font-weight: 600; cursor: pointer; transition: background 0.15s; }
  button:hover:not(:disabled) { background: #1d4ed8; }
  button:disabled { background: #1e2235; color: #4b5563; cursor: not-allowed; }
  .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 20px; }
  .metric { background: #0f1117; border: 1px solid #1e2235; border-radius: 8px; padding: 16px; text-align: center; }
  .metric .val { font-size: 1.7rem; font-weight: 700; color: #60a5fa; line-height: 1; }
  .metric .lbl { font-size: 0.7rem; color: #475569; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.08em; }
  .log-wrap { background: #080a10; border: 1px solid #1a1d27; border-radius: 8px; overflow: hidden; }
  .log { padding: 16px; font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace; font-size: 0.78rem; max-height: 400px; overflow-y: auto; line-height: 1.8; }
  .log .l-start { color: #34d399; }
  .log .l-step  { color: #94a3b8; }
  .log .l-act   { color: #fbbf24; }
  .log .l-pos   { color: #34d399; }
  .log .l-neg   { color: #f87171; }
  .log .l-end   { color: #60a5fa; font-weight: 700; border-top: 1px solid #1e2235; padding-top: 8px; margin-top: 4px; }
  .log .l-empty { color: #374151; font-style: italic; }
  .badge-p { color: #34d399; font-weight: 700; }
  .badge-f { color: #f87171; font-weight: 700; }
  .spin { display: inline-block; width: 13px; height: 13px; border: 2px solid #ffffff33; border-top-color: #fff; border-radius: 50%; animation: spin 0.65s linear infinite; margin-right: 8px; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @media (max-width: 600px) { .metrics { grid-template-columns: repeat(2, 1fr); } }
</style>
</head>
<body>
<header>
  <h1>🏢 OpenOfficeEnv</h1>
  <span>Hybrid RL + LLM Agent · OpenEnv Hackathon</span>
</header>
<div class="container">
  <div class="card">
    <h2>Episode Configuration</h2>
    <div class="controls">
      <select id="taskSel">
        <option value="email">📧 Email Triage — Easy</option>
        <option value="data">📊 Data Cleaning — Medium</option>
        <option value="code">🐛 Code Repair — Hard</option>
      </select>
      <button id="runBtn" onclick="runEpisode()">▶ Run Agent</button>
    </div>
    <div class="metrics" id="metrics" style="display:none">
      <div class="metric"><div class="val" id="mSteps">—</div><div class="lbl">Steps</div></div>
      <div class="metric"><div class="val" id="mScore">—</div><div class="lbl">Score</div></div>
      <div class="metric"><div class="val" id="mAvg">—</div><div class="lbl">Avg Reward</div></div>
      <div class="metric"><div class="val" id="mResult">—</div><div class="lbl">Outcome</div></div>
    </div>
  </div>
  <div class="card">
    <h2>Episode Log</h2>
    <div class="log-wrap">
      <div class="log" id="log"><div class="l-empty">Select a task and click Run Agent to begin.</div></div>
    </div>
  </div>
</div>
<script>
async function runEpisode() {
  const task = document.getElementById('taskSel').value;
  const btn  = document.getElementById('runBtn');
  const log  = document.getElementById('log');

  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span>Running...';
  log.innerHTML = '';
  document.getElementById('metrics').style.display = 'none';

  try {
    const res  = await fetch('/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({task})
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    log.innerHTML += `<div class="l-start">[START] task=${data.task} model=${data.model}</div>`;

    let total = 0;
    for (const s of data.steps) {
      total += s.reward;
      const rc = s.reward >= 0 ? 'l-pos' : 'l-neg';
      log.innerHTML +=
        `<div class="l-step">[STEP] step=${s.step} ` +
        `<span class="l-act">${s.action}</span> ` +
        `<span class="${rc}">reward=${s.reward.toFixed(2)}</span> ` +
        `done=${s.done} error=${s.error}</div>`;
    }

    const ok    = data.success;
    const badge = ok ? '<span class="badge-p">✓ PASS</span>' : '<span class="badge-f">✗ FAIL</span>';
    log.innerHTML +=
      `<div class="l-end">[END] success=${data.success} steps=${data.total_steps} ` +
      `score=${data.score.toFixed(4)} rewards=${data.rewards} ${badge}</div>`;

    document.getElementById('mSteps').textContent  = data.total_steps;
    document.getElementById('mScore').textContent  = data.score.toFixed(3);
    document.getElementById('mAvg').textContent    = data.steps.length
      ? (total / data.steps.length).toFixed(3) : '0.000';
    document.getElementById('mResult').innerHTML   = badge;
    document.getElementById('metrics').style.display = 'grid';
    log.scrollTop = log.scrollHeight;

  } catch (e) {
    log.innerHTML = `<div class="l-neg">Error: ${e.message}</div>`;
  }

  btn.disabled = false;
  btn.innerHTML = '▶ Run Agent';
}
</script>
</body>
</html>
"""


class DemoHandler(BaseHTTPRequestHandler):

    def log_message(self, *args):
        return

    def _send(self, body: bytes, ct: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(HTML.encode(), "text/html")

    def do_POST(self):
        if self.path == "/run":
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length) or b"{}")
            result = _run_episode(body.get("task", "email"))
            self._send(json.dumps(result).encode(), "application/json")
        elif self.path in ["/reset", "/"]:
            # forward to env for Phase 1 compatibility
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length) or b"{}")
            from env.core import OpenOfficeEnv
            global _phase1_env
            _phase1_env = OpenOfficeEnv(body.get("task", "email"))
            obs = _phase1_env.reset()
            self._send(json.dumps(obs.model_dump()).encode(), "application/json")
        elif self.path == "/step":
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length) or b"{}")
            if _phase1_env is None:
                self._send(b'{"error":"not initialized"}', "application/json", 400)
                return
            obs, reward, done, info = _phase1_env.step(body.get("action", "finish()"))
            resp = {"observation": obs.model_dump(), "reward": reward.value, "done": done, "info": info}
            self._send(json.dumps(resp).encode(), "application/json")
        else:
            self._send(b'{"error":"not found"}', "application/json", 404)


_phase1_env = None


def _run_episode(task_name: str) -> dict:
    from env.core import OpenOfficeEnv
    from inference import get_action

    model   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    env     = OpenOfficeEnv(task_name)
    obs     = env.reset()
    steps   = []
    history = []
    done    = False
    step_n  = 0

    while not done and step_n < 30:
        step_n += 1
        action_str = get_action(task_name, obs.model_dump(), history)
        history.append(action_str)
        obs, reward, done, info = env.step(action_str)
        err = info.get("error")
        steps.append({
            "step":   step_n,
            "action": action_str[:100],
            "reward": round(reward.value, 4),
            "done":   done,
            "error":  str(err) if err else "null",
        })

    rewards = [s["reward"] for s in steps]
    score   = max(0.02, min(0.98, sum(rewards) / len(rewards) if rewards else 0.5))
    success = done and (rewards[-1] >= 0 if rewards else False)

    return {
        "task":        task_name,
        "model":       model,
        "steps":       steps,
        "total_steps": step_n,
        "score":       score,
        "success":     success,
        "rewards":     ",".join(f"{r:.2f}" for r in rewards),
    }


def main():
    port = int(os.environ.get("PORT", 7860))
    server = HTTPServer(("0.0.0.0", port), DemoHandler)
    print(f"[DEMO] Running at http://0.0.0.0:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()