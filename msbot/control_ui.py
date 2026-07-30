# -*- coding: utf-8 -*-
"""
Browser-based control panel (stdlib only — works with the embedded-Python
exe which has no tkinter).

Start the tiny HTTP server, open the page in the user's default browser:
input target seconds, press start.  The bot waits for each start signal,
reports live status, and pauses after every win until the next input.
"""
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

_PAGE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>Minesweeper Bot</title>
<style>
 body{font-family:system-ui,'Microsoft YaHei',sans-serif;background:#1e1f24;color:#eee;
      display:flex;justify-content:center;padding-top:60px}
 .card{background:#2a2c33;padding:32px 40px;border-radius:14px;width:420px;
      box-shadow:0 6px 24px rgba(0,0,0,.4)}
 h1{font-size:20px;margin:0 0 18px}
 .row{display:flex;gap:10px;margin-bottom:16px}
 input{flex:1;padding:10px 12px;font-size:16px;border-radius:8px;border:1px solid #444;
      background:#1b1c20;color:#eee}
 button{padding:10px 26px;font-size:16px;border:0;border-radius:8px;cursor:pointer;
      background:#4f8cff;color:#fff}
 button:hover{background:#3d78e8}
 #status{font-size:14px;line-height:1.9;white-space:pre-wrap}
 .tag{display:inline-block;padding:2px 10px;border-radius:10px;font-size:12px;margin-right:6px}
 .playing{background:#2d5b2d}.waiting{background:#6b5b23}.won{background:#245a6b}
 .lost{background:#6b2424}.stopped{background:#555}
</style></head><body>
<div class="card">
 <h1>扫雷机器人控制台</h1>
 <div class="row">
  <input id="t" type="number" min="5" step="1" placeholder="目标时间（秒）" autofocus>
  <button onclick="start()">开始</button>
 </div>
 <div id="status">loading...</div>
</div>
<script>
async function start(){
  const t=document.getElementById('t').value;
  if(!t||t<5){alert('请输入 >=5 的秒数');return}
  await fetch('/start',{method:'POST',body:'t='+t});
}
document.getElementById('t').addEventListener('keydown',e=>{if(e.key==='Enter')start()});
async function poll(){
  try{
    const s=await (await fetch('/state')).json();
    document.getElementById('status').innerHTML=
      `<span class="tag ${s.status}">${s.status}</span>`+
      (s.target?`目标: ${s.target}s<br>`:'')+s.detail;
  }catch(e){}
  setTimeout(poll,1200);
}
poll();
</script></body></html>"""


class ControlServer:
    """Threaded status/control server.  The player blocks on
    wait_for_start(); the web page provides the target time."""

    def __init__(self, port=0):
        self.state = {"status": "waiting", "target": None, "detail": ""}
        self._event = threading.Event()
        self._pending = None
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/state"):
                    body = json.dumps(outer.state).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                else:
                    body = _PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                data = self.rfile.read(n).decode()
                if self.path.startswith("/start"):
                    try:
                        t = float(data.split("=", 1)[1])
                    except (ValueError, IndexError):
                        t = None
                    if t and t >= 1:
                        outer._pending = t
                        outer._event.set()
                self.send_response(302)
                self.send_header("Location", "/")
                self.end_headers()

        self._srv = HTTPServer(("127.0.0.1", port), H)
        self.port = self._srv.server_address[1]
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.port}/"

    def open_browser(self):
        try:
            webbrowser.open(self.url)
        except Exception:
            pass

    # ----------------------------------------------------------
    def wait_for_start(self):
        """Block until the user submits a target time.  Returns seconds."""
        self.state["status"] = "waiting"
        self._event.clear()
        self._event.wait()
        t, self._pending = self._pending, None
        self.state["status"] = "playing"
        self.state["target"] = t
        return t

    def set(self, status=None, detail=None):
        if status:
            self.state["status"] = status
        if detail is not None:
            self.state["detail"] = detail
