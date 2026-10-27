"""
Manual saturation labelling tool for the per-context chimera panels.

A tiny local web app -- standard library only, no Flask, no install. It shows
one context panel at a time, takes a saturation value (or "no saturation") plus an
optional low-confidence flag and notes, and appends to a CSV after EVERY entry, so the
work is never lost and can be resumed at any point.

Design choices that matter for the science:

* BLIND BY DEFAULT. The automated ``saturation_detector`` verdict is not shown
  unless ``--show-auto`` is passed, so manual labels stay independent and can
  be compared against the detector afterwards.
* The CSV is keyed on ``(comparison, context_label_8)``, the same key the
  automated saturation tables use, so the two join directly.
* Writes are atomic (temp file + os.replace), so an interrupted save cannot
  corrupt the label file.

Usage::

    python analysis_script/label_saturation.py
    python analysis_script/label_saturation.py --port 8800
    python analysis_script/label_saturation.py --labeler yanle
    python analysis_script/label_saturation.py --show-auto      # not blind

Then open http://127.0.0.1:8765 . Labels land in
``outputs/synthetic_context_8block/manual_saturation_labels.csv``.

Keyboard
    click     read the AG mean-logit off the plot at that point
    0-9 . -   or type a saturation value directly
    Enter     save and go to the next unlabelled panel
    n         mark "no saturation" and advance
    s         skip (no label) and advance
    l         toggle the low-confidence flag
    Backspace clear this panel's label
    [ / ]     previous / next panel
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = (ROOT / "outputs" / "synthetic_context_8block" / "plots"
                 / "contexts" / "context_plot_index.csv")
DEFAULT_OUT = (ROOT / "outputs" / "synthetic_context_8block"
               / "manual_saturation_labels.csv")
AUTO_CSV = (ROOT / "outputs" / "synthetic_context" / "mean_logit_saturation"
            / "chimera_mean_logit_saturation.csv")

FIELDS = ["comparison", "arm", "context_label_8", "n_F_blocks", "n_points",
          "median_mean_logit", "image", "saturation_value", "no_saturation",
          "low_confidence", "notes", "labeler", "labeled_at"]

_LOCK = threading.Lock()


# ==========================================================================
# state
# ==========================================================================

class Store:
    def __init__(self, index_csv: Path, out_csv: Path, labeler: str):
        self.out_csv = out_csv
        self.labeler = labeler
        with index_csv.open() as fh:
            self.items = list(csv.DictReader(fh))
        if not self.items:
            raise SystemExit(f"no rows in {index_csv}")
        for it in self.items:
            it["key"] = f"{it['comparison']}|{it['context_label_8']}"
        self.labels: dict[str, dict] = {}
        if out_csv.exists():
            with out_csv.open() as fh:
                for r in csv.DictReader(fh):
                    k = f"{r['comparison']}|{r['context_label_8']}"
                    self.labels[k] = r
            print(f"[resume] {len(self.labels)} existing labels in "
                  f"{out_csv.name}")

    # -- queries ---------------------------------------------------------
    def payload(self) -> dict:
        out = []
        for i, it in enumerate(self.items):
            lab = self.labels.get(it["key"], {})
            out.append({
                "i": i,
                "key": it["key"],
                "comparison": it["comparison"],
                "arm": it["arm"],
                "label": it["context_label_8"],
                "n_F_blocks": it.get("n_F_blocks", ""),
                "n_points": it.get("n_points", ""),
                "median": it.get("median_mean_logit", ""),
                "image": it["image"],
                "auto": it.get("auto", ""),
                "cal": {k: it.get(k, "") for k in
                        ("img_w", "img_h", "ax_left_px", "ax_right_px",
                         "ax_top_px", "ax_bottom_px",
                         "x_min", "x_max", "y_min", "y_max")},
                "saturation_value": lab.get("saturation_value", ""),
                "no_saturation": lab.get("no_saturation", ""),
                "low_confidence": lab.get("low_confidence", ""),
                "notes": lab.get("notes", ""),
            })
        return {"items": out, "labeler": self.labeler,
                "out_csv": str(self.out_csv.relative_to(ROOT))}

    # -- mutation --------------------------------------------------------
    def save(self, body: dict) -> dict:
        key = body.get("key", "")
        item = next((x for x in self.items if x["key"] == key), None)
        if item is None:
            raise KeyError(key)

        if body.get("clear"):
            self.labels.pop(key, None)
        else:
            self.labels[key] = {
                "comparison": item["comparison"],
                "arm": item["arm"],
                "context_label_8": item["context_label_8"],
                "n_F_blocks": item.get("n_F_blocks", ""),
                "n_points": item.get("n_points", ""),
                "median_mean_logit": item.get("median_mean_logit", ""),
                "image": item["image"],
                "saturation_value": body.get("saturation_value", ""),
                "no_saturation": "1" if body.get("no_saturation") else "",
                "low_confidence": "1" if body.get("low_confidence") else "",
                "notes": body.get("notes", ""),
                "labeler": self.labeler,
                "labeled_at": datetime.now(timezone.utc)
                              .isoformat(timespec="seconds"),
            }
        self._flush()
        return {"ok": True, "n_labeled": len(self.labels)}

    def _flush(self) -> None:
        """Atomic write: a crash mid-save cannot truncate the label file."""
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        order = {it["key"]: i for i, it in enumerate(self.items)}
        rows = sorted(self.labels.values(),
                      key=lambda r: order.get(
                          f"{r['comparison']}|{r['context_label_8']}", 1 << 30))
        fd, tmp = tempfile.mkstemp(dir=str(self.out_csv.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in FIELDS})
            os.replace(tmp, self.out_csv)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise


def attach_auto(store: Store) -> int:
    """Optionally attach the automated detector's verdict (non-blind mode)."""
    if not AUTO_CSV.exists():
        return 0
    auto: dict[str, str] = {}
    with AUTO_CSV.open() as fh:
        for r in csv.DictReader(fh):
            lab6 = r.get("context_label", "")
            auto[f"{r.get('comparison','')}|{lab6}"] = r.get(
                "classification", "")
    hit = 0
    for it in store.items:
        v = auto.get(it["key"])
        if v:
            it["auto"] = v
            hit += 1
    return hit


# ==========================================================================
# HTTP
# ==========================================================================

HTML = r"""<!doctype html>
<meta charset="utf-8"><title>Saturation labelling</title>
<style>
 :root{--ink:#111;--muted:#6b7280;--line:#e5e7eb;--accent:#2563eb;--ok:#15803d;
       --warn:#b45309}
 *{box-sizing:border-box}
 body{margin:0;font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      color:var(--ink);display:grid;grid-template-columns:1fr 330px;height:100vh}
 #left{display:flex;align-items:center;justify-content:center;background:#fafafa;
       padding:12px;overflow:auto}
 #stage{position:relative;line-height:0}
 #stage img{max-width:100%;max-height:calc(100vh - 24px);object-fit:contain;
            cursor:crosshair}
 #hline{position:absolute;left:0;right:0;height:0;border-top:1px dashed #dc2626;
        display:none;pointer-events:none}
 #dot{position:absolute;width:11px;height:11px;margin:-6px 0 0 -6px;
      border-radius:50%;background:#dc2626;border:2px solid #fff;display:none;
      pointer-events:none;box-shadow:0 0 0 1px #dc2626}
 #readout{position:absolute;background:#dc2626;color:#fff;font:11px ui-monospace,
      monospace;padding:2px 6px;border-radius:4px;display:none;pointer-events:none;
      transform:translate(8px,-50%);white-space:nowrap}
 #right{border-left:1px solid var(--line);padding:16px;overflow:auto}
 h1{font-size:15px;margin:0 0 2px}
 .sub{color:var(--muted);font-size:12px;margin-bottom:12px}
 .bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:8px 0 14px}
 .bar>div{height:100%;background:var(--ok);width:0%}
 .meta{font-size:12px;color:var(--muted);margin-bottom:12px}
 .meta b{color:var(--ink);font-weight:600}
 label{display:block;font-size:12px;color:var(--muted);margin:10px 0 4px}
 input[type=text],textarea{width:100%;padding:8px;border:1px solid var(--line);
   border-radius:6px;font:inherit}
 input#val{font-size:20px;font-weight:600}
 textarea{height:52px;resize:vertical}
 .row{display:flex;gap:6px}
 button{flex:1;padding:8px;border:1px solid var(--line);background:#fff;
   border-radius:6px;cursor:pointer;font:inherit}
 button:hover{background:#f3f4f6}
 button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
 button.active{background:var(--ink);color:#fff;border-color:var(--ink)}
 button.warn.active{background:var(--warn);color:#fff;border-color:var(--warn)}
 .hint{font-size:11px;color:var(--muted);margin-top:4px}
 .keys{margin-top:14px;font-size:11px;color:var(--muted);line-height:1.9}
 .keys kbd{background:#f3f4f6;border:1px solid var(--line);border-radius:4px;
   padding:1px 5px;font:11px ui-monospace,monospace}
 .done{color:var(--ok);font-weight:600}
 .auto{background:#fff7ed;border:1px solid #fed7aa;padding:6px 8px;
   border-radius:6px;font-size:11px;margin-bottom:10px}
 #toast{position:fixed;left:50%;bottom:18px;transform:translateX(-50%);
   background:#111;color:#fff;padding:7px 14px;border-radius:999px;font-size:12px;
   opacity:0;transition:opacity .18s;pointer-events:none}
</style>
<div id="left"><div id="stage">
  <img id="img" alt="">
  <div id="hline"></div><div id="dot"></div><div id="readout"></div>
</div></div>
<div id="right">
  <h1 id="title">&mdash;</h1>
  <div class="sub" id="pos">&mdash;</div>
  <div class="bar"><div id="prog"></div></div>
  <div id="autobox"></div>
  <div class="meta" id="meta"></div>

  <label>saturation value (AG mean-logit)</label>
  <input id="val" type="text" inputmode="decimal" autocomplete="off"
         placeholder="click the plot, or type">
  <div class="hint">click anywhere on the panel to read the y value off the axis</div>
  <div class="row" style="margin-top:8px">
    <button id="bno">no saturation <kbd>n</kbd></button>
    <button id="blow" class="warn">low confidence <kbd>l</kbd></button>
  </div>

  <label>notes</label>
  <textarea id="notes" placeholder="optional"></textarea>

  <div class="row" style="margin-top:12px">
    <button id="bprev">&larr; prev</button>
    <button id="bsave" class="primary">save &amp; next <kbd>&#9166;</kbd></button>
    <button id="bnext">next &rarr;</button>
  </div>
  <div class="row" style="margin-top:6px">
    <button id="bskip">skip <kbd>s</kbd></button>
    <button id="bclear">clear label</button>
    <button id="bjump">first unlabelled</button>
  </div>

  <div class="keys">
    <kbd>click</kbd> read y off the plot &nbsp; <kbd>&#9166;</kbd> save &amp; next<br>
    <kbd>n</kbd> no saturation &nbsp; <kbd>l</kbd> low confidence &nbsp;
    <kbd>s</kbd> skip<br>
    <kbd>[</kbd> <kbd>]</kbd> prev / next &nbsp; <kbd>&#9003;</kbd> clear<br>
    saved to <span id="out"></span>
  </div>
</div>
<div id="toast"></div>
<script>
let S={items:[],i:0,low:false};
const $=id=>document.getElementById(id);

function toast(m){const t=$("toast");t.textContent=m;t.style.opacity=1;
  clearTimeout(t._h);t._h=setTimeout(()=>t.style.opacity=0,1000);}

function labeled(it){return it.saturation_value!=="" || it.no_saturation==="1";}

function hideMarker(){["hline","dot","readout"].forEach(i=>$(i).style.display="none");}

function render(){
  const it=S.items[S.i]; if(!it)return;
  hideMarker();
  $("img").src="/img/"+it.image;
  $("title").textContent=it.comparison+"  ·  "+it.label;
  const done=S.items.filter(labeled).length;
  $("pos").innerHTML=(S.i+1)+" of "+S.items.length+
    " &nbsp;·&nbsp; <span class='done'>"+done+" labelled</span>";
  $("prog").style.width=(100*done/S.items.length)+"%";
  $("meta").innerHTML="<b>"+it.n_F_blocks+"</b> FAS blocks &nbsp;·&nbsp; n=<b>"+
    it.n_points+"</b> &nbsp;·&nbsp; median <b>"+
    (it.median?Number(it.median).toFixed(3):"—")+"</b>";
  $("autobox").innerHTML = it.auto ?
    "<div class='auto'>auto detector: <b>"+it.auto+"</b></div>" : "";
  $("val").value = it.no_saturation==="1" ? "" : (it.saturation_value||"");
  $("notes").value = it.notes||"";
  S.low = it.low_confidence==="1";
  $("blow").classList.toggle("active", S.low);
  $("bno").classList.toggle("active", it.no_saturation==="1");
  $("val").focus(); $("val").select();
}

/* click -> AG mean-logit, using the axes geometry recorded at plot time */
function onPlotClick(e){
  const it=S.items[S.i], cal=it&&it.cal;
  const img=$("img");
  if(!cal || cal.ax_top_px===""||cal.ax_top_px===undefined){
    toast("no click calibration — regenerate the plots"); return; }
  const r=img.getBoundingClientRect();
  if(!img.naturalWidth){ return; }
  const sx=r.width/img.naturalWidth, sy=r.height/img.naturalHeight;
  const px=(e.clientX-r.left)/sx, py=(e.clientY-r.top)/sy;   // natural px
  const top=+cal.ax_top_px, bot=+cal.ax_bottom_px;
  const lef=+cal.ax_left_px, rig=+cal.ax_right_px;
  if(py<top-3||py>bot+3||px<lef-3||px>rig+3){
    toast("click inside the plot area"); return; }
  const fy=(bot-py)/(bot-top), fx=(px-lef)/(rig-lef);
  const yv=(+cal.y_min)+fy*((+cal.y_max)-(+cal.y_min));
  const xv=(+cal.x_min)+fx*((+cal.x_max)-(+cal.x_min));
  $("val").value=yv.toFixed(3);
  // marker, in rendered (CSS) pixels relative to the image box
  const cx=(px*sx)+(img.offsetLeft), cy=(py*sy)+(img.offsetTop);
  $("dot").style.left=cx+"px"; $("dot").style.top=cy+"px"; $("dot").style.display="block";
  $("hline").style.top=cy+"px"; $("hline").style.display="block";
  const ro=$("readout");
  ro.textContent="y = "+yv.toFixed(3)+"   (x = "+xv.toFixed(2)+")";
  ro.style.left=cx+"px"; ro.style.top=cy+"px"; ro.style.display="block";
}
$("img").addEventListener("click",onPlotClick);

async function post(body){
  const r=await fetch("/api/label",{method:"POST",
    headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  return r.json();
}

async function save(opts={}){
  const it=S.items[S.i];
  const v=$("val").value.trim();
  const no=!!opts.no_saturation;
  if(!no && v===""){ toast("click the plot or type a value, or press n"); return false; }
  if(!no && isNaN(Number(v))){ toast("not a number"); return false; }
  await post({key:it.key,saturation_value:no?"":v,no_saturation:no,
              low_confidence:S.low,notes:$("notes").value});
  it.saturation_value=no?"":v; it.no_saturation=no?"1":"";
  it.low_confidence=S.low?"1":""; it.notes=$("notes").value;
  toast((no?"no saturation":("saved "+v))+(S.low?"  (low conf)":""));
  return true;
}

async function clearLabel(){
  const it=S.items[S.i];
  await post({key:it.key,clear:true});
  it.saturation_value="";it.no_saturation="";it.low_confidence="";it.notes="";
  toast("cleared"); render();
}

function go(d){ S.i=Math.min(S.items.length-1,Math.max(0,S.i+d)); render(); }
function nextUnlabelled(){
  const j=S.items.findIndex((x,k)=>k>S.i && !labeled(x));
  S.i = j>=0 ? j : Math.min(S.i+1,S.items.length-1); render();
}
function firstUnlabelled(){
  const j=S.items.findIndex(x=>!labeled(x));
  if(j<0){toast("all labelled");return;} S.i=j; render();
}
function toggleLow(){ S.low=!S.low; $("blow").classList.toggle("active",S.low); }

$("bsave").onclick=async()=>{ if(await save()) nextUnlabelled(); };
$("bno").onclick=async()=>{ if(await save({no_saturation:true})) nextUnlabelled(); };
$("blow").onclick=()=>{ toggleLow(); $("val").focus(); };
$("bskip").onclick=()=>go(1);
$("bprev").onclick=()=>go(-1);
$("bnext").onclick=()=>go(1);
$("bclear").onclick=clearLabel;
$("bjump").onclick=firstUnlabelled;

document.addEventListener("keydown",async e=>{
  if(e.target.id==="notes") return;
  if(e.key==="Enter"){ e.preventDefault(); if(await save()) nextUnlabelled(); return; }
  if(e.key==="n"){ e.preventDefault();
    if(await save({no_saturation:true})) nextUnlabelled(); return; }
  if(e.key==="l"){ e.preventDefault(); toggleLow(); return; }
  if(e.key==="s"){ e.preventDefault(); go(1); return; }
  if(e.key==="["){ e.preventDefault(); go(-1); return; }
  if(e.key==="]"){ e.preventDefault(); go(1); return; }
  if(e.key==="Backspace" && $("val").value===""){ e.preventDefault();
    clearLabel(); return; }
});

fetch("/api/items").then(r=>r.json()).then(d=>{
  S.items=d.items; $("out").textContent=d.out_csv;
  const j=S.items.findIndex(x=>!labeled(x));
  S.i = j>=0 ? j : 0;
  render();
});
</script>
"""


def make_handler(store: Store):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == "/":
                return self._send(200, HTML.encode(), "text/html; charset=utf-8")
            if path == "/api/items":
                with _LOCK:
                    return self._send(200, json.dumps(store.payload()).encode())
            if path.startswith("/img/"):
                rel = path[len("/img/"):]
                target = (ROOT / rel).resolve()
                # never serve outside the repository
                if not str(target).startswith(str(ROOT)) or not target.is_file():
                    return self._send(404, b"not found", "text/plain")
                return self._send(200, target.read_bytes(), "image/png")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if unquote(urlparse(self.path).path) != "/api/label":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                with _LOCK:
                    res = store.save(body)
            except KeyError as exc:
                return self._send(400, json.dumps({"error": str(exc)}).encode())
            return self._send(200, json.dumps(res).encode())

    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--labeler", default=os.environ.get("USER", "unknown"))
    ap.add_argument("--show-auto", action="store_true",
                    help="show the automated detector's verdict (NOT blind)")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args(argv)

    # Unbuffered, so the URL appears immediately even when piped or nohup'd.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    index = Path(args.index)
    if not index.exists():
        raise SystemExit(
            f"missing {index}\nRun: python analysis_script/"
            "make_chimera8_context_plots.py")

    store = Store(index, Path(args.out), args.labeler)
    if args.show_auto:
        n = attach_auto(store)
        print(f"[not blind] attached {n} automated verdicts")

    todo = sum(1 for it in store.items if it["key"] not in store.labels)
    print(f"{len(store.items)} panels, {todo} still unlabelled")
    print(f"labels -> {store.out_csv}")
    url = f"http://127.0.0.1:{args.port}"
    print(f"\n  {url}\n\nCtrl-C to stop (every label is already saved).")

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped. {len(store.labels)} labels in {store.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
