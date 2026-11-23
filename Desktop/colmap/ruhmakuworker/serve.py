#!/usr/bin/env python3
"""
rumahku reconstruction backend — a zero-dependency job service that trains a
Gaussian splat from an uploaded capture using the Brush CLI on a GPU.

Stateless job model (serverless-ready): each job is a pure function of its
uploaded dataset. Endpoints:

  POST /jobs?iters=2000&max_res=1024   body = dataset.zip  -> {"job_id": "..."}
      the zip must contain a nerfstudio dataset (transforms.json + images/,
      optional seed.ply), at the root or in a single subfolder.
      ?refine=colmap runs COLMAP pose-prior refinement first (denser SfM init +
      refined poses, ~+0.3 dB / +0.04 SSIM; see backend/colmap_refine.py) —
      Brush trainer only, adds ~5 min.
  GET  /jobs/{id}                       -> {"state": queued|running|done|error,
                                            "elapsed": s, "iter": n, ...}
  GET  /jobs/{id}/result                -> the exported .ply (octet-stream)
  GET  /health                          -> {"ok": true}

Runs the SAME Brush trainer as the on-device app, just on a real GPU. Uses only
the Python standard library so it runs anywhere python3 is present; the design
(stateless per-job, one binary) ports directly to a serverless GPU container.

Env: BRUSH_CLI (path to the brush-cli binary), JOBS_DIR, PORT.
"""
import signal
import glob
import hmac
import io
import json
import os
from pathlib import Path
import queue
import re
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from something.main import run_pipeline

BRUSH = os.environ.get("BRUSH_CLI", os.path.expanduser("~/rumahku/brush-cli"))
JOBS_DIR = os.environ.get("JOBS_DIR", os.path.expanduser("~/rumahku/jobs"))
PORT = int(os.environ.get("PORT", "8000"))
# Listen address. Defaults to all interfaces (correct for the container/cloud
# deploy behind its own network controls), but the laptop deploy should set
# BIND to the VPN address (e.g. the NetBird IP) so :8000 is NOT reachable from
# the plain LAN — only over the encrypted mesh.
BIND = os.environ.get("BIND", "0.0.0.0")
# How many jobs may train at once. GPU training is memory-bound, so the default
# is 1 (queue the rest) — bump it if the box has spare VRAM / multiple GPUs.
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "1"))
# Upload guards: cap the compressed body, the total uncompressed size, and the
# entry count so a huge or zip-bomb upload can't OOM the process or fill the disk.
MAX_UPLOAD_BYTES = int(os.environ.get(
    "MAX_UPLOAD_BYTES", str(3 * 1024**3)))       # 3 GiB compressed
MAX_UNCOMPRESSED = int(os.environ.get(
    "MAX_UNCOMPRESSED", str(12 * 1024**3)))      # 12 GiB extracted
MAX_ENTRIES = int(os.environ.get("MAX_ENTRIES", "20000"))
# Clamp training params so one request can't monopolise the single GPU worker
# forever or drive the rasteriser to OOM the card.
ITERS_MAX = int(os.environ.get("ITERS_MAX", "60000"))
RES_MAX = int(os.environ.get("RES_MAX", "2048"))
# Bearer-token auth. A request to /jobs is authorized if its `Authorization:
# Bearer <token>` matches RUMAHKU_TOKEN (a bootstrap env token) OR any *active*
# credential in CREDS_FILE (named, per-device tokens managed by the credctl TUI,
# backend/rumahku_credctl.py). When neither exists the backend is OPEN (the
# single-user default). /health is always open so a monitor can probe it.
AUTH_TOKEN = os.environ.get("RUMAHKU_TOKEN", "")
CREDS_FILE = os.environ.get(
    "RUMAHKU_CREDS", os.path.expanduser("~/rumahku/credentials.json"))
# Tokens are cached and only re-read when CREDS_FILE's mtime changes, so the TUI's
# add/revoke takes effect on the next request without restarting the server.
_CREDS_CACHE = {"mtime": None, "tokens": frozenset(), "broken": False}
# Job-dir retention. Each finished job leaves ~0.2-2 GB under JOBS_DIR (images +
# ns_out training artifacts + colmap); the result .ply is downloaded once, so
# finished dirs are pure clutter and grow unbounded. The reaper keeps the newest
# JOB_RETAIN_MAX finished dirs and additionally deletes any finished dir older
# than JOB_RETAIN_HOURS. A queued/running job is never touched. Tune via env;
# set JOB_RETAIN_MAX absurdly high to effectively disable count-based reaping.
JOB_RETAIN_MAX = int(os.environ.get("JOB_RETAIN_MAX", "20"))
JOB_RETAIN_HOURS = float(os.environ.get("JOB_RETAIN_HOURS", "168"))
REAP_EVERY_S = float(os.environ.get("JOB_REAP_EVERY_S", "3600"))
# When this backend runs inside a distrobox, the container's Vulkan falls back to
# CPU (the NVIDIA userspace only works on the host), so training is ~20x slower.
# Run brush-cli on the HOST via distrobox-host-exec instead — paths are identical
# (the home is a shared mount) and the host's NVIDIA Vulkan drives the GPU.
# Set HOST_EXEC="" to force in-container execution.
HOST_EXEC = os.environ.get(
    "HOST_EXEC", shutil.which("distrobox-host-exec") or "")
# nerfstudio (Splatfacto/gsplat) runs from its official Docker image on the GPU
# via podman — it has CUDA + gsplat prebuilt, avoiding an in-container install.
NERF_IMAGE = os.environ.get(
    "NERF_IMAGE", "ghcr.io/nerfstudio-project/nerfstudio:latest")

# Optional COLMAP pose-prior refinement (see backend/colmap_refine.py), enabled
# per-request with ?refine=colmap. It runs in-process (colmap + numpy live in the
# container; only Brush needs the host), producing a refined SfM-seeded model
# that Brush trains on for higher quality.
REFINE_SCRIPT = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "colmap_refine.py")
os.makedirs(JOBS_DIR, exist_ok=True)

JOBS = {}            # job_id -> {state, queued_at, started, iter, total, error, ply}
LOCK = threading.Lock()
WORK_Q = queue.Queue()   # (job_id, ds_dir, iters, max_res) awaiting a worker
ITER_RE = re.compile(rb"iter[=\s]+(\d+)")            # Brush:      "iter=1234"
# Splatfacto/ns-train prints a live rich table whose rows look like
# "1230 (12.30%)   4.5 ms   ...". Match the leading step + its "(NN.NN%)" so we
# don't pick up the timing/gaussian-count columns. The two patterns are disjoint
# (a Brush log has no "(NN.NN%)"; an ns-train log has no "iter="), so a job's log
# only ever matches one.
# Splatfacto: "1230 (12.30%)"
STEP_RE = re.compile(rb"(\d+)\s*\(\d+(?:\.\d+)?%\)")
PSNR_RE = re.compile(rb"PSNR\s+([\d.]+)\s+SSIM\s+([\d.]+)")


def _active_tokens():
    """The set of bearer tokens that authorize a request: RUMAHKU_TOKEN (if set)
    plus every active credential in CREDS_FILE. Re-reads the file only when its
    mtime changes. Empty set => the backend is open."""
    tokens = set()
    if AUTH_TOKEN:
        tokens.add(AUTH_TOKEN)
    try:
        mtime = os.path.getmtime(CREDS_FILE)
    except OSError:
        # file absent => legitimately no creds
        _CREDS_CACHE["broken"] = False
        return tokens
    if _CREDS_CACHE["mtime"] != mtime:
        try:
            with open(CREDS_FILE) as f:
                fresh = set()
                for c in json.load(f).get("credentials", []):
                    if c.get("active", True) and c.get("token"):
                        fresh.add(c["token"])
            # atomic dict writes under the GIL
            _CREDS_CACHE["mtime"] = mtime
            _CREDS_CACHE["tokens"] = frozenset(fresh)
            _CREDS_CACHE["broken"] = False
        except (OSError, ValueError, TypeError):
            # Present but unreadable/malformed: KEEP the last-known-good tokens and
            # flag broken (don't advance mtime, so we retry next request). _authed
            # then fails CLOSED instead of silently reopening the backend.
            _CREDS_CACHE["broken"] = True
    return tokens | _CREDS_CACHE["tokens"]


def _dir_size(path):
    """Bytes on disk under `path` (best-effort; skips vanished/unreadable files)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _job_terminal(job_id):
    """True if the dir for `job_id` is safe to delete: its job has finished, or
    it's an orphan from a previous run (a restart clears the in-memory JOBS, so
    anything on disk with no live entry is left over from before)."""
    with LOCK:
        j = JOBS.get(job_id)
    if j is None:
        return True
    return j.get("state") in ("done", "error", "cancelled")


def _reap_jobs():
    """Bound JOBS_DIR growth: keep the newest JOB_RETAIN_MAX finished job dirs and
    delete any finished dir older than JOB_RETAIN_HOURS. Never touches a
    queued/running job. Best-effort; logs each deletion and the total freed."""
    try:
        entries = []
        for name in os.listdir(JOBS_DIR):
            p = os.path.join(JOBS_DIR, name)
            if os.path.isdir(p) and _job_terminal(name):
                try:
                    entries.append((os.path.getmtime(p), p, name))
                except OSError:
                    pass
    except OSError:
        return
    # newest first; index >= JOB_RETAIN_MAX is over cap
    entries.sort(reverse=True)
    now = time.time()
    freed = 0
    for i, (mtime, p, name) in enumerate(entries):
        over_cap = i >= JOB_RETAIN_MAX
        too_old = (now - mtime) > JOB_RETAIN_HOURS * 3600
        if not (over_cap or too_old):
            continue
        sz = _dir_size(p)
        shutil.rmtree(p, ignore_errors=True)
        with LOCK:
            JOBS.pop(name, None)
        freed += sz
        print("reaped job %s (%d MB, %s)" % (
            name, sz >> 20, "cap" if over_cap else "age"), flush=True)
    if freed:
        print("reaper freed %d MB (kept newest %d)" %
              (freed >> 20, JOB_RETAIN_MAX), flush=True)


def _reaper_loop():
    while True:
        try:
            _reap_jobs()
        except Exception as e:  # noqa: BLE001 — never let the reaper kill the thread
            print("reaper error: %s" % e, flush=True)
        time.sleep(REAP_EVERY_S)


def _worker():
    """Pull queued jobs and train them one at a time (per worker), so many
    concurrent uploads don't thrash the GPU — they queue instead."""
    while True:
        job_id, ds_dir, iters, max_res, eval_split, trainer, refine = WORK_Q.get()
        try:
            with LOCK:
                j = JOBS.get(job_id)
                if j is None or j.get("state") in ("error", "cancelled"):
                    continue  # gone/cancelled while queued
                j.update(state="running", started=time.time())
            _run_job(job_id, ds_dir)
        finally:
            WORK_Q.task_done()


def _queue_position(job_id: str) -> int:
    """1-based position of a queued job among all queued jobs (0 if running)."""
    with LOCK:
        me = JOBS.get(job_id, {})
        if me.get("state") != "queued":
            return 0
        ahead = sum(
            1 for j in JOBS.values()
            if j.get("state") == "queued" and j.get("queued_at", 0) < me.get("queued_at", 0)
        )
    return ahead + 1


def _colmap_refine(job_id: str, ds_dir: str):
    """Run COLMAP pose-prior refinement before training. Returns the refined
    colmap-model dir (with images/ + sparse/0), or None to fall back to the raw
    ARCore poses. Runs in-process (colmap + numpy live in the container). The
    process is tracked so a DELETE can cancel it mid-refine."""
    log_path = os.path.join(ds_dir, "refine.log")
    with LOCK:
        JOBS[job_id]["phase"] = "refine"
    cmd = ["python3", REFINE_SCRIPT, ds_dir]
    try:
        with open(log_path, "wb") as lf:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=lf,
                                    cwd=ds_dir, text=True)
        with LOCK:
            JOBS[job_id]["proc"] = proc
        out, _ = proc.communicate()
        rc = proc.returncode
    except Exception:  # noqa: BLE001
        rc, out = 1, ""
    with LOCK:
        JOBS[job_id]["phase"] = None
    if rc != 0:
        return None
    for line in reversed(out.splitlines()):
        if line.startswith("REFINED "):
            return line[len("REFINED "):].strip()
    return None


def _run_job(job_id: str, ds_dir: str) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    dataset_dir = Path(ds_dir).resolve()
    error = None

    try:
        output_dir = Path(tempfile.mkdtemp(prefix="colmap-", dir=dataset_dir))
        mesh_path = output_dir / "mesh.ply"
        log_path = output_dir / "colmap.log"
        command = [
            sys.executable, "-u", "-m", "something.main",
            str(dataset_dir), str(output_dir),
        ]

        with log_path.open("wb") as log_file:
            with LOCK:
                job = JOBS[job_id]
                if job["state"] in ("cancelling", "cancelled"):
                    job["state"] = "cancelled"
                    return

                process = subprocess.Popen(
                    command,
                    cwd=project_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                job.update(proc=process, phase="reconstruct")

            exit_code = process.wait()

        if exit_code != 0:
            error = f"COLMAP exited with code {exit_code}: {_log_tail(log_path)}"
        elif not mesh_path.is_file() or mesh_path.stat().st_size == 0:
            error = "COLMAP finished without producing a nonempty mesh.ply"
    except OSError as exc:
        error = f"COLMAP job failed: {exc}"

    with LOCK:
        job = JOBS[job_id]
        job.pop("proc", None)
        job.pop("phase", None)

        if job["state"] in ("cancelling", "cancelled"):
            job["state"] = "cancelled"
        elif error is not None:
            job.update(state="error", error=error)
        else:
            job.update(state="done", ply=str(mesh_path))



def _log_tail(log_path: str, n: int = 400) -> str:
    try:
        with open(log_path, "rb") as f:
            f.seek(max(0, os.path.getsize(log_path) - n))
            return f.read().decode("utf-8", "replace").replace("\n", " ")[-n:]
    except OSError:
        return ""


def _find_dataset_root(base: str):
    """Locate the folder containing transforms.json (root or one subfolder)."""
    for root, _dirs, files in os.walk(base):
        if "transforms.json" in files and "images" in _dirs:
            return root
    return None

  
def _safe_extract(zf, dest, max_bytes, max_entries):
    """Extract a zip member-by-member into `dest`, bounding the total uncompressed
    bytes (defeats zip bombs even with lying size headers, since we count bytes we
    actually write) and rejecting entries that would escape `dest` (zip-slip)."""
    infos = zf.infolist()
    if len(infos) > max_entries:
        raise ValueError(f"too many entries ({len(infos)} > {max_entries})")
    dest_real = os.path.realpath(dest)
    total = 0
    for info in infos:
        target = os.path.realpath(os.path.join(dest, info.filename))
        if target != dest_real and not target.startswith(dest_real + os.sep):
            raise ValueError(f"unsafe path in zip: {info.filename}")
        if info.is_dir():
            os.makedirs(target, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(
                        f"uncompressed size exceeds {max_bytes} bytes")
                dst.write(chunk)


class Handler(BaseHTTPRequestHandler):
    # Abort a stalled socket read (slowloris / dribbled body) instead of pinning a
    # thread forever. Applies per socket operation, so a steady large upload is fine.
    timeout = 60

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self):
        """True if the request may proceed. Open only when no auth is configured
        at all; if credentials are configured but currently unreadable/malformed,
        fail CLOSED rather than silently reopening. Constant-time token compare.
        /health bypasses this (checked by its callers)."""
        tokens = _active_tokens()
        hdr = self.headers.get("Authorization", "")
        supplied = hdr[len("Bearer "):] if hdr.startswith("Bearer ") else None
        if tokens:
            return supplied is not None and any(
                hmac.compare_digest(supplied, t) for t in tokens)
        # No active tokens: open unless the creds file is present-but-broken.
        return not _CREDS_CACHE.get("broken", False)

    def do_POST(self):
        if not self.path.startswith("/jobs"):
            return self._json(404, {"error": "not found"})
        if not self._authed():
            return self._json(401, {"error": "unauthorized"})
        q = parse_qs(urlparse(self.path).query)

        # TODO: upravit parametre aby sedeli s config.py
        try:
            iters = int(q.get("iters", ["2000"])[0])
            max_res = int(q.get("max_res", ["1024"])[0])
            # hold out every Nth view for PSNR
            eval_split = int(q.get("eval_split", ["0"])[0])
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"error": "bad numeric parameter"})

        # clamp so one request can't
        iters = max(1, min(iters, ITERS_MAX))
        # monopolise / OOM the GPU
        max_res = max(64, min(max_res, RES_MAX))
        trainer = q.get("trainer", ["brush"])[0]          # brush | splatfacto
        refine = q.get("refine", [""])[0]
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0:
            return self._json(400, {"error": "empty body (expected dataset.zip)"})
        if n > MAX_UPLOAD_BYTES:
            return self._json(413, {"error": f"upload too large (> {MAX_UPLOAD_BYTES} bytes)"})
        job_id = uuid.uuid4().hex[:12]
        ds_dir = os.path.join(JOBS_DIR, job_id)
        os.makedirs(ds_dir, exist_ok=True)
        tmp_name = None
        try:
            # Stream the body to a temp file in bounded chunks — never hold the whole
            # upload in RAM — reading at most the declared Content-Length.
            with tempfile.NamedTemporaryFile(dir=JOBS_DIR, delete=False) as tmp:
                tmp_name = tmp.name
                remaining = n
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    tmp.write(chunk)
                    remaining -= len(chunk)
            with zipfile.ZipFile(tmp_name) as zf:
                _safe_extract(zf, ds_dir, MAX_UNCOMPRESSED, MAX_ENTRIES)
        except Exception as e:  # noqa: BLE001
            # don't leave a partial dir
            shutil.rmtree(ds_dir, ignore_errors=True)
            return self._json(400, {"error": f"bad upload: {e}"})
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
        root = _find_dataset_root(ds_dir)
        if not root:
            shutil.rmtree(ds_dir, ignore_errors=True)
            return self._json(400, {"error": "no transforms.json in upload"})
        with LOCK:
            JOBS[job_id] = {"state": "queued", "queued_at": time.time(),
                            "started": None, "iter": 0, "total": iters, "trainer": trainer}
        WORK_Q.put((job_id, root, iters, max_res, eval_split, trainer, refine))
        self._json(200, {"job_id": job_id})

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"ok": True, "brush": os.path.exists(BRUSH)})
        if not self._authed():
            return self._json(401, {"error": "unauthorized"})
        parts = self.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "jobs":
            job_id = parts[1]
            with LOCK:
                j = JOBS.get(job_id)
            if not j:
                return self._json(404, {"error": "no such job"})
            if len(parts) >= 3 and parts[2] == "result":
                if j.get("state") != "done":
                    return self._json(409, {"error": f"state={j.get('state')}"})
                ply = j["ply"]
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(os.path.getsize(ply)))
                self.end_headers()
                with open(ply, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
                return
            started = j.get("started")
            out = {"state": j["state"],
                   "elapsed": round(time.time() - started) if started else 0,
                   "iter": j.get("iter", 0), "total": j.get("total", 0)}
            if j["state"] == "queued":
                out["queue_pos"] = _queue_position(job_id)
            if j.get("phase"):
                out["phase"] = j["phase"]   # e.g. "refine" (COLMAP running)
            if j.get("psnr") is not None:
                out["psnr"] = round(j["psnr"], 2)
                out["ssim"] = round(j.get("ssim", 0.0), 3)
                if j.get("lpips") is not None:
                    out["lpips"] = round(j["lpips"], 3)
                if j.get("verdict"):
                    # good | fair | poor | review
                    out["verdict"] = j["verdict"]
                if j.get("sfm"):
                    out["sfm"] = j["sfm"]            # {points, reproj, track}
                # Colour-corrected reference (bilateral-grid runs only) — a
                # train-only diagnostic the app shows alongside the raw score.
                if j.get("cc_psnr") is not None:
                    out["cc_psnr"] = round(j["cc_psnr"], 2)
            if j.get("error"):
                out["error"] = j["error"]
            return self._json(200, out)
        self._json(404, {"error": "not found"})

    def do_DELETE(self):
        if not self._authed():
            return self._json(401, {"error": "unauthorized"})

        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) != 2 or parts[0] != "jobs":
            return self._json(404, {"error": "not found"})

        with LOCK:
            job = JOBS.get(parts[1])

            if job is None:
                response = (404, {"error": "no such job"})
            elif job["state"] in ("done", "error"):
                response = (409, {"error": "job already finished"})
            else:
                try:
                    if job["state"] == "queued":
                        job["state"] = "cancelled"
                    elif job["state"] == "running":
                        process = job.get("proc")
                        if process is not None:
                            try:
                                os.killpg(process.pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass  # The process group already exited.
                        job["state"] = "cancelling"
                except OSError:
                    response = (
                        500, {"error": "could not signal the COLMAP process group"})
                else:
                    state = job["state"]
                    response = (
                        202 if state == "cancelling" else 200,
                        {"ok": True, "state": state},
                    )

        return self._json(*response)

    def log_message(self, *args):  # quiet
        pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    # When stdout is a pipe (systemd journal, Docker), Python block-buffers it, so
    # startup/reaper logs can sit unflushed for a long time. Force line buffering
    # so `journalctl`/`docker logs` show them promptly.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001
        pass
    _ntok = len(_active_tokens())
    print(f"rumahku backend on {BIND}:{PORT}  (brush={BRUSH}, jobs={JOBS_DIR}, "
          f"workers={MAX_CONCURRENT}, auth={'on (%d token%s)' % (_ntok, '' if _ntok == 1 else 's') if _ntok else 'off'}, "
          f"retain={JOB_RETAIN_MAX}/{JOB_RETAIN_HOURS:g}h)", flush=True)
    # BIND may name an interface (e.g. the NetBird VPN) that isn't up yet at boot,
    # so retry for a minute before giving up rather than crash-looping.
    srv = None
    for attempt in range(60):
        try:
            srv = Server((BIND, PORT), Handler)
            break
        except OSError as e:
            print(
                f"bind {BIND}:{PORT} failed ({e}); retry {attempt + 1}/60", flush=True)
            time.sleep(1)
    if srv is None:
        raise SystemExit(f"could not bind {BIND}:{PORT} after 60s")
    for _ in range(MAX_CONCURRENT):
        threading.Thread(target=_worker, daemon=True).start()
    threading.Thread(target=_reaper_loop, daemon=True).start()
    srv.serve_forever()
