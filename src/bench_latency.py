"""Đo latency suy luận trên CPU, ghi vào results/latency.json (mỗi target đo xong là lưu ngay).

Ba target, cùng một tập request:
  inprocess : gọi thẳng SentimentClassifier (tách từ + tokenize + model), tách riêng thời gian từng chặng
  uvicorn   : script tự bật server local, đo qua HTTP rồi tắt
  docker    : script tự chạy container từ image, đo qua HTTP rồi dừng (cần build image trước)

python -X utf8 -m src.bench_latency                                   # fp32, cả 3 target
python -X utf8 -m src.bench_latency --quantize int8 --targets inprocess uvicorn
Số đo phụ thuộc máy (CPU, cắm sạc hay chạy pin, tải nền): chỉ so sánh các số đo trên cùng một máy.
"""

import argparse
import contextlib
import ctypes
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import httpx
import numpy as np
import torch

from app.main import MODEL_DIR, SentimentClassifier
from src.data import load_split
from src.dataset import segment

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "latency.json"
DOCKER_WINDOWS = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
PACKAGES = ("torch", "transformers", "tokenizers", "underthesea", "fastapi", "uvicorn", "pydantic")
SEED = 42
# p95 của 50 request chỉ dựa vào 2–3 request chậm nhất nên dao động mạnh giữa các lần đo; lấy nhiều hơn cho ổn định
N1, N16, N_WARMUP = 300, 80, 10


def cpu_name():
    if sys.platform == "win32":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    with open("/proc/cpuinfo", encoding="utf-8") as f:
        return next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), platform.processor())


def on_ac_power():
    """Laptop chạy pin thường tự hạ xung CPU nên latency tăng. Ghi lại để biết số đo trong điều kiện nào."""
    if sys.platform != "win32":
        return None

    class PowerStatus(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_byte), ("flag", ctypes.c_byte), ("percent", ctypes.c_byte),
                    ("saver", ctypes.c_byte), ("life", ctypes.c_ulong), ("full_life", ctypes.c_ulong)]
    status = PowerStatus()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
    return {0: False, 1: True}.get(status.ac)


def power_plan():
    """Chế độ điện của Windows (vd "Silent" giới hạn công suất CPU) ảnh hưởng thẳng tới latency."""
    if sys.platform != "win32":
        return None
    out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True).stdout
    return out[out.find("(") + 1:out.rfind(")")] or None


def environment():
    return {"cpu": cpu_name(), "logical_cpus": os.cpu_count(), "os": platform.platform(), "power_plan": power_plan(),
            "torch_threads": torch.get_num_threads(), "quantized_engine": torch.backends.quantized.engine,
            "versions": {"python": platform.python_version(), **{p: version(p) for p in PACKAGES}}}


def make_requests(n1, n16, n_warmup):
    """Câu validation (chỉ lấy text), xáo cố định theo seed, nên mọi target nhận đúng cùng các request.
    Warm-up dùng câu khác với câu được đo."""
    texts = load_split("validation").text.sample(frac=1, random_state=SEED).tolist()
    rest = texts[16 * n16:]
    assert n1 <= 16 * n16 and 16 * n_warmup <= len(rest)
    measured = {1: [[t] for t in texts[:n1]], 16: [texts[i:i + 16] for i in range(0, 16 * n16, 16)]}
    warmup = {1: [[t] for t in rest[:n_warmup]], 16: [rest[i:i + 16] for i in range(0, 16 * n_warmup, 16)]}
    return measured, warmup


def summarize(total_ms, batch_size):
    """p50 = trung vị; p95 = 95% request nhanh hơn mức này (đuôi chậm, thứ người dùng cảm thấy).
    Mỗi câu = mỗi request chia batch_size: chi phí chia đều, không phải thời gian chờ của một câu."""
    ms = np.asarray(total_ms)

    def stats(a):
        return {"p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
                "mean": round(float(a.mean()), 2)}
    return {"n_requests": len(ms), "batch_size": batch_size,
            "per_request_ms": stats(ms), "per_sentence_ms": stats(ms / batch_size)}


def time_request(clf, texts):
    """Một request trong tiến trình, đo từng chặng (ms), đúng thứ tự như SentimentClassifier.predict_proba."""
    t0 = time.perf_counter()
    segmented = segment(texts)
    t1 = time.perf_counter()
    batch = clf.tokenize(segmented)
    t2 = time.perf_counter()
    clf.forward(batch)
    t3 = time.perf_counter()
    return {"segment": (t1 - t0) * 1e3, "tokenize": (t2 - t1) * 1e3, "model": (t3 - t2) * 1e3, "total": (t3 - t0) * 1e3}


def bench_inprocess(clf, requests, warmup):
    out = {}
    for bs, reqs in requests.items():
        for texts in warmup[bs]:
            time_request(clf, texts)
        rows = [time_request(clf, texts) for texts in reqs]
        res = summarize([r["total"] for r in rows], bs)
        res["stage_mean_ms"] = {k: round(float(np.mean([r[k] for r in rows])), 2) for k in ("segment", "tokenize", "model")}
        res["segment_share"] = round(sum(r["segment"] for r in rows) / sum(r["total"] for r in rows), 3)
        out[f"batch_{bs}"] = res
    return out


def bench_http(base_url, requests, warmup):
    """Đo phía client: gồm cả validate, JSON, mạng loopback và hàng đợi của server."""
    out = {}
    with httpx.Client(base_url=base_url, timeout=60) as client:  # giữ kết nối (keep-alive) như một client thật
        for bs, reqs in requests.items():
            for texts in warmup[bs]:
                client.post("/predict", json={"texts": texts}).raise_for_status()
            ms = []
            for texts in reqs:
                t0 = time.perf_counter()
                r = client.post("/predict", json={"texts": texts})
                ms.append((time.perf_counter() - t0) * 1e3)
                r.raise_for_status()
            out[f"batch_{bs}"] = summarize(ms, bs)
    return out


def wait_ready(base_url, t0, proc=None, timeout=300):
    """Cold start = từ lúc ra lệnh khởi động tới khi /health trả 200 (gồm khởi động Python, import, nạp model)."""
    while time.perf_counter() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError("server đã dừng trong lúc khởi động")
        with contextlib.suppress(httpx.TransportError):
            r = httpx.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return round(time.perf_counter() - t0, 2), r.json()
        time.sleep(0.1)
    raise TimeoutError(f"{base_url} chưa sẵn sàng sau {timeout} s")


@contextlib.contextmanager
def local_uvicorn(quantize, port):
    base_url = f"http://127.0.0.1:{port}"
    log_path = Path(tempfile.gettempdir()) / f"bench_uvicorn_{port}.log"  # không dùng PIPE: pipe đầy sẽ treo server
    with open(log_path, "w", encoding="utf-8") as log:
        t0 = time.perf_counter()
        proc = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                                 "--port", str(port)], cwd=ROOT, env={**os.environ, "QUANTIZE": quantize},
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            yield (base_url, *wait_ready(base_url, t0, proc))
        finally:
            proc.terminate()
            proc.wait(timeout=30)


def docker_bin():
    return shutil.which("docker") or (DOCKER_WINDOWS if Path(DOCKER_WINDOWS).exists() else None)


@contextlib.contextmanager
def docker_container(docker, image, quantize, port):
    name, base_url = "phobert-bench", f"http://127.0.0.1:{port}"
    subprocess.run([docker, "rm", "-f", name], capture_output=True)  # dọn container sót lại nếu lần trước bị ngắt
    t0 = time.perf_counter()
    subprocess.run([docker, "run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:8000",
                    "-v", f"{ROOT / 'models'}:/models:ro", "-e", f"QUANTIZE={quantize}", image],
                   check=True, capture_output=True)
    try:
        ready_s, health = wait_ready(base_url, t0)
        probe = subprocess.run([docker, "exec", name, "python", "-c", "import os, torch; print(os.cpu_count(), "
                                "torch.get_num_threads(), torch.backends.quantized.engine)"],
                               capture_output=True, text=True, check=True).stdout.split()
        inside = {"logical_cpus": int(probe[0]), "torch_threads": int(probe[1]), "quantized_engine": probe[2]}
        yield base_url, ready_s, health, inside
    finally:
        subprocess.run([docker, "stop", name], capture_output=True)


def docker_problem(docker, image):
    """Trả về lý do không đo được Docker, hoặc None nếu đo được."""
    if docker is None:
        return "không tìm thấy lệnh docker"
    if subprocess.run([docker, "info"], capture_output=True).returncode != 0:
        return "Docker daemon chưa chạy"
    if subprocess.run([docker, "image", "inspect", image], capture_output=True).returncode != 0:
        return f"chưa có image {image} (docker build -t {image} .)"
    return None


def save(variant, target, entry, settings):
    """Gộp vào file cũ: đo lại một target không xoá số của target khác."""
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    data["note"] = ("Đo trên CPU laptop, phụ thuộc máy. per_sentence = per_request / batch_size. "
                    "HTTP đo phía client (gồm JSON, loopback); Docker đi qua port forwarding của Docker Desktop.")
    data["environment"], data["settings"] = environment(), settings
    data.setdefault("runs", {}).setdefault(variant, {})[target] = entry
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def report(variant, target, entry):
    for key in ("batch_1", "batch_16"):
        r = entry[key]
        print(f"{variant:>4} {target:<9} {key:<8}: p50 {r['per_request_ms']['p50']:7.1f} ms | "
              f"p95 {r['per_request_ms']['p95']:7.1f} ms | mỗi câu (p50) {r['per_sentence_ms']['p50']:6.1f} ms"
              + (f" | tách từ {r['segment_share']:.0%}" if "segment_share" in r else ""))


def main():
    p = argparse.ArgumentParser(description="Đo latency API PhoBERT trên CPU")
    p.add_argument("--quantize", choices=["none", "int8"], default="none")
    p.add_argument("--targets", nargs="+", choices=["inprocess", "uvicorn", "docker"],
                   default=["inprocess", "uvicorn", "docker"])
    p.add_argument("--n1", type=int, default=N1, help="số request batch 1")
    p.add_argument("--n16", type=int, default=N16, help="số request batch 16 (validation có 1.583 câu: tối đa 88)")
    p.add_argument("--warmup", type=int, default=N_WARMUP, help="số request chạy trước, không tính")
    p.add_argument("--image", default="phobert-sentiment-api:1.0")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    variant = "fp32" if args.quantize == "none" else "int8"
    requests, warmup = make_requests(args.n1, args.n16, args.warmup)
    settings = {"data": f"validation, xáo seed {SEED}, chỉ text", "n_requests": {"batch_1": args.n1, "batch_16": args.n16},
                "warmup_requests_per_batch_size": args.warmup, "sequential": "một request một lúc, không chạy song song"}
    for target in args.targets:
        entry = {"measured_at": datetime.now().isoformat(timespec="seconds"), "on_ac_power": on_ac_power()}
        if target == "inprocess":
            t0 = time.perf_counter()
            clf = SentimentClassifier(MODEL_DIR, args.quantize)
            clf.predict_proba(["khởi động"])  # giống lifespan của API
            entry["model_load_seconds"] = round(time.perf_counter() - t0, 2)
            entry.update(bench_inprocess(clf, requests, warmup))
            del clf  # trả RAM trước khi bật server ở target sau
            gc.collect()
        elif target == "uvicorn":
            with local_uvicorn(args.quantize, args.port) as (url, ready_s, health):
                assert health["quantization"] == args.quantize, health
                entry.update(cold_start_seconds=ready_s, model_load_seconds=health["load_seconds"])
                entry.update(bench_http(url, requests, warmup))
        else:
            docker = docker_bin()
            if problem := docker_problem(docker, args.image):
                print(f"Bỏ qua docker: {problem}")
                continue
            size = subprocess.run([docker, "image", "inspect", "-f", "{{.Size}}", args.image],
                                  capture_output=True, text=True, check=True).stdout.strip()
            with docker_container(docker, args.image, args.quantize, args.port) as (url, ready_s, health, inside):
                assert health["quantization"] == args.quantize, health
                entry.update(image=args.image, image_size_mb=round(int(size) / 2**20), container=inside,
                             cold_start_seconds=ready_s, model_load_seconds=health["load_seconds"])
                entry.update(bench_http(url, requests, warmup))
        save(variant, target, entry, settings)
        report(variant, target, entry)


if __name__ == "__main__":
    main()
