"""Đóng gói src/ thành một kernel Kaggle (loại script) để train trên GPU T4.

src/ là nguồn code duy nhất: script này nhúng nguyên văn mọi file src/*.py vào kernel. Trên Kaggle,
kernel ghi lại chúng thành thư mục src/ rồi gọi `python -m src.train`, nên không có code nào bị chép tay.
Kết quả nằm trong kaggle/build/: run_phobert.py và kernel-metadata.json (đã điền tên tài khoản Kaggle).

Chạy từ thư mục project. Mỗi --run là một lần train, các run chạy lần lượt trong cùng một phiên GPU:
python -X utf8 kaggle/build_kernel.py --run "--max_len 128 --class_weights none --seed 42"
"""

import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # để import được src.train khi chạy file này trực tiếp
from src.train import make_run_name, parse_args as parse_train_args  # noqa: E402

TEMPLATE = ROOT / "kaggle" / "kernel-metadata.json"
BUILD_DIR = ROOT / "kaggle" / "build"
PINNED = ("underthesea", "transformers")  # trên Kaggle cài đúng bản trong requirements.txt nếu bản có sẵn khác

# Phần chạy trên Kaggle. __SOURCES__, __PINS__, __RUNS__ được thay bằng dữ liệu thật lúc build
RUNTIME = r'''
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

SOURCES = __SOURCES__
PINS = __PINS__
RUNS = __RUNS__

# /kaggle/working là thư mục output của kernel: file ghi ở đây tải về được sau khi chạy xong
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
for rel_path, text in SOURCES.items():
    path = WORK / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def installed_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


# Tách từ và tokenizer phải giống hệt lúc chạy thử trên máy. torch giữ bản GPU có sẵn của Kaggle
to_install = [f"{pkg}=={ver}" for pkg, ver in PINS.items() if installed_version(pkg) != ver]
if to_install:
    print("pip install", *to_install, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *to_install], check=True)
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"])  # ghi tên GPU và VRAM vào log

failed = []
for args in RUNS:
    print("\n=== python -m src.train " + " ".join(args), flush=True)
    # -u: in log ngay, để `kaggle kernels logs` thấy tiến độ trong lúc chạy
    if subprocess.run([sys.executable, "-X", "utf8", "-u", "-m", "src.train", *args], cwd=WORK).returncode != 0:
        failed.append(args)  # run lỗi không chặn các run sau
shutil.rmtree(WORK / "data", ignore_errors=True)  # không để bộ dữ liệu nằm trong output của kernel
if failed:
    sys.exit(f"Các run bị lỗi: {failed}")
'''


def read_pins():
    """Lấy version của các gói trong PINNED từ requirements.txt, để chỉ ghim version ở một nơi."""
    pins = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line and line.split("==")[0].strip() in PINNED:
            name, ver = line.split("==")
            pins[name.strip()] = ver.strip()
    return pins


def kaggle_username():
    """Lấy tên tài khoản qua chính thư viện kaggle (đọc cấu hình đăng nhập có sẵn), không in hay chép khoá.
    Template giữ chỗ KAGGLE_USERNAME để repo công khai không chứa tên tài khoản."""
    from kaggle.api.kaggle_api_extended import KaggleApi  # chỉ cài trong venv, không nằm trong requirements.txt
    api = KaggleApi()
    api.authenticate()
    return api.get_config_value("username")


def main():
    parser = argparse.ArgumentParser(description="Sinh kernel Kaggle từ src/")
    parser.add_argument("--run", action="append", required=True, help='tham số cho src.train, vd "--max_len 128 ..."')
    runs = [shlex.split(r) for r in parser.parse_args().run]

    # Kiểm tra tham số ngay trên máy, để lỗi gõ sai không tốn một phiên GPU trên Kaggle
    names = [make_run_name(parse_train_args(r)) for r in runs]
    if len(set(names)) != len(names):
        sys.exit(f"Hai run trùng tên sẽ ghi đè kết quả của nhau: {names}")

    sources = {f"src/{p.name}": p.read_text(encoding="utf-8") for p in sorted((ROOT / "src").glob("*.py"))}
    hashes = ", ".join(f"{k}={hashlib.sha256(v.encode()).hexdigest()[:12]}" for k, v in sources.items())
    # Thay __SOURCES__ sau cùng, để chữ giữ chỗ không bao giờ bị thay nhầm bên trong code được nhúng
    body = (RUNTIME.replace("__PINS__", repr(read_pins())).replace("__RUNS__", repr(runs))
            .replace("__SOURCES__", "{\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in sources.items()) + "}"))
    header = (f"# Sinh tự động bởi kaggle/build_kernel.py lúc {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.\n"
              f"# Không sửa file này: sửa src/ rồi build lại. sha256 các file nhúng: {hashes}\n")

    with open(TEMPLATE, encoding="utf-8") as f:
        meta = json.load(f)
    meta["id"] = meta["id"].replace("KAGGLE_USERNAME", kaggle_username())
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    (BUILD_DIR / meta["code_file"]).write_text(header + body, encoding="utf-8")
    with open(BUILD_DIR / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Đã tạo {BUILD_DIR / meta['code_file']} ({len(sources)} file src, {len(runs)} run).")
    for name in names:
        print(f"  {name}: results/phobert_{name}.json, results/figures/{{cm,curves}}_phobert_{name}.png, "
              f"results/errors_phobert_{name}.csv, models/phobert_{name}/")
    kernel = meta["id"]
    print("\nCác bước tiếp theo:\n"
          f"  kaggle kernels push -p kaggle/build\n"
          f"  kaggle kernels status {kernel}\n"
          f"  kaggle kernels logs {kernel} -f\n"
          f'  kaggle kernels output {kernel} -p kaggle/output --file-pattern "^results/"')


if __name__ == "__main__":
    main()
