# tools/install_torch.py
import os, platform, shutil, subprocess, sys, glob, json

CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
CPU_INDEX  = "https://download.pytorch.org/whl/cpu"
# Pin versions here
VER_TORCH, VER_VISION, VER_AUDIO = "2.5.1", "0.20.1", "2.5.1"

def run(cmd):
    print(">", " ".join(cmd)); sys.stdout.flush()
    return subprocess.call(cmd)

def check_torch_in_subproc():
    """
    Returns a short status line from a fresh interpreter, e.g.:
    '2.5.1+cu121 cuda True' or '2.5.1 cuda False'
    Raises CalledProcessError if import fails.
    """
    code = (
        "import torch, json;"
        "print(getattr(torch,'__version__','?'), 'cuda', bool(getattr(torch,'cuda',None) and torch.cuda.is_available()))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    return out

def read_cuda_tag_in_subproc():
    """
    Returns something like '12.1' if torch.version.cuda is present, else None.
    """
    code = (
        "import torch, json;"
        "print(json.dumps({'cuda': getattr(getattr(torch,'version',None),'cuda',None)}))"
    )
    try:
        j = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        return (json.loads(j) or {}).get("cuda")
    except Exception:
        return None

def has_nvidia():
    # nvidia-smi presence is a reliable signal on Windows/Linux
    return shutil.which("nvidia-smi") is not None

def site_packages_dir():
    import site
    # Prefer site.getsitepackages() when available (venv)
    sps = []
    try:
        sps = site.getsitepackages()
    except Exception:
        pass
    for d in sps or []:
        if d.endswith(("site-packages", "dist-packages")):
            return d
    # Fallback: derive from site module location
    return os.path.join(os.path.dirname(site.__file__), "site-packages")

def clean_stuck_torch_dirs():
    sp = site_packages_dir()
    for pat in ["torch*", "torchvision*", "torchaudio*"]:
        for path in glob.glob(os.path.join(sp, pat)):
            # Sometimes a temp folder like '~orch' lingers
            base = os.path.basename(path)
            if base.startswith("~"):
                try:
                    shutil.rmtree(path, ignore_errors=False)
                    print(f"[install_torch] removed leftover {path}")
                except Exception as e:
                    print(f"[install_torch] could not remove {path}: {e}")

def uninstall_torch():
    run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
    clean_stuck_torch_dirs()

def install_cuda():
    return run([sys.executable, "-m", "pip", "install",
                f"torch=={VER_TORCH}+cu121",
                f"torchvision=={VER_VISION}+cu121",
                f"torchaudio=={VER_AUDIO}+cu121",
                "--index-url", CUDA_INDEX])

def install_cpu():
    return run([sys.executable, "-m", "pip", "install",
                f"torch=={VER_TORCH}",
                f"torchvision=={VER_VISION}",
                f"torchaudio=={VER_AUDIO}",
                "--index-url", CPU_INDEX])

def detect_build_from_version_str(ver_str: str) -> str | None:
    """
    Inspect torch __version__ for a wheel tag like '+cu121'.
    Returns 'cu121', 'cpu', or None if unknown.
    """
    if not ver_str:
        return None
    if "+cu121" in ver_str:
        return "cu121"
    if "+" in ver_str:
        # Some other local tag; treat as CPU unless cuda is reported
        return "cpu"
    # Vanilla version like '2.5.1' -> likely CPU
    return "cpu"

def parse_installed_state():
    """
    Returns (present: bool, version: str|None, build: 'cpu'|'cu121'|None)
    """
    try:
        out = check_torch_in_subproc()  # e.g. "2.5.1+cu121 cuda True"
    except subprocess.CalledProcessError:
        return False, None, None

    tokens = out.split()
    ver = tokens[0] if tokens else None
    build = detect_build_from_version_str(ver)

    # Fallback confirmation: ask torch.version.cuda
    if build != "cu121":
        cuda = read_cuda_tag_in_subproc()
        if cuda and str(cuda).startswith("12.1"):
            build = "cu121"

    return True, ver, build

def main():
    plat = platform.system().lower()
    nvidia = has_nvidia()
    want_cuda = nvidia and plat != "darwin"
    desired_build = "cu121" if want_cuda else "cpu"

    print(f"[install_torch] platform={plat} nvidia={nvidia} desired_build={desired_build}")

    # Current state (fresh interpreter)
    try:
        before = check_torch_in_subproc()
        print("[install_torch] before:", before)
    except subprocess.CalledProcessError:
        print("[install_torch] before: torch not installed")

    present, ver, build = parse_installed_state()
    print(f"[install_torch] detected: present={present} version={ver} build={build}  target={VER_TORCH}+{desired_build}")

    # --- Idempotency guard: skip if already correct ---
    if present and ver and build:
        same_ver = (ver.split("+")[0] == VER_TORCH)  # ignore local tag
        if same_ver and build == desired_build:
            print("[install_torch] correct Torch already installed; skipping.")
            return

    # --- Install/Swap to desired build ---
    uninstall_torch()
    rc = install_cuda() if desired_build == "cu121" else install_cpu()

    # If CUDA failed, fall back to CPU
    if rc != 0 and desired_build == "cu121":
        print("[install_torch] CUDA install failed; trying CPU wheels.")
        uninstall_torch()
        rc = install_cpu()

    if rc != 0:
        print("[install_torch] installation failed.")
        sys.exit(1)

    # Final verification
    try:
        after = check_torch_in_subproc()
        print("[install_torch] after:", after)
        if want_cuda and "cuda False" in after:
            print("[install_torch] WARNING: NVIDIA GPU present but CUDA build not active.")
            sys.exit(1)
    except subprocess.CalledProcessError:
        print("[install_torch] torch import failed after install.")
        sys.exit(1)

if __name__ == "__main__":
    main()
