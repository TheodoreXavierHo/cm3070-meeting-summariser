# tools/install_torch.py
import os, platform, shutil, subprocess, sys, glob, json

# ---- PyTorch wheels repos ----
CUDA_INDEXES = {
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu118": "https://download.pytorch.org/whl/cu118",
}
CPU_INDEX  = "https://download.pytorch.org/whl/cpu"

# Order matters: try newer CUDA first, then older
CANDIDATE_CUDA_TAGS = ["cu121", "cu118"]

def run(cmd):
    print(">", " ".join(cmd)); sys.stdout.flush()
    return subprocess.call(cmd)

def check_torch_in_subproc():
    """
    Returns a short status line from a fresh interpreter, e.g.:
    '2.7.1+cu118 cuda True' or '2.7.1+cpu cuda False'
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
    sps = []
    try:
        sps = site.getsitepackages()
    except Exception:
        pass
    for d in sps or []:
        if d.endswith(("site-packages", "dist-packages")):
            return d
    return os.path.join(os.path.dirname(site.__file__), "site-packages")

def clean_stuck_torch_dirs():
    sp = site_packages_dir()
    for pat in ["torch*", "torchvision*", "torchaudio*"]:
        for path in glob.glob(os.path.join(sp, pat)):
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

def detect_build_from_version_str(ver_str: str) -> str | None:
    """
    Inspect torch __version__ for a wheel tag like '+cu118' / '+cpu'.
    Returns 'cu121' / 'cu118' / 'cpu' (best-effort).
    """
    if not ver_str:
        return None
    for tag in CANDIDATE_CUDA_TAGS:
        if f"+{tag}" in ver_str:
            return tag
    if "+cpu" in ver_str or "+" not in ver_str:
        return "cpu"
    return "cpu"

def parse_installed_state():
    """
    Returns (present: bool, version: str|None, build: 'cpu'|CUDA_TAG|None)
    """
    try:
        out = check_torch_in_subproc()  # e.g. "2.7.1+cu118 cuda True"
    except subprocess.CalledProcessError:
        return False, None, None

    tokens = out.split()
    ver = tokens[0] if tokens else None
    build = detect_build_from_version_str(ver)

    # Fallback confirmation: ask torch.version.cuda
    if build == "cpu":
        cuda = read_cuda_tag_in_subproc()
        if cuda:
            # We don't map minors precisely; presence implies a CUDA build
            # but we'll keep 'cpu' unless it's one of our known tags.
            for tag in CANDIDATE_CUDA_TAGS:
                if tag.endswith("121") and str(cuda).startswith("12.1"):
                    build = "cu121"
                if tag.endswith("118") and str(cuda).startswith("11.8"):
                    build = "cu118"
    return True, ver, build

def pip_install_torch(index_url: str, want_cuda: bool) -> int:
    """
    Install torch first (no version pins), verify import, then best-effort install
    torchvision+torchaudio from the same index. Return 0 on success, non-zero otherwise.
    """
    # Always upgrade pip/wheel for wheel resolution reliability
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])

    # 1) Install torch (latest on that channel)
    rc = run([sys.executable, "-m", "pip", "install", "torch", "--index-url", index_url])
    if rc != 0:
        return rc

    # Verify torch import + (optional) CUDA activation
    try:
        after = check_torch_in_subproc()
        print("[install_torch] after torch:", after)
        if want_cuda and "cuda True" not in after:
            # CUDA wheel didn’t activate; treat as failure so caller can try next tag
            return 2
    except subprocess.CalledProcessError:
        return 3

    # 2) Best-effort install of vision/audio (unversioned) from same channel
    # If not available for this tag, we won’t fail the whole install.
    run([sys.executable, "-m", "pip", "install", "torchvision", "--index-url", index_url])
    run([sys.executable, "-m", "pip", "install", "torchaudio", "--index-url", index_url])

    # Final sanity re-check
    try:
        after = check_torch_in_subproc()
        print("[install_torch] final:", after)
    except subprocess.CalledProcessError:
        return 4
    return 0

def main():
    plat = platform.system().lower()        # 'windows', 'linux', 'darwin'
    mach = platform.machine().lower()       # 'amd64'/'x86_64', 'arm64', ...
    nvidia = has_nvidia()
    force_cpu = os.getenv("FYP_TORCH_CPU") == "1"

    # Only attempt CUDA on Windows/Linux x86_64 with NVIDIA present, unless overridden.
    cuda_capable_os = plat in ("windows", "linux")
    cuda_capable_arch = mach in ("x86_64", "amd64")
    allow_cuda = (not force_cpu) and nvidia and cuda_capable_os and cuda_capable_arch

    print(f"[install_torch] platform={plat} arch={mach} nvidia={nvidia} force_cpu={force_cpu} allow_cuda={allow_cuda}")

    # Current state (fresh interpreter)
    try:
        before = check_torch_in_subproc()
        print("[install_torch] before:", before)
    except subprocess.CalledProcessError:
        print("[install_torch] before: torch not installed")

    present, ver, build = parse_installed_state()
    print(f"[install_torch] detected: present={present} version={ver} build={build}")

    # Idempotency: if Torch is already installed and matches mode, skip
    if present and ver:
        if allow_cuda and build in CANDIDATE_CUDA_TAGS:
            print("[install_torch] correct CUDA Torch already installed; skipping.")
            return
        if (not allow_cuda) and build == "cpu":
            print("[install_torch] correct CPU Torch already installed; skipping.")
            return

    # (Re)install
    uninstall_torch()

    if allow_cuda:
        for tag in CANDIDATE_CUDA_TAGS:
            print(f"[install_torch] trying CUDA build: {tag}")
            idx = CUDA_INDEXES[tag]
            rc = pip_install_torch(idx, want_cuda=True)
            if rc == 0:
                print(f"[install_torch] SUCCESS: CUDA is active with {tag}")
                return
            print(f"[install_torch] CUDA install did not activate for {tag} (rc={rc}); trying next…")
            uninstall_torch()

    # CPU fallback
    print("[install_torch] installing CPU wheels…")
    rc = pip_install_torch(CPU_INDEX, want_cuda=False)
    if rc != 0:
        print("[install_torch] CPU installation failed (rc=%s)." % rc)
        sys.exit(1)

if __name__ == "__main__":
    main()
