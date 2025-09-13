# tools/install_torch.py
import os, platform, shutil, subprocess, sys, glob

CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
CPU_INDEX  = "https://download.pytorch.org/whl/cpu"
VER_TORCH, VER_VISION, VER_AUDIO = "2.5.1", "0.20.1", "2.5.1"

def run(cmd):
    print(">", " ".join(cmd)); sys.stdout.flush()
    return subprocess.call(cmd)

def check_torch_in_subproc():
    code = (
        "import torch; "
        "print(getattr(torch,'__version__','?'), 'cuda', torch.cuda.is_available())"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    return out

def has_nvidia():
    return shutil.which("nvidia-smi") is not None

def site_packages_dir():
    import site
    for d in site.getsitepackages():
        if d.endswith("site-packages") or d.endswith("dist-packages"):
            return d
    # fallback for virtualenv
    return os.path.join(os.path.dirname(site.__file__), "site-packages")

def clean_stuck_torch_dirs():
    sp = site_packages_dir()
    for pat in ["torch*", "torchvision*", "torchaudio*"]:
        for path in glob.glob(os.path.join(sp, pat)):
            if os.path.basename(path).startswith("~"):  # e.g. ~orch temp folder
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

def main():
    print(f"[install_torch] platform={platform.system().lower()} want_cuda={has_nvidia()}")
    # Read current state in a fresh subprocess (avoids stale import)
    try:
        print("[install_torch] before:", check_torch_in_subproc())
    except subprocess.CalledProcessError:
        print("[install_torch] before: torch not installed")

    want_cuda = has_nvidia() and platform.system().lower() != "darwin"
    # If torch exists but CPU-only, switch; if missing, install preferred
    # Try CPU->CUDA switch first if we want CUDA
    if want_cuda:
        uninstall_torch()
        rc = install_cuda()
        if rc != 0:
            print("[install_torch] CUDA install failed; falling back to CPU wheels.")
            uninstall_torch()
            rc = install_cpu()
            if rc != 0:
                print("[install_torch] CPU install also failed.")
                sys.exit(1)
    else:
        # No NVIDIA GPU detected (or macOS) -> CPU wheels
        # If already installed, do nothing
        # But if missing, install CPU wheels
        try:
            _ = check_torch_in_subproc()  # raises if not installed
            print("[install_torch] torch already present; skipping install.")
        except subprocess.CalledProcessError:
            rc = install_cpu()
            if rc != 0:
                print("[install_torch] CPU install failed.")
                sys.exit(1)

    # Final verification in a new interpreter
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
