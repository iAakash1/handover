"""Create the Python environments the projects need, inside .venvs/ at the repository root.

I froze these requirement files from the exact environments that trained and evaluated the models, so a
fresh setup behaves the same way. Needs Python 3.10 (64-bit) and an NVIDIA GPU with a recent driver for the
GPU environments. Each environment is skipped if it already exists.

Usage (from the repository root):
    python setup/setup_envs.py            # all environments
    python setup/setup_envs.py vlm keras  # only the ones you name

Environments and the projects that use them:
    keras     01_cnn_keras_tomato                          (TensorFlow 2.20 / Keras 3, CPU)
    soynet    03_soybean_soynet                            (TensorFlow 2.10 + PyTorch, CUDA 12.1)
    vlm       02, 04, 06, 07, 08, evaluation              (PyTorch 2.5.1 CUDA 12.1, transformers 5.3)
    florence  failed_experiments/01_florence2_finetuning                              (same as vlm but transformers 4.46.3)
    tih       10_tih_tomato_qwen_qlora                     (PyTorch 2.7.1 CUDA 11.8, plantdx package)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVS = ["keras", "soynet", "vlm", "florence", "tih"]


def run(cmd):
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    if sys.version_info[:2] != (3, 10):
        print(f"warning: these environments were built with Python 3.10, you are running {sys.version.split()[0]}")
    wanted = sys.argv[1:] or ENVS
    for name in wanted:
        env = os.path.join(ROOT, ".venvs", name)
        py = os.path.join(env, "Scripts" if os.name == "nt" else "bin", "python")
        marker = os.path.join(env, ".ready")
        if os.path.exists(marker):
            print(f"[{name}] already set up, skipping")
            continue
        if not os.path.exists(py):  # a half-finished env is reused and completed
            run([sys.executable, "-m", "venv", env])
        run([py, "-m", "pip", "install", "--upgrade", "pip"])
        run([py, "-m", "pip", "install", "-r", os.path.join(ROOT, "setup", "requirements", f"{name}.txt")])
        if name == "tih":  # the TIH project is a proper package; install it in editable mode
            run([py, "-m", "pip", "install", "--no-deps", "-e",
                 os.path.join(ROOT, "projects", "10_tih_tomato_qwen_qlora")])
        with open(marker, "w") as fh:
            fh.write("installed by setup/setup_envs.py\n")
        print(f"[{name}] ready")


if __name__ == "__main__":
    main()
