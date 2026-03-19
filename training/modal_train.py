"""MolSight fine-tuning on Modal GPUs.

Usage:
    modal run training/modal_train.py::upload
    modal run training/modal_train.py::train_sft
    modal run training/modal_train.py::train_grpo
    modal run training/modal_train.py::download
"""

import os
import subprocess

import modal

# ──────────────────────────────────────────────────────────────
# Modal resources
# ──────────────────────────────────────────────────────────────

app = modal.App("molsight-finetune")

# Volumes for persistent storage across runs
code_vol = modal.Volume.from_name("molsight-code", create_if_missing=True)
data_vol = modal.Volume.from_name("molsight-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("molsight-checkpoints", create_if_missing=True)

MOLSIGHT_DIR = "/molsight"
DATA_DIR = "/data"
CKPT_DIR = "/checkpoints"

# Container image with all dependencies
train_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy<2",
        "torch==2.4.1",
        "torchvision==0.19.1",
        "albumentations>=1.3.0,<2.0",
        "timm>=0.9.0",
        "transformers>=4.36.0,<5",
        "datasets>=2.14.0",
        "tensorboard",
        "pandas",
        "openpyxl",
        "opencv-python-headless",
        "Pillow",
        "rdkit-pypi",
        "epam.indigo",
        "SmilesPE",
        "safetensors",
        "scipy",
    )
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
)

# ──────────────────────────────────────────────────────────────
# Local paths
# ──────────────────────────────────────────────────────────────

LOCAL_MOLSIGHT = os.path.join(os.path.expanduser("~"), "Documents", "Projects", "MolSight")
LOCAL_TRAINING_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data"
)
LOCAL_CHECKPOINT = os.path.join(LOCAL_MOLSIGHT, "pubchem_uspto_smiles_edges_30.pth")


# ──────────────────────────────────────────────────────────────
# Upload: push code, data, and pretrained checkpoint to volumes
# ──────────────────────────────────────────────────────────────


@app.function(
    volumes={
        MOLSIGHT_DIR: code_vol,
        DATA_DIR: data_vol,
        CKPT_DIR: ckpt_vol,
    },
    image=train_image,
    timeout=600,
)
def upload():
    """Upload is called with `modal run` but actual upload happens via volume put_directory below."""
    print("Volumes mounted. Upload completed via local_entrypoint.")


@app.local_entrypoint()
def upload_entrypoint():
    """Push local files to Modal volumes."""
    import shutil
    import tempfile

    print("Uploading MolSight source code...")
    # Upload source code
    with code_vol.batch_upload(force=True) as batch:
        for root, dirs, files in os.walk(LOCAL_MOLSIGHT):
            # Skip venv, __pycache__, .git, runs
            dirs[:] = [
                d for d in dirs
                if d not in ("venv", "__pycache__", ".git", "runs", "data")
            ]
            for fname in files:
                if fname.endswith((".py", ".json", ".sh", ".safetensors", ".so")):
                    local_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(local_path, LOCAL_MOLSIGHT)
                    batch.put_file(local_path, rel_path)
    print("  Source code uploaded.")

    print("Uploading training data...")
    with data_vol.batch_upload(force=True) as batch:
        for root, dirs, files in os.walk(LOCAL_TRAINING_DATA):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fname in files:
                local_path = os.path.join(root, fname)
                rel_path = os.path.relpath(local_path, LOCAL_TRAINING_DATA)
                batch.put_file(local_path, rel_path)
    print("  Training data uploaded.")

    if os.path.exists(LOCAL_CHECKPOINT):
        print("Uploading pretrained checkpoint...")
        with ckpt_vol.batch_upload(force=True) as batch:
            batch.put_file(LOCAL_CHECKPOINT, "pubchem_uspto_smiles_edges_30.pth")
        print("  Checkpoint uploaded.")
    else:
        print(f"  WARNING: Pretrained checkpoint not found at {LOCAL_CHECKPOINT}")
        print("  The training script will download it automatically.")

    print("\nAll uploads complete!")


# ──────────────────────────────────────────────────────────────
# Stage 1: Supervised Fine-Tuning (SFT)
# ──────────────────────────────────────────────────────────────


@app.function(
    gpu="A10G",
    volumes={
        MOLSIGHT_DIR: code_vol,
        DATA_DIR: data_vol,
        CKPT_DIR: ckpt_vol,
    },
    image=train_image,
    timeout=14400,  # 4 hours
    memory=32768,
)
def train_sft():
    """Stage 1: Supervised fine-tuning on patent SMILES with Indigo augmentation."""
    os.chdir(MOLSIGHT_DIR)

    # Symlink data and checkpoint into expected locations
    os.makedirs("data", exist_ok=True)
    for item in ("pubchem", "real"):
        src = os.path.join(DATA_DIR, item)
        dst = os.path.join("data", item)
        if not os.path.exists(dst) and os.path.exists(src):
            os.symlink(src, dst)

    # Symlink pretrained checkpoint
    ckpt_src = os.path.join(CKPT_DIR, "pubchem_uspto_smiles_edges_30.pth")
    if not os.path.exists("pubchem_uspto_smiles_edges_30.pth"):
        if os.path.exists(ckpt_src):
            os.symlink(ckpt_src, "pubchem_uspto_smiles_edges_30.pth")
        else:
            # Download if not available
            print("Downloading pretrained checkpoint...")
            import urllib.request
            url = "https://huggingface.co/Robert-zwr/MolSight/resolve/main/pubchem_uspto_smiles_edges_30.pth?download=true"
            urllib.request.urlretrieve(url, "pubchem_uspto_smiles_edges_30.pth")

    cmd = [
        "torchrun", "--nproc_per_node=1",
        "train.py",
        "--data_path", "data",
        "--train_datasets", "pubchem,patent",
        "--valid_file", "real/patent_realimg_val.csv",
        "--vocab_file", "vocab/vocab_chars.json",
        "--formats", "char",
        "--dynamic_indigo",
        "--augment",
        "--mol_augment",
        "--include_condensed",
        "--encoder", "efficientvit",
        "--input_size", "512",
        "--use_qknorm",
        "--use_swiglu",
        "--use_rmsnorm",
        "--encoder_lr", "1e-5",
        "--decoder_lr", "1e-4",
        "--epochs", "5",
        "--batch_size", "8",
        "--accum_freq", "8",
        "--warmup_ratio", "0.05",
        "--scheduler", "cosine",
        "--load_path", "pubchem_uspto_smiles_edges_30.pth",
        "--load_model_only",
        "--smiles_only",
        "--amp",
        "--do_train",
        "--do_valid",
        "--save_mode", "all",
        "--print_freq", "50",
        "--num_workers", "4",
        "--backend", "gloo",
        "--exp_name", "patent_sft",
    ]

    print(f"Running SFT:\n  {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"Training exited with code {result.returncode}")

    # Copy best/final checkpoint to shared volume
    sft_runs = "runs/patent_sft/ckpt_model"
    if os.path.isdir(sft_runs):
        import shutil
        # Find the last epoch checkpoint
        ckpts = sorted(
            [f for f in os.listdir(sft_runs) if f.startswith("epoch_")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        if ckpts:
            final_ckpt = os.path.join(sft_runs, ckpts[-1])
            dst = os.path.join(CKPT_DIR, "patent_sft_final.pth")
            shutil.copy2(final_ckpt, dst)
            ckpt_vol.commit()
            print(f"Saved SFT checkpoint: {ckpts[-1]} -> {dst}")

    return result.returncode


# ──────────────────────────────────────────────────────────────
# Stage 2: GRPO with LoRA
# ──────────────────────────────────────────────────────────────


@app.function(
    gpu="A10G",
    volumes={
        MOLSIGHT_DIR: code_vol,
        DATA_DIR: data_vol,
        CKPT_DIR: ckpt_vol,
    },
    image=train_image,
    timeout=14400,  # 4 hours
    memory=32768,
)
def train_grpo():
    """Stage 2: GRPO/LoRA refinement on patent data."""
    os.chdir(MOLSIGHT_DIR)

    # Symlink data and checkpoint
    os.makedirs("data", exist_ok=True)
    for item in ("pubchem", "real"):
        src = os.path.join(DATA_DIR, item)
        dst = os.path.join("data", item)
        if not os.path.exists(dst) and os.path.exists(src):
            os.symlink(src, dst)

    sft_ckpt = os.path.join(CKPT_DIR, "patent_sft_final.pth")
    if not os.path.exists(sft_ckpt):
        print("ERROR: SFT checkpoint not found. Run train_sft first.")
        return 1

    cmd = [
        "torchrun", "--nproc_per_node=1",
        "post_train.py",
        "--data_path", "data",
        "--train_datasets", "pubchem,patent",
        "--valid_file", "real/patent_realimg_val.csv",
        "--vocab_file", "vocab/vocab_chars.json",
        "--formats", "grpo",
        "--dynamic_indigo",
        "--augment",
        "--mol_augment",
        "--include_condensed",
        "--encoder", "efficientvit",
        "--input_size", "512",
        "--use_qknorm",
        "--use_swiglu",
        "--use_rmsnorm",
        "--lora",
        "--encoder_lr", "0",
        "--decoder_lr", "5e-5",
        "--epochs", "2",
        "--batch_size", "4",
        "--accum_freq", "4",
        "--n_samples", "4",
        "--warmup_ratio", "0.05",
        "--scheduler", "cosine",
        "--load_path", sft_ckpt,
        "--load_model_only",
        "--smiles_only",
        "--amp",
        "--do_train",
        "--do_valid",
        "--save_mode", "all",
        "--print_freq", "20",
        "--num_workers", "4",
        "--backend", "gloo",
        "--exp_name", "patent_grpo",
    ]

    print(f"Running GRPO:\n  {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"Training exited with code {result.returncode}")

    # Copy final checkpoint
    grpo_runs = "runs/patent_grpo/ckpt_model"
    if os.path.isdir(grpo_runs):
        import shutil
        ckpts = sorted(
            [f for f in os.listdir(grpo_runs) if f.startswith("epoch_")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        if ckpts:
            final_ckpt = os.path.join(grpo_runs, ckpts[-1])
            dst = os.path.join(CKPT_DIR, "patent_grpo_final.pth")
            shutil.copy2(final_ckpt, dst)
            ckpt_vol.commit()
            print(f"Saved GRPO checkpoint: {ckpts[-1]} -> {dst}")

    return result.returncode


# ──────────────────────────────────────────────────────────────
# Download: pull fine-tuned checkpoint to local machine
# ──────────────────────────────────────────────────────────────


@app.function(
    volumes={CKPT_DIR: ckpt_vol},
    image=train_image,
    timeout=600,
)
def list_checkpoints():
    """List available checkpoints on the volume."""
    for f in sorted(os.listdir(CKPT_DIR)):
        size_mb = os.path.getsize(os.path.join(CKPT_DIR, f)) / (1024 * 1024)
        print(f"  {f} ({size_mb:.1f} MB)")


@app.local_entrypoint()
def download():
    """Download the fine-tuned checkpoint to local machine."""
    import tempfile

    # Prefer GRPO checkpoint, fall back to SFT
    for name in ("patent_grpo_final.pth", "patent_sft_final.pth"):
        local_dst = os.path.join(LOCAL_MOLSIGHT, name)
        try:
            # Read from volume
            data = b""
            for chunk in ckpt_vol.read_file(name):
                data += chunk
            with open(local_dst, "wb") as f:
                f.write(data)
            size_mb = len(data) / (1024 * 1024)
            print(f"Downloaded {name} ({size_mb:.1f} MB) to {local_dst}")
        except Exception as e:
            print(f"  {name} not found on volume: {e}")
            continue

    print("\nDone! To use the fine-tuned model:")
    print(f"  Copy checkpoint to {LOCAL_MOLSIGHT}/")
    print("  Then run the app with checkpoint_path parameter")
