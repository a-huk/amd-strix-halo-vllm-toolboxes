#!/usr/bin/env python3
import subprocess, time, json, sys, os, requests, argparse
from pathlib import Path

# =========================
# ⚙️ GLOBAL SETTINGS
# =========================

# HARDWARE: 1x Strix Halo (128GB, RDNA 3.5)
GPU_UTIL = "0.90" 
# 1. THROUGHPUT CONFIG
OFF_NUM_PROMPTS      = 200 
OFF_FORCED_OUTPUT    = "512"
# Default fallback if not specified in MODEL_TABLE
DEFAULT_BATCH_TOKENS = "8192"

# Fallbacks
FALLBACK_INPUT_LEN  = 1024
FALLBACK_OUTPUT_LEN = 512

RESULTS_DIR = Path("benchmark_results")
RESULTS_DIR.mkdir(exist_ok=True)

# =========================
# 🛠️ MODEL CONFIGURATION 🛠️
# =========================

MODEL_TABLE = {
    # 1. Llama 3.1 8B Instruct
    # MAD uses 131k tokens. We scale to 32k for 32GB VRAM safety.
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "trust_remote": False,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "32768" 
    },
    
    "google/gemma-3-12b-it": {
        "trust_remote": False,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "32768" 
    },
    # 2. GPT-OSS 20B (MXFP4)
    # MAD Row 0 uses 8192. We match this exactly.
    "openai/gpt-oss-20b": {
        "trust_remote": True,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "8192"
    },
    
    "openai/gpt-oss-120b": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "64",
        "max_tokens": "8192"
    },


    "Qwen/Qwen3-14B-AWQ": {
        "trust_remote": True,
        "valid_tp": [1], # Too big for single GPU
        "max_num_seqs": "32", # Lower concurrency for safety
        "max_tokens": "16384", # Lower batch size because Eager mode is CPU intensive
        "enforce_eager": False, 
        "env": {"VLLM_USE_TRITON_AWQ": "1"} # Fixes "Unsupported Hardware" error
    },

    # 4. Qwen 30B 4-bit
    "cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit": {
        "trust_remote": True,
        "enforce_eager": False, 
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "32768"
    },

    # 5. Qwen 80B AWQ (The Big One) [NEW]
    # Size: ~48GB. Fits on 2x32GB (64GB). Leftover for Cache: ~16GB.
    # Config: 20k ctx fits in that cache. Eager mode required for stability.
     "dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16": {
        "trust_remote": True,
        "valid_tp": [1], # Too big for single GPU
        "max_num_seqs": "32", # Lower concurrency for safety
        "max_tokens": "16384", # Lower batch size because Eager mode is CPU intensive
        "enforce_eager": True, 
        "env": {"VLLM_USE_TRITON_AWQ": "1"} # Fixes "Unsupported Hardware" error
    },

}

MODELS_TO_RUN = [
    #"meta-llama/Meta-Llama-3.1-8B-Instruct",
    #"google/gemma-3-12b-it",
    #"Qwen/Qwen3-14B-AWQ",
    #"openai/gpt-oss-20b",
    #"openai/gpt-oss-120b",
    "cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit",
    "dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16",
]

# =========================
# UTILS
# =========================

def log(msg): print(f"\n[BENCH] {msg}")

def get_gpu_count():
    try:
        # Using rocm-smi --showid to list GPUs. 
        # Output format: "GPU[0] : Device Name: ..."
        res = subprocess.run(["rocm-smi", "--showid"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            # Filter specifically for the target GPU as requested
            # target_gpu = "AMD Radeon AI PRO R9700"
            # count = 0
            # for line in res.stdout.strip().split('\n'):
            #     if "Device Name" in line and target_gpu in line:
            #         count += 1
            
            # return count if count > 0 else 1
            return 1 # Force return 1 for Strix Halo APU
        else:
            log("rocm-smi failed, defaulting to 1 GPU (Hardcoded Fallback)")
            return 1
    except Exception as e:
        log(f"Error detecting GPUs: {e}, defaulting to 1 GPU")
        return 1

def kill_vllm():
    subprocess.run("pgrep -f 'vllm serve' | xargs -r kill -9", 
                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)

def nuke_vllm_cache():
    cache = Path.home() / ".cache" / "vllm"
    if cache.exists():
        try:
            subprocess.run(["rm", "-rf", str(cache)], check=True)
            cache.mkdir(parents=True, exist_ok=True)
            time.sleep(2)
        except: pass

def get_dataset():
    data_path = Path("ShareGPT_V3_unfiltered_cleaned_split.json")
    if data_path.exists(): return str(data_path)
    
    log("Downloading ShareGPT dataset...")
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    try:
        r = requests.get(url, stream=True, timeout=15)
        r.raise_for_status()
        with open(data_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return str(data_path)
    except Exception as e:
        log(f"WARNING: ShareGPT download failed ({e}). using RANDOM.")
        return None



def get_model_args(model, tp_size):
    config = MODEL_TABLE.get(model, {"max_num_seqs": "32"})
    
    # Allow per-model GPU utilization override
    util = config.get("gpu_util", GPU_UTIL)

    cmd = [
        "--model", model,
        "--gpu-memory-utilization", util,
        "--dtype", "auto",
        "--tensor-parallel-size", str(tp_size),
        "--max-num-seqs", config["max_num_seqs"]
    ]
    
    # Optional: if a model really needs a hard limit, we can still support "ctx" in config,
    # but by default we rely on auto.
    if "ctx" in config:
        cmd.extend(["--max-model-len", config["ctx"]])
        
    if config.get("trust_remote"): cmd.append("--trust-remote-code")
    if config.get("enforce_eager"): cmd.append("--enforce-eager")
    
    return cmd

def run_throughput(model, tp_size):
    if tp_size not in MODEL_TABLE[model]["valid_tp"]: return
    
    model_safe = model.replace("/", "_")
    output_file = RESULTS_DIR / f"{model_safe}_tp{tp_size}_throughput.json"
    
    if output_file.exists():
        log(f"SKIP Throughput {model} (TP={tp_size})")
        return

    dataset_path = get_dataset()
    dataset_args = ["--dataset-name", "sharegpt", "--dataset-path", dataset_path] if dataset_path else ["--input-len", "1024"]
    
    # Retrieve Model-Specific Batch Tokens
    batch_tokens = MODEL_TABLE[model].get("max_tokens", DEFAULT_BATCH_TOKENS)

    log(f"START Throughput {model} (TP={tp_size}) [Batch: {batch_tokens}]...")
    kill_vllm()
    nuke_vllm_cache()

    cmd = ["vllm", "bench", "throughput"] + get_model_args(model, tp_size)
    cmd.extend([
        "--num-prompts", str(OFF_NUM_PROMPTS),
        "--max-num-batched-tokens", batch_tokens,
        "--output-len", OFF_FORCED_OUTPUT,
        "--output-json", str(output_file),
        "--disable-log-stats"
    ])
    cmd.extend(dataset_args)

    # ENV Setup: Global + Model Specific
    env = os.environ.copy()
    
    # Inject model specific env vars (e.g. for AWQ)
    model_env = MODEL_TABLE[model].get("env", {})
    env.update(model_env)

    try: 
        subprocess.run(cmd, check=True, env=env)
    except: 
        log(f"ERROR: Throughput failed {model}")



def print_summary(tps):
    print(f"\n{'MODEL':<40} | {'TP':<2} | {'TOK/S':<8}")
    print("-" * 60)
    
    for m in MODELS_TO_RUN:
        msafe = m.replace("/", "_")
        for tp in tps:
            if tp not in MODEL_TABLE[m]["valid_tp"]: continue
            
            try: 
                tdata = json.loads((RESULTS_DIR / f"{msafe}_tp{tp}_throughput.json").read_text())
                tok_s = f"{tdata.get('tokens_per_second', 0):.1f}"
            except: tok_s = "N/A"

            name_cell = m.split('/')[-1]
            print(f"{name_cell:<40} | {tp:<2} | {tok_s:<8}")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, nargs="+", default=[1])
    args = parser.parse_args()
    
    gpu_count = get_gpu_count()
    log(f"Detected {gpu_count} AMD GPU(s)")
    
    valid_tp_args = [t for t in args.tp if t <= gpu_count]
    if not valid_tp_args:
        log(f"Requested TP={args.tp} but only {gpu_count} GPU(s) detected. Nothing to run.")
        sys.exit(0)

    kill_vllm()
    for tp in valid_tp_args:
        for m in MODELS_TO_RUN:
            run_throughput(m, tp)
    print_summary(valid_tp_args)
