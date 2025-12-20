
import os
import json
import re
from pathlib import Path



# Config
SCRIPT_DIR = Path(__file__).parent.resolve()
BENCHMARK_SOURCES = {
    "Triton": SCRIPT_DIR.parent / "benchmarks" / "benchmark_results",
    "ROCm": SCRIPT_DIR.parent / "benchmarks" / "benchmark_results_rocm_attn" / "benchmark_results"
}
OUTPUT_FILE = SCRIPT_DIR / "results.json"

# Regex to parse model name for quantization and parameters
PARAMS_REGEX = r"(\d+(?:\.\d+)?)B"
QUANT_REGEX = r"(FP8|AWQ|GPTQ|BF16|4bit|Int4)"

def extract_meta(model_name):
    # Params
    params_match = re.search(PARAMS_REGEX, model_name, re.IGNORECASE)
    params_b = float(params_match.group(1)) if params_match else None
    
    # Quant
    quant_match = re.search(QUANT_REGEX, model_name, re.IGNORECASE)
    quant = quant_match.group(1).upper() if quant_match else "BF16"
    
    # Refine quant if 4bit
    if quant == "4BIT" or quant == "INT4":
        if "GPTQ" in model_name: quant = "GPTQ-4bit"
        elif "AWQ" in model_name: quant = "AWQ-4bit"
        else: quant = "4-bit"

    return params_b, quant

def parse_logs():
    runs = []
    
    for backend_name, bench_dir in BENCHMARK_SOURCES.items():
        if not bench_dir.exists():
            print(f"Warning: {bench_dir} does not exist, skipping.")
            continue

        print(f"Scanning {bench_dir} for {backend_name} results...")
        files = list(bench_dir.glob("*.json"))
        
        for f in files:
            fname = f.name
            try:
                data = json.loads(f.read_text())
            except:
                print(f"Skipping bad JSON: {fname}")
                continue

            # Filename parsing
            parts = fname.split("_tp")
            if len(parts) < 2: continue
            
            model_part = parts[0]
            rest = parts[1] # "1_throughput.json"
            
            # TP
            tp_match = re.match(r"^(\d+)", rest)
            if not tp_match: continue
            tp = int(tp_match.group(1))
            
            # Model Name
            if "_" in model_part:
                model_display = model_part.replace("_", "/", 1)
            else:
                model_display = model_part
                
            params_b, quant = extract_meta(model_display)
            
            base_run = {
                "model": model_display,
                "model_clean": model_display,
                "env": f"TP{tp}",
                "gpu_config": "dual" if tp > 1 else "single",
                "quant": quant,
                "params_b": params_b,
                "name_params_b": params_b,
                "backend": backend_name, # "Triton" or "ROCm"
                "error": False
            }

            if "throughput" in fname:
                tps = data.get("tokens_per_second", 0)
                run = base_run.copy()
                run["test"] = "Throughput"
                run["tps_mean"] = tps
                if tps == 0 or (isinstance(data, dict) and "error" in str(data).lower()): # checking if error string is in json dump
                     run["error"] = True
                runs.append(run)

            elif "latency" in fname:
                raw = data.get("raw_output", "")
                qps_match = re.search(r"_qps([\d\.]+)_", fname)
                qps = qps_match.group(1) if qps_match else "?"
                
                ttft = 0.0
                tpot = 0.0
                
                ttft_m = re.search(r"(?:Mean TTFT|TTFT).*?([\d\.]+)", raw)
                if ttft_m: ttft = float(ttft_m.group(1))
                
                tpot_m = re.search(r"(?:Mean TPOT|TPOT).*?([\d\.]+)", raw)
                if tpot_m: tpot = float(tpot_m.group(1))
                
                # TTFT
                r1 = base_run.copy()
                r1["test"] = f"TTFT @ QPS {qps}"
                r1["tps_mean"] = ttft
                runs.append(r1)
                
                # TPOT
                r2 = base_run.copy()
                r2["test"] = f"TPOT @ QPS {qps}"
                r2["tps_mean"] = tpot
                runs.append(r2)

    return runs

if __name__ == "__main__":
    data = {"runs": parse_logs()}
    
    runs_count = len(data["runs"])
    print(f"Parsed {runs_count} runs.")
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Written to {OUTPUT_FILE}")
