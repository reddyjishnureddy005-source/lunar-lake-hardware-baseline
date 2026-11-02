import csv
import gc
import sys
import time
import openvino_genai as ov_genai

MODEL_PATH = "models/Qwen2.5-1.5B-Instruct-int4-ov"
DEVICES = ["CPU", "GPU", "NPU"]
WARMUP_RUNS = 1
TIMED_RUNS = 5
MAX_NEW_TOKENS = 64
PROMPT = ["Explain the difference between a stack and a queue in three concise sentences."]

def build_pipeline(model_path: str, device: str) -> ov_genai.LLMPipeline:
    if device == "NPU":
        return ov_genai.LLMPipeline(model_path, device)
    scheduler_config = ov_genai.SchedulerConfig()
    scheduler_config.enable_prefix_caching = False
    scheduler_config.max_num_batched_tokens = sys.maxsize
    return ov_genai.LLMPipeline(model_path, device, scheduler_config=scheduler_config)

def run_benchmark():
    results = []

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = MAX_NEW_TOKENS
    config.ignore_eos = True
    config.apply_chat_template = False

    for device in DEVICES:
        print(f"\n=== Target Device: {device} ===")
        try:
            compile_start = time.perf_counter()
            pipe = build_pipeline(MODEL_PATH, device)
            compile_time = time.perf_counter() - compile_start
            print(f"Compiled in {compile_time:.2f}s | Running {WARMUP_RUNS} warmup...")

            for _ in range(WARMUP_RUNS):
                pipe.generate(PROMPT, config)

            for run_idx in range(1, TIMED_RUNS + 1):
                res = pipe.generate(PROMPT, config)
                perf = res.perf_metrics

                ttft_ms = perf.get_ttft().mean
                tpot_ms = perf.get_tpot().mean
                tps = perf.get_throughput().mean
                gen_tokens = perf.get_num_generated_tokens()

                print(f"Run {run_idx}: TTFT = {ttft_ms:.2f} ms | TPOT = {tpot_ms:.2f} ms | Throughput = {tps:.2f} tok/s ({gen_tokens} tokens)")

                results.append({
                    "device": device,
                    "run": run_idx,
                    "ttft_ms": round(ttft_ms, 2),
                    "tpot_ms": round(tpot_ms, 2),
                    "tokens_per_sec": round(tps, 2),
                    "generated_tokens": gen_tokens,
                    "compile_time_sec": round(compile_time, 2)
                })

            del pipe
            gc.collect()

        except Exception as e:
            print(f"Execution failed on {device}: {e}")

    csv_file = "hardware_baseline.csv"
    fieldnames = ["device", "run", "ttft_ms", "tpot_ms", "tokens_per_sec", "generated_tokens", "compile_time_sec"]
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved baseline metrics to {csv_file}")

if __name__ == "__main__":
    run_benchmark()