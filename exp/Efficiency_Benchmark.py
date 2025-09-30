# benchmark_efficiency.py
import os, math, time, json, random
import torch
import torch.nn as nn
import pandas as pd
from dataloaders.prepare.six.six_datasets_loader import load_data
torch.backends.cudnn.benchmark = True
DEVICE = "cuda"

# -----------------------
# Utils
# -----------------------
def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def measure_inference(model, B, L, D, warmup=10, iters=50, amp=True):
    model.eval().to(DEVICE)
    x = torch.randn(B, L, D, device=DEVICE)
    torch.cuda.reset_peak_memory_stats()
    # warmup
    for _ in range(warmup):
        with torch.cuda.amp.autocast(enabled=amp):
            _ = model(x)
    torch.cuda.synchronize()
    # measure
    starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
    times = []
    for _ in range(iters):
        with torch.cuda.amp.autocast(enabled=amp):
            starter.record()
            _ = model(x)
            ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))  # ms
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    return float(torch.tensor(times).mean()), float(torch.tensor(times).std()), peak_mem

def measure_training_step(model, B, L, D, warmup=10, iters=50, amp=True, lr=1e-3):
    model.train().to(DEVICE)
    x = torch.randn(B, L, D, device=DEVICE)
    y = torch.randn(B, L, D, device=DEVICE)  # 假设输出同形
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loss_fn = nn.MSELoss()

    # warmup
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(x)
            loss = loss_fn(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
    times = []
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(x)
            loss = loss_fn(out, y)
        starter.record()
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))  # ms

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    return float(torch.tensor(times).mean()), float(torch.tensor(times).std()), peak_mem


# -----------------------
# Main sweep
# -----------------------
def run_sweep(models, L_list, B=16, D=512, amp=True):
    set_seed(42)
    results = []

    for name, model in models.items():
        print(f"\n==> Benchmark {name}")
        params = sum(p.numel() for p in model.parameters()) / 1e6

        for L in L_list:
            inf_mean, inf_std, inf_mem = measure_inference(model, B, L, D, amp=amp)
            tr_mean, tr_std, tr_mem = measure_training_step(model, B, L, D, amp=amp)

            row = {
                "model": name, "params_M": params,
                "L": L, "B": B, "D": D,
                "infer_ms": inf_mean, "infer_std": inf_std, "infer_mem_MB": inf_mem,
                "train_ms": tr_mean,  "train_std": tr_std, "train_mem_MB": tr_mem,
                "throughput_infer_tok_per_s": (B*L) / (inf_mean/1000.0),
                "throughput_train_tok_per_s": (B*L) / (tr_mean/1000.0),
                "type": "synthetic"
            }

            print(f"[{name}] L={L}: "
                  f"infer {inf_mean:.1f}±{inf_std:.1f} ms, mem {inf_mem:.0f} MB | "
                  f"train {tr_mean:.1f}±{tr_std:.1f} ms, mem {tr_mem:.0f} MB")

            results.append(row)

    return results


def Real_data_practical_efficiency(models, dataset_path, L_list, B=16, amp=True):
    set_seed(42)
    results = []

    for name, model in models.items():
        print(f"\n==> Real-data Benchmark {name}")
        params = sum(p.numel() for p in model.parameters()) / 1e6

        for L in L_list:
            train_loader, val_loader, test_loader, scaler = load_data(dataset_path, L, L, B)
            x_real, y_real = next(iter(test_loader))
            x_real, y_real = x_real.to(DEVICE), y_real.to(DEVICE)

            # 测推理
            model.eval()
            with torch.no_grad():
                starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
                torch.cuda.reset_peak_memory_stats()
                with torch.cuda.amp.autocast(enabled=amp):
                    starter.record()
                    _ = model(x_real)
                    ender.record()
                torch.cuda.synchronize()
                inf_time = starter.elapsed_time(ender)
            inf_mem = torch.cuda.max_memory_allocated() / (1024**2)

            row = {
                "model": name, "params_M": params,
                "L": L, "B": B,
                "infer_ms_real": inf_time,
                "infer_mem_real_MB": inf_mem,
                "type": "real"
            }
            print(f"[{name}] L={L}: real-data infer {inf_time:.1f} ms, mem {inf_mem:.0f} MB")
            results.append(row)

    return results



# -----------------------
# Save results to CSV per model
# -----------------------
def save_results_to_csv(synthetic_results, real_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 按模型名分组
    model_names = set([row["model"] for row in synthetic_results])
    for model_name in model_names:
        # 筛选该模型的数据
        syn_rows = [r for r in synthetic_results if r["model"] == model_name]
        real_rows = [r for r in real_results if r["model"] == model_name]

        # 按 L 对齐
        L_list = sorted(set([r["L"] for r in syn_rows] + [r["L"] for r in real_rows]))
        merged_rows = []
        for L in L_list:
            syn_row = next((r for r in syn_rows if r["L"] == L), {})
            real_row = next((r for r in real_rows if r["L"] == L), {})

            row = {
                "model": model_name,
                "params_M": syn_row.get("params_M", real_row.get("params_M", "")),
                "L": L,
                "B": syn_row.get("B", real_row.get("B", "")),
                "D": syn_row.get("D", ""),
                "infer_ms": syn_row.get("infer_ms", ""),
                "infer_std": syn_row.get("infer_std", ""),
                "infer_mem_MB": syn_row.get("infer_mem_MB", ""),
                "train_ms": syn_row.get("train_ms", ""),
                "train_std": syn_row.get("train_std", ""),
                "train_mem_MB": syn_row.get("train_mem_MB", ""),
                "throughput_infer_tok_per_s": syn_row.get("throughput_infer_tok_per_s", ""),
                "throughput_train_tok_per_s": syn_row.get("throughput_train_tok_per_s", ""),
                "infer_ms_real": real_row.get("infer_ms_real", ""),
                "infer_mem_real_MB": real_row.get("infer_mem_real_MB", "")
            }
            merged_rows.append(row)

        # 保存 CSV
        csv_path = os.path.join(save_dir, f"{model_name}.csv")
        df = pd.DataFrame(merged_rows)
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV for {model_name} at {csv_path}")


import os
import pandas as pd

def save_synthetic_only_csv(synthetic_results, save_dir):
    """
    仅保存合成数据（synthetic_results），忽略真实数据列。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 按模型名分组
    model_names = set([row["model"] for row in synthetic_results])
    for model_name in model_names:
        # 筛选该模型的合成数据
        syn_rows = [r for r in synthetic_results if r["model"] == model_name]

        # 按 L 排序
        L_list = sorted(set([r["L"] for r in syn_rows]))
        rows_to_save = []
        for L in L_list:
            syn_row = next((r for r in syn_rows if r["L"] == L), {})
            row = {
                "model": model_name,
                "params_M": syn_row.get("params_M", ""),
                "L": L,
                "B": syn_row.get("B", ""),
                "D": syn_row.get("D", ""),
                "infer_ms": syn_row.get("infer_ms", ""),
                "infer_std": syn_row.get("infer_std", ""),
                "infer_mem_MB": syn_row.get("infer_mem_MB", ""),
                "train_ms": syn_row.get("train_ms", ""),
                "train_std": syn_row.get("train_std", ""),
                "train_mem_MB": syn_row.get("train_mem_MB", ""),
                "throughput_infer_tok_per_s": syn_row.get("throughput_infer_tok_per_s", ""),
                "throughput_train_tok_per_s": syn_row.get("throughput_train_tok_per_s", "")
            }
            rows_to_save.append(row)

        # 保存 CSV
        csv_path = os.path.join(save_dir, f"{model_name}_synthetic.csv")
        df = pd.DataFrame(rows_to_save)
        df.to_csv(csv_path, index=False)
        print(f"Saved synthetic-only CSV for {model_name} at {csv_path}")


# -----------------------
# Main function
# -----------------------
def benchmark_efficiency(models, dataset_path, L_list, B=16, D=512, amp=True):
    # B1: 合成数据
    results_synthetic = run_sweep(models, L_list, B, D, amp)

    # B2: 真实数据
    results_real = Real_data_practical_efficiency(models, dataset_path, L_list, B, amp)

    # 保存 CSV，每个模型一个文件
    save_results_to_csv(results_synthetic, results_real, save_dir=dataset_path)

    # save_synthetic_only_csv(results_synthetic, save_dir=dataset_path)





if __name__ == "__main__":
    assert torch.cuda.is_available()
    # 这里 MODELS 应该是 dict 而不是 list, 否则上面的 .items() 会报错
    MODELS = {
        "koslm": torch.nn.Identity(),   # placeholder
        "transformer": torch.nn.Identity(),
        "transformer_flash": torch.nn.Identity(),
        "mamba": torch.nn.Identity(),
    }
    L_LIST = [128, 256, 512, 1024, 2048, 4096, 8192]
    benchmark_efficiency(MODELS, "./data/ETTm2", L_LIST, B=16, D=512, amp=True)
