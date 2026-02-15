import time

import torch


def print_system_info() -> bool:
    """Print CPU/MPS availability and return whether MPS is available."""
    mps_available = torch.backends.mps.is_available()

    print("=== System Info ===")
    if mps_available:
        print("MPS is available")
        gpu_count = torch.mps.device_count()
        max_memory_gb = torch.mps.recommended_max_memory() / 1e9
        print(f"GPU count on macOS: {gpu_count}")
        print(f"Recommended max GPU working set: {max_memory_gb:.2f} GB")
    else:
        print("MPS is not available. CPU-only benchmark will run.")

    print()
    return mps_available


def run_matmul_timing(
    n: int,
    device: str,
    warmup_iters: int,
    measure_iters: int,
    dtype: torch.dtype,
) -> tuple[float, float]:
    """
    Time N x N matrix multiplication on one device.
    Returns:
        (avg_latency_ms, throughput_iter_per_sec)
    """
    a = torch.randn((n, n), device=device, dtype=dtype)
    b = torch.randn((n, n), device=device, dtype=dtype)

    for _ in range(warmup_iters):
        _ = a @ b

    if device == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(measure_iters):
        _ = a @ b

    if device == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start
    avg_latency_ms = (elapsed / measure_iters) * 1000.0
    throughput = measure_iters / elapsed
    return avg_latency_ms, throughput


def estimate_working_set_gb(n: int, dtype: torch.dtype) -> float:
    """Approximate memory for A, B, and output of matmul."""
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    total_bytes = 3 * n * n * bytes_per_element
    return total_bytes / 1_000_000_000


def benchmark() -> None:
    matrix_sizes = [1024, 2048, 4096]  # each value is N for N x N matrices
    warmup_iters = 5
    measure_iters = 20
    dtype = torch.float32

    mps_available = print_system_info()
    devices = ["cpu", "mps"] if mps_available else ["cpu"]

    print("=== CPU vs MPS Matmul Benchmark ===")
    print(f"dtype={dtype}, warmup={warmup_iters}, measured={measure_iters}")
    print()
    print(
        "size | est_mem(GB) | device | avg_latency(ms) | throughput(iter/s)"
    )
    print("-" * 66)

    results: dict[int, dict[str, tuple[float, float]]] = {}

    for n in matrix_sizes:
        results[n] = {}
        working_set_gb = estimate_working_set_gb(n, dtype)

        for device in devices:
            latency_ms, throughput = run_matmul_timing(
                n=n,
                device=device,
                warmup_iters=warmup_iters,
                measure_iters=measure_iters,
                dtype=dtype,
            )
            results[n][device] = (latency_ms, throughput)
            print(
                f"{n:>4} | {working_set_gb:>11.3f} | {device:>6} |"
                f" {latency_ms:>15.3f} | {throughput:>17.2f}"
            )

        if "cpu" in results[n] and "mps" in results[n]:
            cpu_latency = results[n]["cpu"][0]
            mps_latency = results[n]["mps"][0]
            speedup = cpu_latency / mps_latency
            print(f"     speedup (cpu/mps): {speedup:.2f}x")
            print("-" * 66)


if __name__ == "__main__":
    benchmark()
