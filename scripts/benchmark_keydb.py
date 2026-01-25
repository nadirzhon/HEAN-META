#!/usr/bin/env python3
"""
Benchmark KeyDB vs Redis performance.

This script compares the performance of KeyDB against Redis
for common operations.

Usage:
    python benchmark_keydb.py --host localhost --port 6379

Requirements:
    pip install redis
"""

import argparse
import time
from typing import Dict

try:
    import redis
except ImportError:
    print("Error: Missing redis package. Install with: pip install redis")
    exit(1)


def benchmark_set(client: redis.Redis, num_ops: int = 10000) -> float:
    """Benchmark SET operations."""
    start = time.time()
    for i in range(num_ops):
        client.set(f"bench:set:{i}", f"value_{i}")
    elapsed = time.time() - start
    return num_ops / elapsed


def benchmark_get(client: redis.Redis, num_ops: int = 10000) -> float:
    """Benchmark GET operations."""
    # Prepare data
    for i in range(num_ops):
        client.set(f"bench:get:{i}", f"value_{i}")

    start = time.time()
    for i in range(num_ops):
        client.get(f"bench:get:{i}")
    elapsed = time.time() - start
    return num_ops / elapsed


def benchmark_hset(client: redis.Redis, num_ops: int = 10000) -> float:
    """Benchmark HSET operations."""
    start = time.time()
    for i in range(num_ops):
        client.hset(f"bench:hash:{i}", "field1", f"value_{i}")
    elapsed = time.time() - start
    return num_ops / elapsed


def benchmark_lpush(client: redis.Redis, num_ops: int = 10000) -> float:
    """Benchmark LPUSH operations."""
    start = time.time()
    for i in range(num_ops):
        client.lpush("bench:list", f"value_{i}")
    elapsed = time.time() - start
    return num_ops / elapsed


def benchmark_sadd(client: redis.Redis, num_ops: int = 10000) -> float:
    """Benchmark SADD operations."""
    start = time.time()
    for i in range(num_ops):
        client.sadd("bench:set", f"member_{i}")
    elapsed = time.time() - start
    return num_ops / elapsed


def benchmark_zadd(client: redis.Redis, num_ops: int = 10000) -> float:
    """Benchmark ZADD operations."""
    start = time.time()
    for i in range(num_ops):
        client.zadd("bench:zset", {f"member_{i}": i})
    elapsed = time.time() - start
    return num_ops / elapsed


def benchmark_pipeline(client: redis.Redis, num_ops: int = 10000) -> float:
    """Benchmark pipelined operations."""
    start = time.time()
    pipe = client.pipeline()
    for i in range(num_ops):
        pipe.set(f"bench:pipe:{i}", f"value_{i}")
    pipe.execute()
    elapsed = time.time() - start
    return num_ops / elapsed


def run_benchmarks(
    host: str = "localhost",
    port: int = 6379,
    password: str | None = None,
    num_ops: int = 10000,
) -> Dict[str, float]:
    """Run all benchmarks and return results."""
    print(f"🔄 Connecting to {host}:{port}...")

    try:
        client = redis.Redis(host=host, port=port, password=password, db=0)
        client.ping()
        print("✅ Connected successfully")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        exit(1)

    # Get server info
    info = client.info()
    server_version = info.get("redis_version", "unknown")
    print(f"📊 Server version: {server_version}")
    print(f"📊 Memory used: {info.get('used_memory_human', 'unknown')}")
    print()

    results = {}

    print(f"🚀 Running benchmarks ({num_ops} operations each)...\n")

    # SET
    print("⏳ Benchmarking SET...", end=" ", flush=True)
    results["SET"] = benchmark_set(client, num_ops)
    print(f"{results['SET']:.0f} ops/sec")

    # GET
    print("⏳ Benchmarking GET...", end=" ", flush=True)
    results["GET"] = benchmark_get(client, num_ops)
    print(f"{results['GET']:.0f} ops/sec")

    # HSET
    print("⏳ Benchmarking HSET...", end=" ", flush=True)
    results["HSET"] = benchmark_hset(client, num_ops)
    print(f"{results['HSET']:.0f} ops/sec")

    # LPUSH
    print("⏳ Benchmarking LPUSH...", end=" ", flush=True)
    results["LPUSH"] = benchmark_lpush(client, num_ops)
    print(f"{results['LPUSH']:.0f} ops/sec")

    # SADD
    print("⏳ Benchmarking SADD...", end=" ", flush=True)
    results["SADD"] = benchmark_sadd(client, num_ops)
    print(f"{results['SADD']:.0f} ops/sec")

    # ZADD
    print("⏳ Benchmarking ZADD...", end=" ", flush=True)
    results["ZADD"] = benchmark_zadd(client, num_ops)
    print(f"{results['ZADD']:.0f} ops/sec")

    # Pipeline
    print("⏳ Benchmarking PIPELINE...", end=" ", flush=True)
    results["PIPELINE"] = benchmark_pipeline(client, num_ops)
    print(f"{results['PIPELINE']:.0f} ops/sec")

    # Cleanup
    print("\n🧹 Cleaning up benchmark data...")
    for pattern in ["bench:*"]:
        keys = client.keys(pattern)
        if keys:
            client.delete(*keys)

    return results


def print_results(results: Dict[str, float]):
    """Print formatted benchmark results."""
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Operation':<15} {'Throughput':>20}")
    print("-" * 60)

    for operation, ops_per_sec in sorted(
        results.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{operation:<15} {ops_per_sec:>15,.0f} ops/sec")

    print("=" * 60)

    # Calculate average
    avg = sum(results.values()) / len(results)
    print(f"{'Average':<15} {avg:>15,.0f} ops/sec")
    print()


def compare_with_baseline(results: Dict[str, float], baseline: Dict[str, float]):
    """Compare results with baseline (e.g., Redis)."""
    print("\n" + "=" * 60)
    print("📈 IMPROVEMENT vs BASELINE")
    print("=" * 60)
    print(f"{'Operation':<15} {'Improvement':>20}")
    print("-" * 60)

    for operation in results:
        if operation in baseline:
            improvement = (results[operation] / baseline[operation] - 1) * 100
            sign = "+" if improvement > 0 else ""
            print(f"{operation:<15} {sign}{improvement:>14.1f}%")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark KeyDB/Redis performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--host", default="localhost", help="Server hostname (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=6379, help="Server port (default: 6379)"
    )
    parser.add_argument("--password", help="Server password")
    parser.add_argument(
        "--num-ops",
        type=int,
        default=10000,
        help="Number of operations per benchmark (default: 10000)",
    )

    args = parser.parse_args()

    results = run_benchmarks(
        host=args.host,
        port=args.port,
        password=args.password,
        num_ops=args.num_ops,
    )

    print_results(results)

    # Example: Expected KeyDB improvements over Redis
    # KeyDB is typically 2-5x faster for multi-threaded workloads
    print("\n💡 Tip: KeyDB typically shows 2-5x improvement over Redis")
    print("   for multi-threaded workloads due to its multi-core support.")


if __name__ == "__main__":
    main()
