#!/usr/bin/env python3
import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

import paddle


def build_ext(verbose: bool, rebuild: bool):
    here = Path(__file__).resolve().parent
    so_candidates = [here / "repro_ext.so", here / "repro_ext_pd_.so"]
    if rebuild:
        subprocess.call(
            ["rm", "-rf", "build", "repro_ext.so", "repro_ext_pd_.so"], cwd=here
        )
    if not any(path.exists() for path in so_candidates):
        cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
        if verbose:
            print(f"[py] building extension: {' '.join(cmd)}", flush=True)
        subprocess.check_call(cmd, cwd=here)
    sys.path.insert(0, str(here))
    return importlib.import_module("repro_ext")


def wait_until(predicate, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Minimal Paddle C++ ATen current-stream deadlock reproducer."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument(
        "--rebuild", action="store_true", help="Force rebuilding repro_ext."
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="Set the host flag before C++ at_tensor[0]=0; this should not hang.",
    )
    parser.add_argument(
        "--set-pool-stream",
        action="store_true",
        help="Call setCurrentCUDAStream with a pool stream before cycle.",
    )
    parser.add_argument(
        "--cpu-sync",
        action="store_true",
        help="Use CPU-side cudaEventSynchronize instead of blocking the stream.",
    )
    args = parser.parse_args()

    if not paddle.is_compiled_with_cuda():
        print("Paddle is not compiled with CUDA", file=sys.stderr)
        return 1

    paddle.set_device(f"gpu:{args.device}")
    ext = build_ext(args.verbose_build, args.rebuild)

    x = paddle.ones([4], dtype="int32")
    raw_before = ext.current_stream_raw(args.device)
    print(
        f"[py] tensor_place={x.place} current_stream_before=0x{raw_before:x} "
        f"control={args.control} set_pool_stream={args.set_pool_stream} "
        f"cpu_sync={args.cpu_sync}",
        flush=True,
    )

    if args.set_pool_stream:
        ext.set_pool_stream(args.device)
        raw_after_set = ext.current_stream_raw(args.device)
        print(f"[py] current_stream_after_set=0x{raw_after_set:x}", flush=True)

    # Build the minimal current_stream -> enq_stream -> event_end -> current_stream cycle.
    ext.start_cycle(args.device)

    if args.control:
        ext.set_host_flag()

    # 第二种解法：先启动 worker，再做 CPU-side sync（避免主线程阻塞在 worker 之前）。
    ext.start_cpp_tensor_set_worker(args.device, not args.control)

    ext.block_current_stream_on_event_end(args.device, args.cpu_sync)
    raw_after_block = ext.current_stream_raw(args.device)
    print(f"[py] current_stream_after_block=0x{raw_after_block:x}", flush=True)
    if not wait_until(ext.cpp_worker_entered, args.timeout):
        print("[py] worker did not enter assignment", flush=True)
        os._exit(3)

    if not wait_until(ext.cpp_worker_done, args.timeout):
        print(
            f"[py] REPRODUCED: C++ worker did not return within {args.timeout}s.\n"
            "[py] C++ at_tensor[0]=0 is stuck before it can set host_flag,\n"
            "[py] while event_end waits for host_flag and current_stream waits for event_end.",
            flush=True,
        )
        os._exit(2)

    ext.join_cpp_worker()
    if not args.control:
        ext.set_host_flag()
    wait_until(ext.event_end_ready, args.timeout)
    ext.cleanup()
    print("[py] finished without hang", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
