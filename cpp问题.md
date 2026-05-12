【torch 兼容接口】Paddle compat 的 at::cuda::getCurrentCUDAStream(device) 在后台线程 B 里返回了和主线程 A 相同的 current stream。PyTorch 语义里 current stream 通常是 thread-local 的，后台线程 B 无意中继承/共享调用方已被 block 的 current stream。所以这更像是 Paddle compat current stream 管理不符合 PyTorch 语义，或者 Paddle 的 device context stream 是 per-device global，而不是 per-thread current stream。最终会造成循环依赖问题，如下图所示。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=158d8f4a7bb0424c8b131ea45113ac55&docGuid=0agAUwXhwHkY2D)
如何复现 bug：

将下面三个文件放在同一文件夹下，运行 `python repro.py --device 0 --timeout 5 --verbose-build --rebuild`。如果已经构建过就去掉 `--verbose-build --rebuild`。

```python
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
        subprocess.call(["rm", "-rf", "build", "repro_ext.so", "repro_ext_pd_.so"], cwd=here)
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
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding repro_ext.")
    parser.add_argument(
        "--control",
        action="store_true",
        help="Set the host flag before C++ at_tensor[0]=0; this should not hang.",
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
        f"[py] tensor_place={x.place} current_stream_before=0x{raw_before:x} control={args.control}",
        flush=True,
    )

    # Build the minimal current_stream -> enq_stream -> event_end -> current_stream cycle.
    ext.start_cycle(args.device)
    ext.block_current_stream_on_event_end(args.device)
    raw_after_block = ext.current_stream_raw(args.device)
    print(f"[py] current_stream_after_block=0x{raw_after_block:x}", flush=True)

    if args.control:
        ext.set_host_flag()

    ext.start_cpp_tensor_set_worker(x, args.device, not args.control)
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

```
```python
import paddle
paddle.enable_compat()

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="repro_ext",
    ext_modules=[CUDAExtension(
        name="repro_ext",
        sources=["repro_ext.cu"],
        extra_compile_args={
            "cxx": ["-O0", "-g", "-pthread"],
            "nvcc": ["-O0", "-g"],
        },
        extra_link_args=["-pthread"],
    )],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)

```
```cpp
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/as_strided.h>
#include <cuda_runtime.h>
#include <paddle/utils/pybind.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace py = pybind11;

struct State {
    cudaStream_t enq_stream = nullptr;
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_end = nullptr;
    int* host_flag = nullptr;
    int* device_flag = nullptr;
};

static State g_state;
static std::atomic<bool> g_cpp_worker_done{false};
static std::atomic<bool> g_cpp_worker_entered{false};
static std::thread g_cpp_worker;

static void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

__global__ void wait_for_host_flag(int* flag) {
    while (atomicAdd(flag, 0) == 0) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(1000);
#endif
    }
}

uintptr_t current_stream_raw(int device) {
    auto stream = at::cuda::getCurrentCUDAStream(device);
    return reinterpret_cast<uintptr_t>(stream.raw_stream());
}

void start_cycle(int device) {
    check(cudaSetDevice(device), "cudaSetDevice");
    if (g_state.enq_stream != nullptr) {
        throw std::runtime_error("start_cycle called twice without cleanup");
    }

    auto current = at::cuda::getCurrentCUDAStream(device);
    auto current_raw = current.raw_stream();

    check(cudaStreamCreateWithFlags(&g_state.enq_stream, cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags(enq_stream)");
    check(cudaEventCreateWithFlags(&g_state.event_start, cudaEventDisableTiming),
          "cudaEventCreateWithFlags(event_start)");
    check(cudaEventCreateWithFlags(&g_state.event_end, cudaEventDisableTiming),
          "cudaEventCreateWithFlags(event_end)");
    check(cudaHostAlloc(&g_state.host_flag, sizeof(int), cudaHostAllocMapped),
          "cudaHostAlloc(host_flag)");
    *g_state.host_flag = 0;
    check(cudaHostGetDevicePointer(&g_state.device_flag, g_state.host_flag, 0),
          "cudaHostGetDevicePointer(device_flag)");

    check(cudaEventRecord(g_state.event_start, current_raw),
          "cudaEventRecord(event_start,current)");
    check(cudaStreamWaitEvent(g_state.enq_stream, g_state.event_start, 0),
          "cudaStreamWaitEvent(enq,event_start)");

    wait_for_host_flag<<<1, 1, 0, g_state.enq_stream>>>(g_state.device_flag);
    check(cudaGetLastError(), "wait_for_host_flag launch");

    check(cudaEventRecord(g_state.event_end, g_state.enq_stream),
          "cudaEventRecord(event_end,enq)");

    std::cerr << "[ext] start_cycle device=" << device
              << " current_stream=" << reinterpret_cast<void*>(current_raw)
              << " enq_stream=" << reinterpret_cast<void*>(g_state.enq_stream)
              << std::endl;
}

void block_current_stream_on_event_end(int device) {
    check(cudaSetDevice(device), "cudaSetDevice");
    if (g_state.event_end == nullptr) {
        throw std::runtime_error("block_current_stream_on_event_end before start_cycle");
    }

    auto current = at::cuda::getCurrentCUDAStream(device);
    auto current_raw = current.raw_stream();
    check(cudaStreamWaitEvent(current_raw, g_state.event_end, 0),
          "cudaStreamWaitEvent(current,event_end)");

    std::cerr << "[ext] block_current_stream_on_event_end device=" << device
              << " current_stream=" << reinterpret_cast<void*>(current_raw)
              << " waits event_end" << std::endl;
}

void set_host_flag() {
    if (g_state.host_flag == nullptr) {
        throw std::runtime_error("set_host_flag before start_cycle");
    }
    *g_state.host_flag = 1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    std::cerr << "[ext] host_flag set" << std::endl;
}

void start_cpp_tensor_set_worker(at::Tensor tensor, int device, bool set_flag_after_assignment) {
    if (g_cpp_worker.joinable()) {
        throw std::runtime_error("cpp worker already running");
    }
    g_cpp_worker_done.store(false, std::memory_order_release);
    g_cpp_worker_entered.store(false, std::memory_order_release);

    g_cpp_worker = std::thread([tensor, device, set_flag_after_assignment]() mutable {
        check(cudaSetDevice(device), "cpp worker cudaSetDevice");
        auto current = at::cuda::getCurrentCUDAStream(device);
        auto current_raw = current.raw_stream();
        std::cerr << "[cpp worker] current_stream="
                  << reinterpret_cast<void*>(current_raw) << std::endl;
        std::cerr << "[cpp worker] before at_tensor[0] = 0" << std::endl;
        g_cpp_worker_entered.store(true, std::memory_order_release);
        tensor[0] = 0;
        std::cerr << "[cpp worker] after at_tensor[0] = 0" << std::endl;
        if (set_flag_after_assignment) {
            set_host_flag();
        }
        g_cpp_worker_done.store(true, std::memory_order_release);
    });
}

bool cpp_worker_entered() {
    return g_cpp_worker_entered.load(std::memory_order_acquire);
}

bool cpp_worker_done() {
    return g_cpp_worker_done.load(std::memory_order_acquire);
}

void join_cpp_worker() {
    if (g_cpp_worker.joinable()) {
        g_cpp_worker.join();
    }
}

bool event_end_ready() {
    if (g_state.event_end == nullptr) {
        return false;
    }
    auto err = cudaEventQuery(g_state.event_end);
    if (err == cudaSuccess) {
        return true;
    }
    if (err == cudaErrorNotReady) {
        (void)cudaGetLastError();
        return false;
    }
    check(err, "cudaEventQuery(event_end)");
    return false;
}

void cleanup() {
    if (g_state.host_flag != nullptr) {
        *g_state.host_flag = 1;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    if (g_cpp_worker.joinable()) {
        g_cpp_worker.join();
    }
    if (g_state.enq_stream != nullptr) {
        cudaStreamSynchronize(g_state.enq_stream);
        cudaStreamDestroy(g_state.enq_stream);
        g_state.enq_stream = nullptr;
    }
    if (g_state.event_start != nullptr) {
        cudaEventDestroy(g_state.event_start);
        g_state.event_start = nullptr;
    }
    if (g_state.event_end != nullptr) {
        cudaEventDestroy(g_state.event_end);
        g_state.event_end = nullptr;
    }
    if (g_state.host_flag != nullptr) {
        cudaFreeHost(g_state.host_flag);
        g_state.host_flag = nullptr;
        g_state.device_flag = nullptr;
    }
}

PYBIND11_MODULE(repro_ext, m) {
    m.def("current_stream_raw", &current_stream_raw);
    m.def("start_cycle", &start_cycle);
    m.def("block_current_stream_on_event_end", &block_current_stream_on_event_end);
    m.def("set_host_flag", &set_host_flag);
    m.def("start_cpp_tensor_set_worker", &start_cpp_tensor_set_worker);
    m.def("cpp_worker_entered", &cpp_worker_entered);
    m.def("cpp_worker_done", &cpp_worker_done);
    m.def("join_cpp_worker", &join_cpp_worker);
    m.def("event_end_ready", &event_end_ready);
    m.def("cleanup", &cleanup);
}

```
