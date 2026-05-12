#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
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
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(err));
  }
}

// 使用 volatile 确保 GPU kernel 能看到 host 对 mapped memory 的写入
__global__ void wait_for_host_flag(volatile int* flag) {
  while (*flag == 0) {
#if __CUDA_ARCH__ >= 700
    __nanosleep(1000);
#endif
  }
}

uintptr_t current_stream_raw(int device) {
  auto stream = at::cuda::getCurrentCUDAStream(device);
  return reinterpret_cast<uintptr_t>(stream.stream());
}

void set_pool_stream(int device) {
  auto pool_stream = c10::cuda::getStreamFromPool(false, device);
  c10::cuda::setCurrentCUDAStream(pool_stream);
  std::cerr << "[ext] set_pool_stream device=" << device
            << " pool_stream=" << reinterpret_cast<void*>(pool_stream.stream())
            << std::endl;
}

void start_cycle(int device) {
  check(cudaSetDevice(device), "cudaSetDevice");
  if (g_state.enq_stream != nullptr) {
    throw std::runtime_error("start_cycle called twice without cleanup");
  }

  auto current = at::cuda::getCurrentCUDAStream(device);
  auto current_raw = current.stream();

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

void block_current_stream_on_event_end(int device, bool cpu_sync) {
  check(cudaSetDevice(device), "cudaSetDevice");
  if (g_state.event_end == nullptr) {
    throw std::runtime_error(
        "block_current_stream_on_event_end before start_cycle");
  }

  if (cpu_sync) {
    std::cerr << "[ext] block_current_stream_on_event_end device=" << device
              << " CPU-sync waiting for event_end" << std::endl;
    check(cudaEventSynchronize(g_state.event_end),
          "cudaEventSynchronize(event_end)");
    std::cerr << "[ext] CPU-sync done" << std::endl;
  } else {
    auto current = at::cuda::getCurrentCUDAStream(device);
    auto current_raw = current.stream();
    check(cudaStreamWaitEvent(current_raw, g_state.event_end, 0),
          "cudaStreamWaitEvent(current,event_end)");

    std::cerr << "[ext] block_current_stream_on_event_end device=" << device
              << " current_stream=" << reinterpret_cast<void*>(current_raw)
              << " waits event_end" << std::endl;
  }
}

void set_host_flag() {
  if (g_state.host_flag == nullptr) {
    throw std::runtime_error("set_host_flag before start_cycle");
  }
  *g_state.host_flag = 1;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  // 触发一次非阻塞的 CUDA 操作来帮助 flush mapped memory 到 GPU
  (void)cudaStreamQuery(0);
  std::cerr << "[ext] host_flag set" << std::endl;
}

void start_cpp_tensor_set_worker(int device, bool set_flag_after_assignment) {
  if (g_cpp_worker.joinable()) {
    throw std::runtime_error("cpp worker already running");
  }
  g_cpp_worker_done.store(false, std::memory_order_release);
  g_cpp_worker_entered.store(false, std::memory_order_release);

  g_cpp_worker = std::thread([device, set_flag_after_assignment]() {
    check(cudaSetDevice(device), "cpp worker cudaSetDevice");
    auto current = at::cuda::getCurrentCUDAStream(device);
    auto current_raw = current.stream();
    std::cerr << "[cpp worker] current_stream="
              << reinterpret_cast<void*>(current_raw) << std::endl;
    // 第二种解法：worker 使用独立 pool stream + 裸 CUDA 操作
    auto worker_stream = c10::cuda::getStreamFromPool(false, device);
    c10::cuda::setCurrentCUDAStream(worker_stream);
    std::cerr << "[cpp worker] worker_stream="
              << reinterpret_cast<void*>(worker_stream.stream()) << std::endl;
    std::cerr << "[cpp worker] before at_tensor[0] = 0" << std::endl;
    g_cpp_worker_entered.store(true, std::memory_order_release);
    float* d_data = nullptr;
    check(cudaMalloc(&d_data, 4 * sizeof(float)), "cudaMalloc");
    std::cerr << "[cpp worker] cudaMalloc done" << std::endl;
    check(cudaMemsetAsync(d_data, 0, 4 * sizeof(float), worker_stream.stream()),
          "cudaMemsetAsync");
    std::cerr << "[cpp worker] cudaMemsetAsync done" << std::endl;
    check(cudaStreamSynchronize(worker_stream.stream()),
          "cudaStreamSynchronize");
    std::cerr << "[cpp worker] cudaStreamSynchronize done" << std::endl;
    // 注：cudaFree 会隐式同步设备上所有 stream，这里不调用以避免死锁
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
  m.def("set_pool_stream", &set_pool_stream);
  m.def("start_cycle", &start_cycle);
  m.def("block_current_stream_on_event_end",
        &block_current_stream_on_event_end,
        py::arg("device"),
        py::arg("cpu_sync") = false);
  m.def("set_host_flag", &set_host_flag);
  m.def("start_cpp_tensor_set_worker", &start_cpp_tensor_set_worker);
  m.def("cpp_worker_entered", &cpp_worker_entered);
  m.def("cpp_worker_done", &cpp_worker_done);
  m.def("join_cpp_worker", &join_cpp_worker);
  m.def("event_end_ready", &event_end_ready);
  m.def("cleanup", &cleanup);
}
