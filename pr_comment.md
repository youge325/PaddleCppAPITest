## [Cpp API Compatibility] 修复 getCurrentCUDAStream thread-local 语义

### 问题

PR #78652 删除了原有的 `thread_local` current stream 状态，改为直接从 Paddle 全局 `GPUContext` 读取 stream。这导致 `getCurrentCUDAStream()` 不再是 thread-local 的——新线程会继承主线程的 current stream，违反 PyTorch 语义并可能引发死锁。

此外，PR #78652 中 `setCurrentCUDAStream` 通过 `getMutableGPUContext()->SetStream()` 同步 GPUContext stream，但 `SetStream` 内部会调用 `cudaStreamDestroy` 销毁旧 stream。当旧 stream 来自 compat pool（如 `getStreamFromPool` 返回的 stream）时，外部并未移交所有权，错误的 destroy 会导致 stream handle 失效，后续重复使用即触发 SegFault。

**Windows 流水线修复：** 测试代码中使用 `std::packaged_task<c10::cuda::CUDAStream()>()` 在 MSVC 上编译失败（error C2512），因为 MSVC 的 `std::future` 实现需要 `CUDAStream` 的默认构造函数。改为 `std::thread` + 引用捕获传递结果。

### 修复

恢复 thread-local 语义，同时保留 FastDeploy #7344 的修复，并解决 stream destroy 问题：

1. **Thread-local current stream**
   - 重新引入 `thread_local std::vector<cudaStream_t> g_thread_local_current_streams`（含 `#ifdef PADDLE_WITH_HIP` 分支）
   - `getCurrentCUDAStream`：优先从 thread-local 读取，未设置时返回 default stream
   - `setCurrentCUDAStream`：将 stream 存入 thread-local 状态

2. **GPUContext 同步（避免 destroy 外部 stream）**
   - 将 `getMutableGPUContext()->SetStream()` 替换为 `SetCUDAStream()`
   - 创建 `phi::CUDAStream` 对象并标记 `owned_=false`，告知 GPUContext 不要 destroy 该 stream handle
   - 首次调用时 GPUContext 会正确释放自己之前创建的 stream，后续切换不再重复 destroy

### 新增测试

- `GetCurrentCUDAStreamIsThreadLocal`：主线程设置 pool stream 后，新线程 `getCurrentCUDAStream()` 应返回 default stream（id == 0）
- `CurrentStreamDeadlockReproducer`：模拟 pool_stream 被 event 阻塞场景，后台线程 sync 该 stream，验证 `wait_for(50ms)` 不会 timeout
- `GetCurrentCUDAStreamStableInUnsetThread`：主线程反复切换 current stream（修改 GPUContext），后台线程（从不调用 `setCurrentCUDAStream`）持续采样，验证每次返回的 default stream 稳定且相等

### 验证

- `ninja -j16`：通过
- `ctest -R "ATen|c10|torch"`：68/68 通过
- `result_cmp.sh`：通过
