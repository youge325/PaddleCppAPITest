# Paddle compat 层兼容方式架构图

本文档说明 Paddle compat 层如何将 PyTorch 的 `c10::Storage` / `c10::DataPtr` 接口映射到 Paddle 内部实现。

---

## 整体分层架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                   PyTorch LibTorch 兼容 API 层                       │
│  c10::Storage  │  c10::DataPtr  │  c10::Allocator  │  at::cuda::*   │
└─────────────────────────────────────────────────────────────────────┘
                              │ compat shim
┌─────────────────────────────────────────────────────────────────────┐
│                   Paddle compat 实现层                               │
│  Storage.h     │  Allocator.h  │  CUDAContextLight.h/.cpp           │
│  (两条内部路径) │  (Device桥接) │  (phi::GPUContext 委托)            │
└─────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                   Paddle 原生实现层（phi）                           │
│  phi::Allocation │ phi::Allocator │ phi::Place │ phi::GPUContext     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## c10::Storage 内部两条路径

PyTorch 的 `StorageImpl` 使用单一 `DataPtr data_ptr_` 成员。Paddle compat 的 `Storage` 为了同时兼容 Paddle 原生内存管理和 PyTorch 外部 DataPtr 语义，采用了两条路径：

```
c10::Storage
  ├── [路径 A：allocation-backed]
  │     shared_ptr<phi::Allocation> allocation_   ← 主所有者
  │     shared_ptr<DataPtr>         data_ptr_     ← 非拥有性视图（含 device 信息）
  │     shared_ptr<void>            external_ctx_ ← nullptr
  │
  │     device_type() / device()  → 从 allocation_->place() 读取
  │     use_count() / unique()    → allocation_.use_count()
  │     data_ptr() / mutable_data_ptr() → *data_ptr_（引用，含 raw ptr + device）
  │
  └── [路径 B：external DataPtr]
        shared_ptr<phi::Allocation> allocation_   ← nullptr
        shared_ptr<DataPtr>         data_ptr_     ← 包装外部指针
        shared_ptr<void>            external_ctx_ ← 真正的上下文所有者（含 deleter）
              （仅当原始 DataPtr 有 deleter 时存在）

        device_type() / device()  → 从 data_ptr_->device() 读取（保留完整 device index）
        use_count() / unique()    → external_ctx_.use_count()（有 deleter 时）
                                    data_ptr_.use_count()（无 deleter 但有效指针时）
                                    0（默认构造空 Storage）
        data_ptr() / mutable_data_ptr() → *data_ptr_（引用）
```

### 路径对比：PyTorch vs Paddle compat

| 属性                | PyTorch StorageImpl              | Paddle compat Storage（路径 A）  | Paddle compat Storage（路径 B）  |
|---------------------|----------------------------------|----------------------------------|----------------------------------|
| 数据所有权          | `DataPtr data_ptr_`（唯一来源）  | `shared_ptr<phi::Allocation>`    | `shared_ptr<void> external_ctx_` |
| 设备信息来源        | `data_ptr_.device()`             | `allocation_->place()`           | `data_ptr_->device()`            |
| 引用计数来源        | `intrusive_ptr<StorageImpl>`     | `allocation_.use_count()`        | `external_ctx_.use_count()`      |
| copy-on-write       | 无（单一 StorageImpl）           | 共享 `allocation_`               | `ensureUniqueDataPtr()` + CoW    |

---

## c10::DataPtr 与 phi::Place 的映射

```
c10::DataPtr
  ├── c10::detail::UniqueVoidPtr ptr_   ← 数据指针 + 上下文/deleter
  └── phi::Place device_                ← 设备信息（通过 c10::Device::_PD_GetInner() 转换）

c10::Device(phi::Place) ──→ Device::inner_ = phi::Place
c10::Device::index()    ──→ phi::Place::GetDeviceId()   ← 保留完整 device index
c10::Device::type()     ──→ phi::Place::GetType()       ← CPU / GPU / XPU 等
```

---

## at::cuda 接口映射（CUDAContextLight）

```
at::cuda::getCurrentCUDABlasHandle()
  └── at::cuda::getCurrentGPUContext()
        └── phi::DeviceContextPool::Instance().Get(phi::GPUPlace(device_id))
              └── static_cast<phi::GPUContext*>(ctx)->cublas_handle()

at::cuda::is_available()
  └── c10::cuda::device_count() > 0
        └── phi::backends::gpu::GetGPUDeviceCount()
              （非 GPU 构建时返回 0，不抛异常——能力探测语义）
```

---

## 注意事项

1. **路径 B 的 CoW 机制**：调用 `mutable_data_ptr()` 时触发 `ensureUniqueDataPtr()`，将 `data_ptr_` 替换为仅包含当前副本的新 `shared_ptr<DataPtr>`，但 `external_ctx_` 仍被所有副本共享，因此 `use_count()` 依赖 `external_ctx_` 而非 `data_ptr_`。

2. **默认构造 Storage**：两条路径均为空，`*data_ptr_` 为 falsy，`use_count()` 返回 0，与 PyTorch 空 `intrusive_ptr<StorageImpl>` 的语义一致。

3. **多卡 device index 保留**：`phi::GPUPlace(n)` 的 device id 为 `n`，通过 `phi::Place::GetDeviceId()` 可完整读回，因此 `DataPtr::device().index()` 在多卡场景下返回正确值。
