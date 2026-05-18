> 2026-05-16 编制/复核：Claude Code

## 对比文件列表

- PyTorch C++ API 头文件: `/home/may/libtorch/include/ATen/ops/index_add.h`
- Paddle compat 层头文件: `/home/may/Paddle/paddle/phi/api/include/compat/ATen/ops/index_add.h`

## 状态说明

- `✅`：接口与语义一致
- `🔧`：接口在，但实现路径或边界行为不同
- `❌`：Torch 有、Paddle 缺失

## API 对比表

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `at::index_add` | `✅` | - [x] | P0 | compat 层已实现自由函数封装 |
| `at::Tensor::index_add` | `✅` | - [x] | P0 | 成员方法已覆盖 |
| `at::Tensor::index_add_` | `✅` | - [x] | P0 | in-place 成员方法已覆盖 |

## 兼容性统计

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 3 |
| 🔧 部分兼容 | 0 |
| ❌ 未实现 | 0 |

## 关键差异说明

无关键差异。Paddle C++ API `index_add` 无 `alpha` 参数，compat 层通过在调用 `paddle::experimental::index_add` 前使用 `paddle::experimental::scale` 对 `source` 进行缩放处理，实现了与 PyTorch `index_add(..., alpha)` 完全一致的行为。

## 备注

- 实现文件：`/home/may/Paddle/paddle/phi/api/include/compat/ATen/ops/index_add.h`
- Paddle 侧测试：`/home/may/Paddle/test/cpp/compat/ATen_index_add_test.cc`
- PCAT 跨框架测试：`/home/may/PaddleCppAPITest/test/ATen/ops/IndexAddTest.cpp`
