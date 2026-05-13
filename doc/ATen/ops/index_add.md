# Paddle compat 层 index_add 算子

本文档介绍 Paddle compat 层中 `index_add` / `index_add_` / `at::index_add` 三件套的兼容实现,以及与 PyTorch 行为的对齐情况。

> **Note**: 本文档参考 `/home/may/Paddle/paddle/phi/api/include/compat/ATen/ops/index_add.h` 以及测试代码 `/home/may/Paddle/test/cpp/compat/ATen_index_add_test.cc` 编写。

---

## 1. index_add 算子概述

`index_add` 在指定维度 `dim` 上,按照 `index` 给出的位置把 `source * alpha` 累加到 `self`,等价于:

```
self[..., index[i], ...] += alpha * source[..., i, ...]
```

### 1.1 核心 API

```cpp
// out-of-place free function
at::Tensor at::index_add(const at::Tensor& self, int64_t dim,
                         const at::Tensor& index, const at::Tensor& source,
                         const at::Scalar& alpha = 1);

// Tensor method: out-of-place
at::Tensor Tensor::index_add(int64_t dim, const at::Tensor& index,
                             const at::Tensor& source,
                             const at::Scalar& alpha = 1) const;

// Tensor method: in-place
at::Tensor& Tensor::index_add_(int64_t dim, const at::Tensor& index,
                               const at::Tensor& source,
                               const at::Scalar& alpha = 1) const;
```

### 1.2 涉及的关键概念

| 概念 | 说明 |
| --- | --- |
| **alpha 缩放** | `source * alpha` 后再累加;`alpha == 1` 时跳过 scale 内核 |
| **负数 dim** | 支持负数维度,由后端 phi kernel 自动 wrap(`paddle/phi/kernels/cpu/index_add_impl.h`) |
| **index 重复** | 同一目标位置出现多次时,按出现次数累加 |
| **index dtype** | 支持 `int64` 和 `int32`,与 PyTorch 一致 |

---

## 2. 源码解析

### 2.1 wrapper 整体结构

```cpp
// paddle/phi/api/include/compat/ATen/ops/index_add.h

inline paddle::Tensor _index_add_apply_alpha(const at::Tensor& source,
                                             const at::Scalar& alpha) {
  if (alpha.to<double>() == 1.0) {
    return source._PD_GetInner();
  }
  return paddle::experimental::scale(source._PD_GetInner(),
                                     phi::Scalar(alpha.to<double>()),
                                     /*bias=*/0.0f,
                                     /*bias_after_scale=*/true);
}

inline at::Tensor index_add(const at::Tensor& self, int64_t dim,
                            const at::Tensor& index, const at::Tensor& source,
                            const at::Scalar& alpha = 1) {
  auto add_value = _index_add_apply_alpha(source, alpha);
  return paddle::experimental::index_add(self._PD_GetInner(),
                                         index._PD_GetInner(),
                                         add_value,
                                         static_cast<int>(dim));
}
```

**实现要点**:

1. **alpha == 1 fast path**:直接复用 `source._PD_GetInner()`,避免一次冗余 scale kernel。
2. **alpha != 1**:用 `paddle::experimental::scale` 把 source 缩放为 `source * alpha`,再交给 `paddle::experimental::index_add`。
3. **dim 直接透传**:负数 dim 由 phi kernel 自行 wrap,wrapper 无需额外处理。
4. **in-place 复用同一缩放路径**:`index_add_` 通过 `const_cast<at::Tensor&>(*this)._PD_GetInner()` 拿到非 const 引用后调用 `paddle::experimental::index_add_`,语义与 PyTorch 完全一致。

### 2.2 与 PyTorch 的行为差异处理

PyTorch 的 `index_add` 在 Python 层文档声称"integral input tensors 需要 integral alpha",但 libtorch CPU 端**实际并不强制抛出**(只是文档约定)。本 wrapper 选择**与 libtorch 实际行为一致**:不在 wrapper 层做 alpha-vs-self-dtype 校验,让 phi 后端按照 scale 后的结果累加。

---

## 3. API 对比表

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `at::index_add(self, dim, index, source, alpha)` | ✅ | - [x] | P0 | 与 libtorch CPU 行为一致 |
| `Tensor::index_add(dim, index, source, alpha) const` | ✅ | - [x] | P0 | 内部直接调用上面的 free function |
| `Tensor::index_add_(dim, index, source, alpha) const` | ✅ | - [x] | P0 | in-place,返回 `Tensor&` |
| `at::index_add_out(out, self, dim, index, source, alpha)` | ❌ | - [ ] | P2 | 输出参数版本暂未实现 |
| `at::index_add_outf(self, dim, index, source, alpha, out)` | ❌ | - [ ] | P3 | functional out 版本暂未实现 |
| `Tensor::index_add(Dimname, ...)` | ❌ | - [ ] | P3 | Named tensor 重载,Paddle 未支持 Dimname |

---

## 4. 兼容性统计

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 3 |
| 🔧 部分兼容 | 0 |
| ❌ 未实现 | 3 |

---

## 5. 关键差异说明

本轮新增接口未引入与 libtorch CPU 行为的可观察差异,所有 ✅ 条目在跨框架 `result_cmp.sh` 中均与 `torch_IndexAddTest` 严格一致。下列条目仅记录"刻意未实现"的语义点,供后续扩展时复核:

1. **整数 self + 浮点 alpha**:libtorch CPU 不抛异常(在 wrapper 早期实现中错误地按文档主动 throw,后续依据 `result_cmp` 差异更正)。当前 Paddle compat 通过 `paddle::experimental::scale` 把 source 先升到 double 再 cast 回原 dtype,与 libtorch CPU 结果一致。
2. **out 重载**:`at::index_add_out` / `at::index_add_outf` 暂未实现,需要用户预先分配输出张量的场景目前请改用 out-of-place 版本后再 `copy_` 到目标。
3. **Dimname 重载**:Paddle 尚未提供 Dimname 体系,本轮不实现 named 维度的 `index_add` 重载。

---

## 6. 测试覆盖

### Paddle ctest

`/home/may/Paddle/test/cpp/compat/ATen_index_add_test.cc` 覆盖 10 个 case:

- `FreeFunctionDefaultAlpha`、`MethodOutOfPlaceDoesNotMutateSelf`、`MethodInplaceMutatesSelf`
- `AlphaTwoScalesSource`、`AlphaNegativeSubtracts`
- `NegativeDimWrapsCorrectly`、`IndexInt32Accepted`、`RepeatedIndexAccumulates`
- `IntegerSelfFloatAlphaDoesNotThrow`(刻意验证与 libtorch CPU 行为一致)

### PCAT 跨框架对比

`/home/may/PaddleCppAPITest/test/ATen/ops/IndexAddTest.cpp` 覆盖 20 个 case,
按以下分组,所有 case 在 `bash test/result_cmp.sh ./build/` 中 MATCH:

- **三件套签名**:`FreeFunctionFloat` / `MethodOutOfPlaceFloat` / `MethodInplaceFloat`
- **dtype 覆盖**:`DtypeDouble` / `DtypeIntAlphaIntegral` / `DtypeLongAlphaIntegral`
- **alpha 变体**:`AlphaPositiveFloat` / `AlphaNegativeFloat` / `AlphaZero`
- **dim 变体**:`DimZero2D` / `DimLast2D` / `DimNegative`
- **shape 档位**:`ShapeMedium2D`
- **index 形态**:`IndexInt32Accepted` / `IndexSingleElement` / `IndexRepeatsAccumulate`
- **异常 / 边界**:`IndexDtypeFloatThrows` / `IndexOutOfBoundsThrows` / `ShapeMismatchThrows` / `IntegerSelfFloatAlphaNoThrow`

---

## 7. 备注

- 实现完全位于头文件中(`inline` 函数),不需要单独 `.cpp`。
- 依赖:`paddle::experimental::index_add` / `index_add_`(`paddle/phi/api/include/api.h:612-614`)、`paddle::experimental::scale`(用于 alpha != 1 路径)、`Tensor::_PD_GetInner()`。
- 不依赖 `phi/common/data_type.h`(此前为校验 alpha dtype 引入过,已随主动校验逻辑一并删除)。

---

## 8. 参考代码路径

| 文件 | 说明 |
| --- | --- |
| `/home/may/Paddle/paddle/phi/api/include/compat/ATen/ops/index_add.h` | wrapper 实现 |
| `/home/may/Paddle/paddle/phi/api/include/compat/ATen/core/TensorBody.h` | method 声明 |
| `/home/may/Paddle/paddle/phi/api/include/compat/ATen/Functions.h` | 头文件聚合入口 |
| `/home/may/Paddle/test/cpp/compat/ATen_index_add_test.cc` | Paddle ctest |
| `/home/may/PaddleCppAPITest/test/ATen/ops/IndexAddTest.cpp` | PCAT 跨框架对比测试 |

---

## 对齐迭代记录(2026-05-12)

### 1) 接口变更
- 接口名:`at::index_add` / `Tensor::index_add` / `Tensor::index_add_`
- 变更类型:新增
- Paddle 兼容层位置:`paddle/phi/api/include/compat/ATen/ops/index_add.h`
- 参考 PyTorch 位置:`aten/src/ATen/native/TensorAdvancedIndexing.cpp`、libtorch headers `<ATen/ops/index_add.h>`

### 2) 测试覆盖
- 测试文件:
  - `Paddle/test/cpp/compat/ATen_index_add_test.cc`(10 cases,Paddle ctest)
  - `PaddleCppAPITest/test/ATen/ops/IndexAddTest.cpp`(20 cases,跨框架 result_cmp)
- 新增/修改用例:全部为新增
- 覆盖点:
  - shape:`{3}`、`{4}`、`{5}`、`{3,4}`、`{2,3,4}`、`{8,16}` 等
  - dtype:kFloat / kDouble / kInt / kLong
  - alpha:1.0(默认)、2.0、2.5、-1.0、0.0、整数 alpha
  - dim:0 / 1 / -1
  - 异常:错误 index dtype、index 越界、shape 不匹配、integer self + float alpha(不应抛)

### 3) 新增接口验证结果
- 新增前状态:Paddle compat 层完全缺失 `index_add`(全树 grep 0 命中)。
- 新增后验证结果:
  - Paddle ctest `ATen_index_add_test` PASSED(0.24s,1/1)。
  - Paddle 全量 `ctest -R "ATen|c10|torch"` 100% 通过(69/69)。
  - PCAT `result_cmp.sh` 中 `paddle_IndexAddTest` 与 `torch_IndexAddTest` MATCH。
- 关键行为说明:
  - 一次迭代修正:初版 wrapper 主动校验"整数 self + 浮点 alpha"并 throw,与 libtorch CPU 实际行为不一致(libtorch 不抛),`result_cmp` 报 DIFFER。第二轮删除该校验,两侧行为对齐,result_cmp MATCH。

### 4) 构建与回归结果
- Paddle 编译:通过(`cd /home/may/Paddle/build && ninja -j16` 成功生成 wheel)
- ctest (ATen|c10|torch):通过(69/69,24.68s)
- result_cmp:`paddle_IndexAddTest` MATCH `torch_IndexAddTest`,其余 DIFFER 项均为预存量,与本次新增无关

### 5) 未完成项与下一轮计划
- 未完成接口:
  - `at::index_add_out(out, self, dim, index, source, alpha)`(P2)
  - `at::index_add_outf(self, dim, index, source, alpha, out)`(P3)
  - `Tensor::index_add(Dimname, ...)`(P3,需 Dimname 支持)
- 下一轮优先级:
  - 若用户明确需要 out 参数版本,补 `index_add_out` / `index_add_outf` 两个重载(本质是 wrapper 末尾追加一次 `result.copy_to(out)`)。
  - Dimname 重载随整个 Dimname 体系一并规划。
