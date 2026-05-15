# PyTorch C++ API (libtorch) 与 Paddle C++ API 映射表

本文梳理了 PyTorch C++ API (libtorch) 与 PaddlePaddle C++ API 的对应关系与差异分析，
帮助开发者快速迁移 PyTorch C++ 使用经验。

> **Note**: 本映射表基于以下路径**自动解析 C++ 函数签名**生成：
> - PyTorch C++ API 头文件: `D:/迅雷下载/libtorch/include/ATen/ops`
> - Paddle compat 层头文件: `D:/Lenovo/Paddle/paddle/phi/api/include/compat/ATen/ops`
> - Paddle `api.h` 头文件: `D:/Lenovo/Paddle/paddle/phi/api/include/api.h`

> **说明**: 对于 compat 层未封装的函数，脚本直接对比 libtorch 头文件与 `paddle::experimental` 命名空间中同名函数的**返回类型、参数类型、参数名、参数默认值、参数数量**，按优先级自动归入差异分类。

## API 映射分类

| 序号 | 类别 | 简介 |
| ---- | ---- | ---- |
| 1 | API 完全一致 | compat 层已实现与 PyTorch C++ API 完全一致的接口，可直接替换命名空间使用 |
| 2 | 仅 API 调用方式不一致 | Paddle 有同名实现，但调用方式与 PyTorch 不一致（签名解析后兜底分类） |
| 3 | 仅参数名不一致 | 功能相同，但部分参数名称不同 |
| 4 | paddle 参数更多 | Paddle 中提供了更多可选参数 |
| 5 | 参数默认值不一致 | 功能相同，但某些参数的默认值不同 |
| 6 | torch 参数更多 | PyTorch 中提供了更多参数 |
| 7 | 输入参数用法不一致 | 对输入参数的处理方式不同 |
| 8 | 输入参数类型不一致 | 要求的输入数据类型不同 |
| 9 | 返回参数类型不一致 | 返回值的类型或结构不同 |
| 10 | 组合替代实现 | 在 Paddle 中没有直接对应的单一 API，需要多个 API 组合实现 |
| 11 | API 别名 | PyTorch 与 Paddle 功能一致，但 API 名称不同 |
| 12 | 功能缺失 | PyTorch C++ API 的功能在 Paddle 中暂时没有等效实现 |

### 1. API 完全一致

**简介：** compat 层已实现与 PyTorch C++ API 完全一致的接口，只需将代码中的命名空间或调用方式按 compat 层声明使用即可。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::_local_scalar_dense` | `at::_local_scalar_dense` (compat层) | API 完全一致 | 头文件: `ATen/ops/_local_scalar_dense.h` |
| 2 | `at::_nnz` | `at::_nnz` (compat层) | API 完全一致 | 头文件: `ATen/ops/_nnz.h` |
| 3 | `at::_values` | `at::_values` (compat层) | API 完全一致 | 头文件: `ATen/ops/_values.h` |
| 4 | `at::abs` | `at::abs` (compat层) | API 完全一致 | 头文件: `ATen/ops/abs.h` |
| 5 | `at::all` | `at::all` (compat层) | API 完全一致 | 头文件: `ATen/ops/all.h` |
| 6 | `at::allclose` | `at::allclose` (compat层) | API 完全一致 | 头文件: `ATen/ops/allclose.h` |
| 7 | `at::any` | `at::any` (compat层) | API 完全一致 | 头文件: `ATen/ops/any.h` |
| 8 | `at::arange` | `at::arange` (compat层) | API 完全一致 | 头文件: `ATen/ops/arange.h` |
| 9 | `at::as_strided` | `at::as_strided` (compat层) | API 完全一致 | 头文件: `ATen/ops/as_strided.h` |
| 10 | `at::cat` | `at::cat` (compat层) | API 完全一致 | 头文件: `ATen/ops/cat.h` |
| 11 | `at::chunk` | `at::chunk` (compat层) | API 完全一致 | 头文件: `ATen/ops/chunk.h` |
| 12 | `at::clamp` | `at::clamp` (compat层) | API 完全一致 | 头文件: `ATen/ops/clamp.h` |
| 13 | `at::coalesce` | `at::coalesce` (compat层) | API 完全一致 | 头文件: `ATen/ops/coalesce.h` |
| 14 | `at::detach` | `at::detach` (compat层) | API 完全一致 | 头文件: `ATen/ops/detach.h` |
| 15 | `at::dsplit` | `at::dsplit` (compat层) | API 完全一致 | 头文件: `ATen/ops/dsplit.h` |
| 16 | `at::empty` | `at::empty` (compat层) | API 完全一致 | 头文件: `ATen/ops/empty.h` |
| 17 | `at::empty_like` | `at::empty_like` (compat层) | API 完全一致 | 头文件: `ATen/ops/empty_like.h` |
| 18 | `at::empty_strided` | `at::empty_strided` (compat层) | API 完全一致 | 头文件: `ATen/ops/empty_strided.h` |
| 19 | `at::equal` | `at::equal` (compat层) | API 完全一致 | 头文件: `ATen/ops/equal.h` |
| 20 | `at::expand` | `at::expand` (compat层) | API 完全一致 | 头文件: `ATen/ops/expand.h` |
| 21 | `at::eye` | `at::eye` (compat层) | API 完全一致 | 头文件: `ATen/ops/eye.h` |
| 22 | `at::flatten` | `at::flatten` (compat层) | API 完全一致 | 头文件: `ATen/ops/flatten.h` |
| 23 | `at::from_blob` | `at::from_blob` (compat层) | API 完全一致 | 头文件: `ATen/ops/from_blob.h` |
| 24 | `at::full` | `at::full` (compat层) | API 完全一致 | 头文件: `ATen/ops/full.h` |
| 25 | `at::hsplit` | `at::hsplit` (compat层) | API 完全一致 | 头文件: `ATen/ops/hsplit.h` |
| 26 | `at::index` | `at::index` (compat层) | API 完全一致 | 头文件: `ATen/ops/index.h` |
| 27 | `at::index_put` | `at::index_put` (compat层) | API 完全一致 | 头文件: `ATen/ops/index_put.h` |
| 28 | `at::is_coalesced` | `at::is_coalesced` (compat层) | API 完全一致 | 头文件: `ATen/ops/is_coalesced.h` |
| 29 | `at::item` | `at::item` (compat层) | API 完全一致 | 头文件: `ATen/ops/item.h` |
| 30 | `at::masked_select` | `at::masked_select` (compat层) | API 完全一致 | 头文件: `ATen/ops/masked_select.h` |
| 31 | `at::narrow` | `at::narrow` (compat层) | API 完全一致 | 头文件: `ATen/ops/narrow.h` |
| 32 | `at::narrow_copy` | `at::narrow_copy` (compat层) | API 完全一致 | 头文件: `ATen/ops/narrow_copy.h` |
| 33 | `at::new_empty` | `at::new_empty` (compat层) | API 完全一致 | 头文件: `ATen/ops/new_empty.h` |
| 34 | `at::new_full` | `at::new_full` (compat层) | API 完全一致 | 头文件: `ATen/ops/new_full.h` |
| 35 | `at::new_ones` | `at::new_ones` (compat层) | API 完全一致 | 头文件: `ATen/ops/new_ones.h` |
| 36 | `at::new_zeros` | `at::new_zeros` (compat层) | API 完全一致 | 头文件: `ATen/ops/new_zeros.h` |
| 37 | `at::ones` | `at::ones` (compat层) | API 完全一致 | 头文件: `ATen/ops/ones.h` |
| 38 | `at::permute` | `at::permute` (compat层) | API 完全一致 | 头文件: `ATen/ops/permute.h` |
| 39 | `at::reciprocal` | `at::reciprocal` (compat层) | API 完全一致 | 头文件: `ATen/ops/reciprocal.h` |
| 40 | `at::record_stream` | `at::record_stream` (compat层) | API 完全一致 | 头文件: `ATen/ops/record_stream.h` |
| 41 | `at::rename` | `at::rename` (compat层) | API 完全一致 | 头文件: `ATen/ops/rename.h` |
| 42 | `at::reshape` | `at::reshape` (compat层) | API 完全一致 | 头文件: `ATen/ops/reshape.h` |
| 43 | `at::resize` | `at::resize` (compat层) | API 完全一致 | 头文件: `ATen/ops/resize.h` |
| 44 | `at::select` | `at::select` (compat层) | API 完全一致 | 头文件: `ATen/ops/select.h` |
| 45 | `at::slice` | `at::slice` (compat层) | API 完全一致 | 头文件: `ATen/ops/slice.h` |
| 46 | `at::sparse_coo_tensor` | `at::sparse_coo_tensor` (compat层) | API 完全一致 | 头文件: `ATen/ops/sparse_coo_tensor.h` |
| 47 | `at::sparse_csr_tensor` | `at::sparse_csr_tensor` (compat层) | API 完全一致 | 头文件: `ATen/ops/sparse_csr_tensor.h` |
| 48 | `at::split` | `at::split` (compat层) | API 完全一致 | 头文件: `ATen/ops/split.h` |
| 49 | `at::split_with_sizes` | `at::split_with_sizes` (compat层) | API 完全一致 | 头文件: `ATen/ops/split_with_sizes.h` |
| 50 | `at::squeeze` | `at::squeeze` (compat层) | API 完全一致 | 头文件: `ATen/ops/squeeze.h` |
| 51 | `at::std` | `at::std` (compat层) | API 完全一致 | 头文件: `ATen/ops/std.h` |
| 52 | `at::sum` | `at::sum` (compat层) | API 完全一致 | 头文件: `ATen/ops/sum.h` |
| 53 | `at::t` | `at::t` (compat层) | API 完全一致 | 头文件: `ATen/ops/t.h` |
| 54 | `at::tensor` | `at::tensor` (compat层) | API 完全一致 | 头文件: `ATen/ops/tensor.h` |
| 55 | `at::tensor_split` | `at::tensor_split` (compat层) | API 完全一致 | 头文件: `ATen/ops/tensor_split.h` |
| 56 | `at::to` | `at::to` (compat层) | API 完全一致 | 头文件: `ATen/ops/to.h` |
| 57 | `at::transpose` | `at::transpose` (compat层) | API 完全一致 | 头文件: `ATen/ops/transpose.h` |
| 58 | `at::unflatten` | `at::unflatten` (compat层) | API 完全一致 | 头文件: `ATen/ops/unflatten.h` |
| 59 | `at::unsafe_split` | `at::unsafe_split` (compat层) | API 完全一致 | 头文件: `ATen/ops/unsafe_split.h` |
| 60 | `at::unsafe_split_with_sizes` | `at::unsafe_split_with_sizes` (compat层) | API 完全一致 | 头文件: `ATen/ops/unsafe_split_with_sizes.h` |
| 61 | `at::unsqueeze` | `at::unsqueeze` (compat层) | API 完全一致 | 头文件: `ATen/ops/unsqueeze.h` |
| 62 | `at::view` | `at::view` (compat层) | API 完全一致 | 头文件: `ATen/ops/view.h` |
| 63 | `at::view_as` | `at::view_as` (compat层) | API 完全一致 | 头文件: `ATen/ops/view_as.h` |
| 64 | `at::vsplit` | `at::vsplit` (compat层) | API 完全一致 | 头文件: `ATen/ops/vsplit.h` |
| 65 | `at::zeros` | `at::zeros` (compat层) | API 完全一致 | 头文件: `ATen/ops/zeros.h` |
| 66 | `at::zeros_like` | `at::zeros_like` (compat层) | API 完全一致 | 头文件: `ATen/ops/zeros_like.h` |

### 2. 仅 API 调用方式不一致

**简介：** Paddle `paddle::experimental` 命名空间中有同名实现，但 compat 层尚未提供完全一致的封装。以下函数经签名对比后归入此类，多为调用语义或底层实现差异。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::complex` | `paddle::experimental::complex` | 仅 API 调用方式不一致 | 头文件: `ATen/ops/complex.h`<br>签名高度相似，调用方式或语义有细微差异 |

### 3. 仅参数名不一致

**简介：** 此类 API 功能相同，但部分参数名称不同。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::acos` | `paddle::experimental::acos` | 仅参数名不一致 | 头文件: `ATen/ops/acos.h`<br>参数类型和默认值相同，仅参数名不同 |
| 2 | `at::acosh` | `paddle::experimental::acosh` | 仅参数名不一致 | 头文件: `ATen/ops/acosh.h`<br>参数类型和默认值相同，仅参数名不同 |
| 3 | `at::amax` | `paddle::experimental::amax` | 仅参数名不一致 | 头文件: `ATen/ops/amax.h`<br>参数类型和默认值相同，仅参数名不同 |
| 4 | `at::amin` | `paddle::experimental::amin` | 仅参数名不一致 | 头文件: `ATen/ops/amin.h`<br>参数类型和默认值相同，仅参数名不同 |
| 5 | `at::angle` | `paddle::experimental::angle` | 仅参数名不一致 | 头文件: `ATen/ops/angle.h`<br>参数类型和默认值相同，仅参数名不同 |
| 6 | `at::asin` | `paddle::experimental::asin` | 仅参数名不一致 | 头文件: `ATen/ops/asin.h`<br>参数类型和默认值相同，仅参数名不同 |
| 7 | `at::asinh` | `paddle::experimental::asinh` | 仅参数名不一致 | 头文件: `ATen/ops/asinh.h`<br>参数类型和默认值相同，仅参数名不同 |
| 8 | `at::atan` | `paddle::experimental::atan` | 仅参数名不一致 | 头文件: `ATen/ops/atan.h`<br>参数类型和默认值相同，仅参数名不同 |
| 9 | `at::atan2` | `paddle::experimental::atan2` | 仅参数名不一致 | 头文件: `ATen/ops/atan2.h`<br>参数类型和默认值相同，仅参数名不同 |
| 10 | `at::atanh` | `paddle::experimental::atanh` | 仅参数名不一致 | 头文件: `ATen/ops/atanh.h`<br>参数类型和默认值相同，仅参数名不同 |
| 11 | `at::bitwise_not` | `paddle::experimental::bitwise_not` | 仅参数名不一致 | 头文件: `ATen/ops/bitwise_not.h`<br>参数类型和默认值相同，仅参数名不同 |
| 12 | `at::bmm` | `paddle::experimental::bmm` | 仅参数名不一致 | 头文件: `ATen/ops/bmm.h`<br>参数类型和默认值相同，仅参数名不同 |
| 13 | `at::ceil` | `paddle::experimental::ceil` | 仅参数名不一致 | 头文件: `ATen/ops/ceil.h`<br>参数类型和默认值相同，仅参数名不同 |
| 14 | `at::cholesky` | `paddle::experimental::cholesky` | 仅参数名不一致 | 头文件: `ATen/ops/cholesky.h`<br>参数类型和默认值相同，仅参数名不同 |
| 15 | `at::cholesky_solve` | `paddle::experimental::cholesky_solve` | 仅参数名不一致 | 头文件: `ATen/ops/cholesky_solve.h`<br>参数类型和默认值相同，仅参数名不同 |
| 16 | `at::conj` | `paddle::experimental::conj` | 仅参数名不一致 | 头文件: `ATen/ops/conj.h`<br>参数类型和默认值相同，仅参数名不同 |
| 17 | `at::copysign` | `paddle::experimental::copysign` | 仅参数名不一致 | 头文件: `ATen/ops/copysign.h`<br>参数类型和默认值相同，仅参数名不同 |
| 18 | `at::cos` | `paddle::experimental::cos` | 仅参数名不一致 | 头文件: `ATen/ops/cos.h`<br>参数类型和默认值相同，仅参数名不同 |
| 19 | `at::cosh` | `paddle::experimental::cosh` | 仅参数名不一致 | 头文件: `ATen/ops/cosh.h`<br>参数类型和默认值相同，仅参数名不同 |
| 20 | `at::det` | `paddle::experimental::det` | 仅参数名不一致 | 头文件: `ATen/ops/det.h`<br>参数类型和默认值相同，仅参数名不同 |
| 21 | `at::digamma` | `paddle::experimental::digamma` | 仅参数名不一致 | 头文件: `ATen/ops/digamma.h`<br>参数类型和默认值相同，仅参数名不同 |
| 22 | `at::divide` | `paddle::experimental::divide` | 仅参数名不一致 | 头文件: `ATen/ops/divide.h`<br>参数类型和默认值相同，仅参数名不同 |
| 23 | `at::dot` | `paddle::experimental::dot` | 仅参数名不一致 | 头文件: `ATen/ops/dot.h`<br>参数类型和默认值相同，仅参数名不同 |
| 24 | `at::erf` | `paddle::experimental::erf` | 仅参数名不一致 | 头文件: `ATen/ops/erf.h`<br>参数类型和默认值相同，仅参数名不同 |
| 25 | `at::erfinv` | `paddle::experimental::erfinv` | 仅参数名不一致 | 头文件: `ATen/ops/erfinv.h`<br>参数类型和默认值相同，仅参数名不同 |
| 26 | `at::exp` | `paddle::experimental::exp` | 仅参数名不一致 | 头文件: `ATen/ops/exp.h`<br>参数类型和默认值相同，仅参数名不同 |
| 27 | `at::expm1` | `paddle::experimental::expm1` | 仅参数名不一致 | 头文件: `ATen/ops/expm1.h`<br>参数类型和默认值相同，仅参数名不同 |
| 28 | `at::floor` | `paddle::experimental::floor` | 仅参数名不一致 | 头文件: `ATen/ops/floor.h`<br>参数类型和默认值相同，仅参数名不同 |
| 29 | `at::floor_divide` | `paddle::experimental::floor_divide` | 仅参数名不一致 | 头文件: `ATen/ops/floor_divide.h`<br>参数类型和默认值相同，仅参数名不同 |
| 30 | `at::fmax` | `paddle::experimental::fmax` | 仅参数名不一致 | 头文件: `ATen/ops/fmax.h`<br>参数类型和默认值相同，仅参数名不同 |
| 31 | `at::fmin` | `paddle::experimental::fmin` | 仅参数名不一致 | 头文件: `ATen/ops/fmin.h`<br>参数类型和默认值相同，仅参数名不同 |
| 32 | `at::hardswish` | `paddle::experimental::hardswish` | 仅参数名不一致 | 头文件: `ATen/ops/hardswish.h`<br>参数类型和默认值相同，仅参数名不同 |
| 33 | `at::heaviside` | `paddle::experimental::heaviside` | 仅参数名不一致 | 头文件: `ATen/ops/heaviside.h`<br>参数类型和默认值相同，仅参数名不同 |
| 34 | `at::i0` | `paddle::experimental::i0` | 仅参数名不一致 | 头文件: `ATen/ops/i0.h`<br>参数类型和默认值相同，仅参数名不同 |
| 35 | `at::imag` | `paddle::experimental::imag` | 仅参数名不一致 | 头文件: `ATen/ops/imag.h`<br>参数类型和默认值相同，仅参数名不同 |
| 36 | `at::inverse` | `paddle::experimental::inverse` | 仅参数名不一致 | 头文件: `ATen/ops/inverse.h`<br>参数类型和默认值相同，仅参数名不同 |
| 37 | `at::isfinite` | `paddle::experimental::isfinite` | 仅参数名不一致 | 头文件: `ATen/ops/isfinite.h`<br>参数类型和默认值相同，仅参数名不同 |
| 38 | `at::isinf` | `paddle::experimental::isinf` | 仅参数名不一致 | 头文件: `ATen/ops/isinf.h`<br>参数类型和默认值相同，仅参数名不同 |
| 39 | `at::isnan` | `paddle::experimental::isnan` | 仅参数名不一致 | 头文件: `ATen/ops/isnan.h`<br>参数类型和默认值相同，仅参数名不同 |
| 40 | `at::kron` | `paddle::experimental::kron` | 仅参数名不一致 | 头文件: `ATen/ops/kron.h`<br>参数类型和默认值相同，仅参数名不同 |
| 41 | `at::lgamma` | `paddle::experimental::lgamma` | 仅参数名不一致 | 头文件: `ATen/ops/lgamma.h`<br>参数类型和默认值相同，仅参数名不同 |
| 42 | `at::log` | `paddle::experimental::log` | 仅参数名不一致 | 头文件: `ATen/ops/log.h`<br>参数类型和默认值相同，仅参数名不同 |
| 43 | `at::log10` | `paddle::experimental::log10` | 仅参数名不一致 | 头文件: `ATen/ops/log10.h`<br>参数类型和默认值相同，仅参数名不同 |
| 44 | `at::log1p` | `paddle::experimental::log1p` | 仅参数名不一致 | 头文件: `ATen/ops/log1p.h`<br>参数类型和默认值相同，仅参数名不同 |
| 45 | `at::log2` | `paddle::experimental::log2` | 仅参数名不一致 | 头文件: `ATen/ops/log2.h`<br>参数类型和默认值相同，仅参数名不同 |
| 46 | `at::logical_and` | `paddle::experimental::logical_and` | 仅参数名不一致 | 头文件: `ATen/ops/logical_and.h`<br>参数类型和默认值相同，仅参数名不同 |
| 47 | `at::logical_not` | `paddle::experimental::logical_not` | 仅参数名不一致 | 头文件: `ATen/ops/logical_not.h`<br>参数类型和默认值相同，仅参数名不同 |
| 48 | `at::logical_or` | `paddle::experimental::logical_or` | 仅参数名不一致 | 头文件: `ATen/ops/logical_or.h`<br>参数类型和默认值相同，仅参数名不同 |
| 49 | `at::logical_xor` | `paddle::experimental::logical_xor` | 仅参数名不一致 | 头文件: `ATen/ops/logical_xor.h`<br>参数类型和默认值相同，仅参数名不同 |
| 50 | `at::masked_scatter` | `paddle::experimental::masked_scatter` | 仅参数名不一致 | 头文件: `ATen/ops/masked_scatter.h`<br>参数类型和默认值相同，仅参数名不同 |
| 51 | `at::maximum` | `paddle::experimental::maximum` | 仅参数名不一致 | 头文件: `ATen/ops/maximum.h`<br>参数类型和默认值相同，仅参数名不同 |
| 52 | `at::minimum` | `paddle::experimental::minimum` | 仅参数名不一致 | 头文件: `ATen/ops/minimum.h`<br>参数类型和默认值相同，仅参数名不同 |
| 53 | `at::multiply` | `paddle::experimental::multiply` | 仅参数名不一致 | 头文件: `ATen/ops/multiply.h`<br>参数类型和默认值相同，仅参数名不同 |
| 54 | `at::mv` | `paddle::experimental::mv` | 仅参数名不一致 | 头文件: `ATen/ops/mv.h`<br>参数类型和默认值相同，仅参数名不同 |
| 55 | `at::nextafter` | `paddle::experimental::nextafter` | 仅参数名不一致 | 头文件: `ATen/ops/nextafter.h`<br>参数类型和默认值相同，仅参数名不同 |
| 56 | `at::nonzero` | `paddle::experimental::nonzero` | 仅参数名不一致 | 头文件: `ATen/ops/nonzero.h`<br>参数类型和默认值相同，仅参数名不同 |
| 57 | `at::real` | `paddle::experimental::real` | 仅参数名不一致 | 头文件: `ATen/ops/real.h`<br>参数类型和默认值相同，仅参数名不同 |
| 58 | `at::relu` | `paddle::experimental::relu` | 仅参数名不一致 | 头文件: `ATen/ops/relu.h`<br>参数类型和默认值相同，仅参数名不同 |
| 59 | `at::relu6` | `paddle::experimental::relu6` | 仅参数名不一致 | 头文件: `ATen/ops/relu6.h`<br>参数类型和默认值相同，仅参数名不同 |
| 60 | `at::rsqrt` | `paddle::experimental::rsqrt` | 仅参数名不一致 | 头文件: `ATen/ops/rsqrt.h`<br>参数类型和默认值相同，仅参数名不同 |
| 61 | `at::sigmoid` | `paddle::experimental::sigmoid` | 仅参数名不一致 | 头文件: `ATen/ops/sigmoid.h`<br>参数类型和默认值相同，仅参数名不同 |
| 62 | `at::sign` | `paddle::experimental::sign` | 仅参数名不一致 | 头文件: `ATen/ops/sign.h`<br>参数类型和默认值相同，仅参数名不同 |
| 63 | `at::silu` | `paddle::experimental::silu` | 仅参数名不一致 | 头文件: `ATen/ops/silu.h`<br>参数类型和默认值相同，仅参数名不同 |
| 64 | `at::sin` | `paddle::experimental::sin` | 仅参数名不一致 | 头文件: `ATen/ops/sin.h`<br>参数类型和默认值相同，仅参数名不同 |
| 65 | `at::sinh` | `paddle::experimental::sinh` | 仅参数名不一致 | 头文件: `ATen/ops/sinh.h`<br>参数类型和默认值相同，仅参数名不同 |
| 66 | `at::sqrt` | `paddle::experimental::sqrt` | 仅参数名不一致 | 头文件: `ATen/ops/sqrt.h`<br>参数类型和默认值相同，仅参数名不同 |
| 67 | `at::square` | `paddle::experimental::square` | 仅参数名不一致 | 头文件: `ATen/ops/square.h`<br>参数类型和默认值相同，仅参数名不同 |
| 68 | `at::tan` | `paddle::experimental::tan` | 仅参数名不一致 | 头文件: `ATen/ops/tan.h`<br>参数类型和默认值相同，仅参数名不同 |
| 69 | `at::tanh` | `paddle::experimental::tanh` | 仅参数名不一致 | 头文件: `ATen/ops/tanh.h`<br>参数类型和默认值相同，仅参数名不同 |
| 70 | `at::trunc` | `paddle::experimental::trunc` | 仅参数名不一致 | 头文件: `ATen/ops/trunc.h`<br>参数类型和默认值相同，仅参数名不同 |
| 71 | `at::_conj` | `paddle::experimental::_conj` | 仅参数名不一致 | 头文件: `ATen/ops/_conj.h`<br>参数类型和默认值相同，仅参数名不同 |
| 72 | `at::log_sigmoid` | `paddle::experimental::log_sigmoid` | 仅参数名不一致 | 头文件: `ATen/ops/log_sigmoid.h`<br>参数类型和默认值相同，仅参数名不同 |

### 4. paddle 参数更多

**简介：** 此类 API 在 Paddle 中提供了更多可选参数。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::argmax` | `paddle::experimental::argmax` | paddle 参数更多 | 头文件: `ATen/ops/argmax.h`<br>PyTorch 3 个参数，Paddle 5 个参数 |
| 2 | `at::argmin` | `paddle::experimental::argmin` | paddle 参数更多 | 头文件: `ATen/ops/argmin.h`<br>PyTorch 3 个参数，Paddle 5 个参数 |
| 3 | `at::baddbmm` | `paddle::experimental::baddbmm` | paddle 参数更多 | 头文件: `ATen/ops/baddbmm.h`<br>PyTorch 5 个参数，Paddle 6 个参数 |
| 4 | `at::bitwise_left_shift` | `paddle::experimental::bitwise_left_shift` | paddle 参数更多 | 头文件: `ATen/ops/bitwise_left_shift.h`<br>PyTorch 2 个参数，Paddle 3 个参数 |
| 5 | `at::bitwise_right_shift` | `paddle::experimental::bitwise_right_shift` | paddle 参数更多 | 头文件: `ATen/ops/bitwise_right_shift.h`<br>PyTorch 2 个参数，Paddle 3 个参数 |
| 6 | `at::channel_shuffle` | `paddle::experimental::channel_shuffle` | paddle 参数更多 | 头文件: `ATen/ops/channel_shuffle.h`<br>PyTorch 2 个参数，Paddle 3 个参数 |
| 7 | `at::conv2d` | `paddle::experimental::conv2d` | paddle 参数更多 | 头文件: `ATen/ops/conv2d.h`<br>PyTorch 7 个参数，Paddle 8 个参数 |
| 8 | `at::conv3d` | `paddle::experimental::conv3d` | paddle 参数更多 | 头文件: `ATen/ops/conv3d.h`<br>PyTorch 7 个参数，Paddle 8 个参数 |
| 9 | `at::cumprod` | `paddle::experimental::cumprod` | paddle 参数更多 | 头文件: `ATen/ops/cumprod.h`<br>PyTorch 3 个参数，Paddle 4 个参数 |
| 10 | `at::cumsum` | `paddle::experimental::cumsum` | paddle 参数更多 | 头文件: `ATen/ops/cumsum.h`<br>PyTorch 3 个参数，Paddle 5 个参数 |
| 11 | `at::diag` | `paddle::experimental::diag` | paddle 参数更多 | 头文件: `ATen/ops/diag.h`<br>PyTorch 2 个参数，Paddle 3 个参数 |
| 12 | `at::dropout` | `paddle::experimental::dropout` | paddle 参数更多 | 头文件: `ATen/ops/dropout.h`<br>PyTorch 3 个参数，Paddle 7 个参数 |
| 13 | `at::frobenius_norm` | `paddle::experimental::frobenius_norm` | paddle 参数更多 | 头文件: `ATen/ops/frobenius_norm.h`<br>PyTorch 3 个参数，Paddle 4 个参数 |
| 14 | `at::hardsigmoid` | `paddle::experimental::hardsigmoid` | paddle 参数更多 | 头文件: `ATen/ops/hardsigmoid.h`<br>PyTorch 1 个参数，Paddle 3 个参数 |
| 15 | `at::linspace` | `paddle::experimental::linspace` | paddle 参数更多 | 头文件: `ATen/ops/linspace.h`<br>PyTorch 4 个参数，Paddle 5 个参数 |
| 16 | `at::logcumsumexp` | `paddle::experimental::logcumsumexp` | paddle 参数更多 | 头文件: `ATen/ops/logcumsumexp.h`<br>PyTorch 2 个参数，Paddle 5 个参数 |
| 17 | `at::logspace` | `paddle::experimental::logspace` | paddle 参数更多 | 头文件: `ATen/ops/logspace.h`<br>PyTorch 5 个参数，Paddle 6 个参数 |
| 18 | `at::logsumexp` | `paddle::experimental::logsumexp` | paddle 参数更多 | 头文件: `ATen/ops/logsumexp.h`<br>PyTorch 3 个参数，Paddle 4 个参数 |
| 19 | `at::lu_solve` | `paddle::experimental::lu_solve` | paddle 参数更多 | 头文件: `ATen/ops/lu_solve.h`<br>PyTorch 3 个参数，Paddle 4 个参数 |
| 20 | `at::matmul` | `paddle::experimental::matmul` | paddle 参数更多 | 头文件: `ATen/ops/matmul.h`<br>PyTorch 2 个参数，Paddle 4 个参数 |
| 21 | `at::max` | `paddle::experimental::max` | paddle 参数更多 | 头文件: `ATen/ops/max.h`<br>PyTorch 1 个参数，Paddle 3 个参数 |
| 22 | `at::mean` | `paddle::experimental::mean` | paddle 参数更多 | 头文件: `ATen/ops/mean.h`<br>PyTorch 2 个参数，Paddle 3 个参数 |
| 23 | `at::min` | `paddle::experimental::min` | paddle 参数更多 | 头文件: `ATen/ops/min.h`<br>PyTorch 1 个参数，Paddle 3 个参数 |
| 24 | `at::mish` | `paddle::experimental::mish` | paddle 参数更多 | 头文件: `ATen/ops/mish.h`<br>PyTorch 1 个参数，Paddle 2 个参数 |
| 25 | `at::pixel_shuffle` | `paddle::experimental::pixel_shuffle` | paddle 参数更多 | 头文件: `ATen/ops/pixel_shuffle.h`<br>PyTorch 2 个参数，Paddle 3 个参数 |
| 26 | `at::pixel_unshuffle` | `paddle::experimental::pixel_unshuffle` | paddle 参数更多 | 头文件: `ATen/ops/pixel_unshuffle.h`<br>PyTorch 2 个参数，Paddle 3 个参数 |
| 27 | `at::prelu` | `paddle::experimental::prelu` | paddle 参数更多 | 头文件: `ATen/ops/prelu.h`<br>PyTorch 2 个参数，Paddle 4 个参数 |
| 28 | `at::prod` | `paddle::experimental::prod` | paddle 参数更多 | 头文件: `ATen/ops/prod.h`<br>PyTorch 2 个参数，Paddle 4 个参数 |
| 29 | `at::randint` | `paddle::experimental::randint` | paddle 参数更多 | 头文件: `ATen/ops/randint.h`<br>PyTorch 3 个参数，Paddle 5 个参数 |
| 30 | `at::random` | `paddle::experimental::random` | paddle 参数更多 | 头文件: `ATen/ops/random.h`<br>PyTorch 2 个参数，Paddle 3 个参数 |
| 31 | `at::randperm` | `paddle::experimental::randperm` | paddle 参数更多 | 头文件: `ATen/ops/randperm.h`<br>PyTorch 2 个参数，Paddle 3 个参数 |
| 32 | `at::repeat_interleave` | `paddle::experimental::repeat_interleave` | paddle 参数更多 | 头文件: `ATen/ops/repeat_interleave.h`<br>PyTorch 2 个参数，Paddle 4 个参数 |
| 33 | `at::round` | `paddle::experimental::round` | paddle 参数更多 | 头文件: `ATen/ops/round.h`<br>PyTorch 1 个参数，Paddle 2 个参数 |
| 34 | `at::selu` | `paddle::experimental::selu` | paddle 参数更多 | 头文件: `ATen/ops/selu.h`<br>PyTorch 1 个参数，Paddle 3 个参数 |
| 35 | `at::set` | `paddle::experimental::set` | paddle 参数更多 | 头文件: `ATen/ops/set.h`<br>PyTorch 1 个参数，Paddle 5 个参数 |
| 36 | `at::trace` | `paddle::experimental::trace` | paddle 参数更多 | 头文件: `ATen/ops/trace.h`<br>PyTorch 1 个参数，Paddle 4 个参数 |
| 37 | `at::tril_indices` | `paddle::experimental::tril_indices` | paddle 参数更多 | 头文件: `ATen/ops/tril_indices.h`<br>PyTorch 4 个参数，Paddle 5 个参数 |
| 38 | `at::triu_indices` | `paddle::experimental::triu_indices` | paddle 参数更多 | 头文件: `ATen/ops/triu_indices.h`<br>PyTorch 4 个参数，Paddle 5 个参数 |
| 39 | `at::uniform` | `paddle::experimental::uniform` | paddle 参数更多 | 头文件: `ATen/ops/uniform.h`<br>PyTorch 4 个参数，Paddle 6 个参数 |
| 40 | `at::var` | `paddle::experimental::var` | paddle 参数更多 | 头文件: `ATen/ops/var.h`<br>PyTorch 2 个参数，Paddle 5 个参数 |
| 41 | `at::_fft_c2r` | `paddle::experimental::_fft_c2r` | paddle 参数更多 | 头文件: `ATen/ops/_fft_c2r.h`<br>PyTorch 4 个参数，Paddle 5 个参数 |
| 42 | `at::_fft_r2c` | `paddle::experimental::_fft_r2c` | paddle 参数更多 | 头文件: `ATen/ops/_fft_r2c.h`<br>PyTorch 4 个参数，Paddle 5 个参数 |
| 43 | `at::_logcumsumexp` | `paddle::experimental::_logcumsumexp` | paddle 参数更多 | 头文件: `ATen/ops/_logcumsumexp.h`<br>PyTorch 2 个参数，Paddle 5 个参数 |
| 44 | `at::conv_transpose2d` | `paddle::experimental::conv_transpose2d` | paddle 参数更多 | 头文件: `ATen/ops/conv_transpose2d.h`<br>PyTorch 8 个参数，Paddle 10 个参数 |
| 45 | `at::conv_transpose3d` | `paddle::experimental::conv_transpose3d` | paddle 参数更多 | 头文件: `ATen/ops/conv_transpose3d.h`<br>PyTorch 8 个参数，Paddle 10 个参数 |
| 46 | `at::range` | `paddle::experimental::range` | paddle 参数更多 | 头文件: `ATen/ops/range.h`<br>PyTorch 3 个参数，Paddle 5 个参数 |

### 5. 参数默认值不一致

**简介：** 此类 API 功能相同，但某些参数的默认值不同。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::fill` | `paddle::experimental::fill` | 参数默认值不一致 | 头文件: `ATen/ops/fill.h`<br>存在参数默认值差异 |
| 2 | `at::roll` | `paddle::experimental::roll` | 参数默认值不一致 | 头文件: `ATen/ops/roll.h`<br>存在参数默认值差异 |
| 3 | `at::tile` | `paddle::experimental::tile` | 参数默认值不一致 | 头文件: `ATen/ops/tile.h`<br>存在参数默认值差异 |

### 6. torch 参数更多

**简介：** 此类 API 在 PyTorch 中提供了更多参数。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::add` | `paddle::experimental::add` | torch 参数更多 | 头文件: `ATen/ops/add.h`<br>PyTorch 3 个参数，Paddle 2 个参数 |
| 2 | `at::bernoulli` | `paddle::experimental::bernoulli` | torch 参数更多 | 头文件: `ATen/ops/bernoulli.h`<br>PyTorch 2 个参数，Paddle 1 个参数 |
| 3 | `at::binomial` | `paddle::experimental::binomial` | torch 参数更多 | 头文件: `ATen/ops/binomial.h`<br>PyTorch 3 个参数，Paddle 2 个参数 |
| 4 | `at::elu` | `paddle::experimental::elu` | torch 参数更多 | 头文件: `ATen/ops/elu.h`<br>PyTorch 4 个参数，Paddle 2 个参数 |
| 5 | `at::embedding` | `paddle::experimental::embedding` | torch 参数更多 | 头文件: `ATen/ops/embedding.h`<br>PyTorch 5 个参数，Paddle 4 个参数 |
| 6 | `at::gather` | `paddle::experimental::gather` | torch 参数更多 | 头文件: `ATen/ops/gather.h`<br>PyTorch 4 个参数，Paddle 3 个参数 |
| 7 | `at::huber_loss` | `paddle::experimental::huber_loss` | torch 参数更多 | 头文件: `ATen/ops/huber_loss.h`<br>PyTorch 4 个参数，Paddle 3 个参数 |
| 8 | `at::index_add` | `paddle::experimental::index_add` | torch 参数更多 | 头文件: `ATen/ops/index_add.h`<br>PyTorch 5 个参数，Paddle 4 个参数 |
| 9 | `at::instance_norm` | `paddle::experimental::instance_norm` | torch 参数更多 | 头文件: `ATen/ops/instance_norm.h`<br>PyTorch 9 个参数，Paddle 4 个参数 |
| 10 | `at::layer_norm` | `paddle::experimental::layer_norm` | torch 参数更多 | 头文件: `ATen/ops/layer_norm.h`<br>PyTorch 6 个参数，Paddle 5 个参数 |
| 11 | `at::log_softmax` | `paddle::experimental::log_softmax` | torch 参数更多 | 头文件: `ATen/ops/log_softmax.h`<br>PyTorch 3 个参数，Paddle 2 个参数 |
| 12 | `at::multinomial` | `paddle::experimental::multinomial` | torch 参数更多 | 头文件: `ATen/ops/multinomial.h`<br>PyTorch 4 个参数，Paddle 3 个参数 |
| 13 | `at::pad` | `paddle::experimental::pad` | torch 参数更多 | 头文件: `ATen/ops/pad.h`<br>PyTorch 4 个参数，Paddle 3 个参数 |
| 14 | `at::poisson` | `paddle::experimental::poisson` | torch 参数更多 | 头文件: `ATen/ops/poisson.h`<br>PyTorch 2 个参数，Paddle 1 个参数 |
| 15 | `at::rrelu` | `paddle::experimental::rrelu` | torch 参数更多 | 头文件: `ATen/ops/rrelu.h`<br>PyTorch 5 个参数，Paddle 4 个参数 |
| 16 | `at::searchsorted` | `paddle::experimental::searchsorted` | torch 参数更多 | 头文件: `ATen/ops/searchsorted.h`<br>PyTorch 6 个参数，Paddle 4 个参数 |
| 17 | `at::softmax` | `paddle::experimental::softmax` | torch 参数更多 | 头文件: `ATen/ops/softmax.h`<br>PyTorch 3 个参数，Paddle 2 个参数 |
| 18 | `at::stft` | `paddle::experimental::stft` | torch 参数更多 | 头文件: `ATen/ops/stft.h`<br>PyTorch 9 个参数，Paddle 6 个参数 |
| 19 | `at::subtract` | `paddle::experimental::subtract` | torch 参数更多 | 头文件: `ATen/ops/subtract.h`<br>PyTorch 3 个参数，Paddle 2 个参数 |
| 20 | `at::_log_softmax` | `paddle::experimental::_log_softmax` | torch 参数更多 | 头文件: `ATen/ops/_log_softmax.h`<br>PyTorch 3 个参数，Paddle 2 个参数 |
| 21 | `at::_softmax` | `paddle::experimental::_softmax` | torch 参数更多 | 头文件: `ATen/ops/_softmax.h`<br>PyTorch 3 个参数，Paddle 2 个参数 |
| 22 | `at::_standard_gamma` | `paddle::experimental::_standard_gamma` | torch 参数更多 | 头文件: `ATen/ops/_standard_gamma.h`<br>PyTorch 2 个参数，Paddle 1 个参数 |

### 7. 输入参数用法不一致

**简介：** 此类 API 对输入参数的处理方式不同。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| - | - | - | - | 暂无 |

### 8. 输入参数类型不一致

**简介：** 此类 API 要求的输入数据类型不同。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::addmm` | `paddle::experimental::addmm` | 输入参数类型不一致 | 头文件: `ATen/ops/addmm.h`<br>存在参数类型差异 |
| 2 | `at::bilinear` | `paddle::experimental::bilinear` | 输入参数类型不一致 | 头文件: `ATen/ops/bilinear.h`<br>存在参数类型差异 |
| 3 | `at::bincount` | `paddle::experimental::bincount` | 输入参数类型不一致 | 头文件: `ATen/ops/bincount.h`<br>存在参数类型差异 |
| 4 | `at::bitwise_and` | `paddle::experimental::bitwise_and` | 输入参数类型不一致 | 头文件: `ATen/ops/bitwise_and.h`<br>存在参数类型差异 |
| 5 | `at::bitwise_or` | `paddle::experimental::bitwise_or` | 输入参数类型不一致 | 头文件: `ATen/ops/bitwise_or.h`<br>存在参数类型差异 |
| 6 | `at::bitwise_xor` | `paddle::experimental::bitwise_xor` | 输入参数类型不一致 | 头文件: `ATen/ops/bitwise_xor.h`<br>存在参数类型差异 |
| 7 | `at::broadcast_tensors` | `paddle::experimental::broadcast_tensors` | 输入参数类型不一致 | 头文件: `ATen/ops/broadcast_tensors.h`<br>存在参数类型差异 |
| 8 | `at::celu` | `paddle::experimental::celu` | 输入参数类型不一致 | 头文件: `ATen/ops/celu.h`<br>存在参数类型差异 |
| 9 | `at::clip` | `paddle::experimental::clip` | 输入参数类型不一致 | 头文件: `ATen/ops/clip.h`<br>存在参数类型差异 |
| 10 | `at::concat` | `paddle::experimental::concat` | 输入参数类型不一致 | 头文件: `ATen/ops/concat.h`<br>存在参数类型差异 |
| 11 | `at::cross` | `paddle::experimental::cross` | 输入参数类型不一致 | 头文件: `ATen/ops/cross.h`<br>存在参数类型差异 |
| 12 | `at::diag_embed` | `paddle::experimental::diag_embed` | 输入参数类型不一致 | 头文件: `ATen/ops/diag_embed.h`<br>存在参数类型差异 |
| 13 | `at::diagonal` | `paddle::experimental::diagonal` | 输入参数类型不一致 | 头文件: `ATen/ops/diagonal.h`<br>存在参数类型差异 |
| 14 | `at::dist` | `paddle::experimental::dist` | 输入参数类型不一致 | 头文件: `ATen/ops/dist.h`<br>存在参数类型差异 |
| 15 | `at::flip` | `paddle::experimental::flip` | 输入参数类型不一致 | 头文件: `ATen/ops/flip.h`<br>存在参数类型差异 |
| 16 | `at::full_like` | `paddle::experimental::full_like` | 输入参数类型不一致 | 头文件: `ATen/ops/full_like.h`<br>存在参数类型差异 |
| 17 | `at::gelu` | `paddle::experimental::gelu` | 输入参数类型不一致 | 头文件: `ATen/ops/gelu.h`<br>存在参数类型差异 |
| 18 | `at::greater_equal` | `paddle::experimental::greater_equal` | 输入参数类型不一致 | 头文件: `ATen/ops/greater_equal.h`<br>存在参数类型差异 |
| 19 | `at::group_norm` | `paddle::experimental::group_norm` | 输入参数类型不一致 | 头文件: `ATen/ops/group_norm.h`<br>存在参数类型差异 |
| 20 | `at::hardshrink` | `paddle::experimental::hardshrink` | 输入参数类型不一致 | 头文件: `ATen/ops/hardshrink.h`<br>存在参数类型差异 |
| 21 | `at::hardtanh` | `paddle::experimental::hardtanh` | 输入参数类型不一致 | 头文件: `ATen/ops/hardtanh.h`<br>存在参数类型差异 |
| 22 | `at::index_fill` | `paddle::experimental::index_fill` | 输入参数类型不一致 | 头文件: `ATen/ops/index_fill.h`<br>存在参数类型差异 |
| 23 | `at::index_select` | `paddle::experimental::index_select` | 输入参数类型不一致 | 头文件: `ATen/ops/index_select.h`<br>存在参数类型差异 |
| 24 | `at::isclose` | `paddle::experimental::isclose` | 输入参数类型不一致 | 头文件: `ATen/ops/isclose.h`<br>存在参数类型差异 |
| 25 | `at::leaky_relu` | `paddle::experimental::leaky_relu` | 输入参数类型不一致 | 头文件: `ATen/ops/leaky_relu.h`<br>存在参数类型差异 |
| 26 | `at::lerp` | `paddle::experimental::lerp` | 输入参数类型不一致 | 头文件: `ATen/ops/lerp.h`<br>存在参数类型差异 |
| 27 | `at::less_equal` | `paddle::experimental::less_equal` | 输入参数类型不一致 | 头文件: `ATen/ops/less_equal.h`<br>存在参数类型差异 |
| 28 | `at::logit` | `paddle::experimental::logit` | 输入参数类型不一致 | 头文件: `ATen/ops/logit.h`<br>存在参数类型差异 |
| 29 | `at::masked_fill` | `paddle::experimental::masked_fill` | 输入参数类型不一致 | 头文件: `ATen/ops/masked_fill.h`<br>存在参数类型差异 |
| 30 | `at::matrix_power` | `paddle::experimental::matrix_power` | 输入参数类型不一致 | 头文件: `ATen/ops/matrix_power.h`<br>存在参数类型差异 |
| 31 | `at::meshgrid` | `paddle::experimental::meshgrid` | 输入参数类型不一致 | 头文件: `ATen/ops/meshgrid.h`<br>存在参数类型差异 |
| 32 | `at::nansum` | `paddle::experimental::nansum` | 输入参数类型不一致 | 头文件: `ATen/ops/nansum.h`<br>存在参数类型差异 |
| 33 | `at::not_equal` | `paddle::experimental::not_equal` | 输入参数类型不一致 | 头文件: `ATen/ops/not_equal.h`<br>存在参数类型差异 |
| 34 | `at::one_hot` | `paddle::experimental::one_hot` | 输入参数类型不一致 | 头文件: `ATen/ops/one_hot.h`<br>存在参数类型差异 |
| 35 | `at::ones_like` | `paddle::experimental::ones_like` | 输入参数类型不一致 | 头文件: `ATen/ops/ones_like.h`<br>存在参数类型差异 |
| 36 | `at::polygamma` | `paddle::experimental::polygamma` | 输入参数类型不一致 | 头文件: `ATen/ops/polygamma.h`<br>存在参数类型差异 |
| 37 | `at::pow` | `paddle::experimental::pow` | 输入参数类型不一致 | 头文件: `ATen/ops/pow.h`<br>存在参数类型差异 |
| 38 | `at::remainder` | `paddle::experimental::remainder` | 输入参数类型不一致 | 头文件: `ATen/ops/remainder.h`<br>存在参数类型差异 |
| 39 | `at::renorm` | `paddle::experimental::renorm` | 输入参数类型不一致 | 头文件: `ATen/ops/renorm.h`<br>存在参数类型差异 |
| 40 | `at::scatter` | `paddle::experimental::scatter` | 输入参数类型不一致 | 头文件: `ATen/ops/scatter.h`<br>存在参数类型差异 |
| 41 | `at::softplus` | `paddle::experimental::softplus` | 输入参数类型不一致 | 头文件: `ATen/ops/softplus.h`<br>存在参数类型差异 |
| 42 | `at::softshrink` | `paddle::experimental::softshrink` | 输入参数类型不一致 | 头文件: `ATen/ops/softshrink.h`<br>存在参数类型差异 |
| 43 | `at::stack` | `paddle::experimental::stack` | 输入参数类型不一致 | 头文件: `ATen/ops/stack.h`<br>存在参数类型差异 |
| 44 | `at::tril` | `paddle::experimental::tril` | 输入参数类型不一致 | 头文件: `ATen/ops/tril.h`<br>存在参数类型差异 |
| 45 | `at::triu` | `paddle::experimental::triu` | 输入参数类型不一致 | 头文件: `ATen/ops/triu.h`<br>存在参数类型差异 |
| 46 | `at::unbind` | `paddle::experimental::unbind` | 输入参数类型不一致 | 头文件: `ATen/ops/unbind.h`<br>存在参数类型差异 |
| 47 | `at::_fft_c2c` | `paddle::experimental::_fft_c2c` | 输入参数类型不一致 | 头文件: `ATen/ops/_fft_c2c.h`<br>存在参数类型差异 |
| 48 | `at::_stack` | `paddle::experimental::_stack` | 输入参数类型不一致 | 头文件: `ATen/ops/_stack.h`<br>存在参数类型差异 |
| 49 | `at::grid_sampler` | `paddle::experimental::grid_sampler` | 输入参数类型不一致 | 头文件: `ATen/ops/grid_sampler.h`<br>存在参数类型差异 |

### 9. 返回参数类型不一致

**简介：** 此类 API 返回值的类型或结构不同。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::aminmax` | `paddle::experimental::aminmax` | 返回参数类型不一致 | 头文件: `ATen/ops/aminmax.h`<br>返回类型不一致 |
| 2 | `at::argsort` | `paddle::experimental::argsort` | 返回参数类型不一致 | 头文件: `ATen/ops/argsort.h`<br>返回类型不一致 |
| 3 | `at::batch_norm` | `paddle::experimental::batch_norm` | 返回参数类型不一致 | 头文件: `ATen/ops/batch_norm.h`<br>返回类型不一致 |
| 4 | `at::cummax` | `paddle::experimental::cummax` | 返回参数类型不一致 | 头文件: `ATen/ops/cummax.h`<br>返回类型不一致 |
| 5 | `at::cummin` | `paddle::experimental::cummin` | 返回参数类型不一致 | 头文件: `ATen/ops/cummin.h`<br>返回类型不一致 |
| 6 | `at::fractional_max_pool2d` | `paddle::experimental::fractional_max_pool2d` | 返回参数类型不一致 | 头文件: `ATen/ops/fractional_max_pool2d.h`<br>返回类型不一致 |
| 7 | `at::fractional_max_pool3d` | `paddle::experimental::fractional_max_pool3d` | 返回参数类型不一致 | 头文件: `ATen/ops/fractional_max_pool3d.h`<br>返回类型不一致 |
| 8 | `at::gru` | `paddle::experimental::gru` | 返回参数类型不一致 | 头文件: `ATen/ops/gru.h`<br>返回类型不一致 |
| 9 | `at::histogram` | `paddle::experimental::histogram` | 返回参数类型不一致 | 头文件: `ATen/ops/histogram.h`<br>返回类型不一致 |
| 10 | `at::kthvalue` | `paddle::experimental::kthvalue` | 返回参数类型不一致 | 头文件: `ATen/ops/kthvalue.h`<br>返回类型不一致 |
| 11 | `at::lstm` | `paddle::experimental::lstm` | 返回参数类型不一致 | 头文件: `ATen/ops/lstm.h`<br>返回类型不一致 |
| 12 | `at::lu_unpack` | `paddle::experimental::lu_unpack` | 返回参数类型不一致 | 头文件: `ATen/ops/lu_unpack.h`<br>返回类型不一致 |
| 13 | `at::median` | `paddle::experimental::median` | 返回参数类型不一致 | 头文件: `ATen/ops/median.h`<br>返回类型不一致 |
| 14 | `at::mode` | `paddle::experimental::mode` | 返回参数类型不一致 | 头文件: `ATen/ops/mode.h`<br>返回类型不一致 |
| 15 | `at::nanmedian` | `paddle::experimental::nanmedian` | 返回参数类型不一致 | 头文件: `ATen/ops/nanmedian.h`<br>返回类型不一致 |
| 16 | `at::nll_loss` | `paddle::experimental::nll_loss` | 返回参数类型不一致 | 头文件: `ATen/ops/nll_loss.h`<br>返回类型不一致 |
| 17 | `at::norm` | `paddle::experimental::norm` | 返回参数类型不一致 | 头文件: `ATen/ops/norm.h`<br>返回类型不一致 |
| 18 | `at::qr` | `paddle::experimental::qr` | 返回参数类型不一致 | 头文件: `ATen/ops/qr.h`<br>返回类型不一致 |
| 19 | `at::rms_norm` | `paddle::experimental::rms_norm` | 返回参数类型不一致 | 头文件: `ATen/ops/rms_norm.h`<br>返回类型不一致 |
| 20 | `at::slogdet` | `paddle::experimental::slogdet` | 返回参数类型不一致 | 头文件: `ATen/ops/slogdet.h`<br>返回类型不一致 |
| 21 | `at::svd` | `paddle::experimental::svd` | 返回参数类型不一致 | 头文件: `ATen/ops/svd.h`<br>返回类型不一致 |
| 22 | `at::topk` | `paddle::experimental::topk` | 返回参数类型不一致 | 头文件: `ATen/ops/topk.h`<br>返回类型不一致 |
| 23 | `at::triangular_solve` | `paddle::experimental::triangular_solve` | 返回参数类型不一致 | 头文件: `ATen/ops/triangular_solve.h`<br>返回类型不一致 |
| 24 | `at::unique_consecutive` | `paddle::experimental::unique_consecutive` | 返回参数类型不一致 | 头文件: `ATen/ops/unique_consecutive.h`<br>返回类型不一致 |
| 25 | `at::where` | `paddle::experimental::where` | 返回参数类型不一致 | 头文件: `ATen/ops/where.h`<br>返回类型不一致 |
| 26 | `at::_aminmax` | `paddle::experimental::_aminmax` | 返回参数类型不一致 | 头文件: `ATen/ops/_aminmax.h`<br>返回类型不一致 |
| 27 | `at::_unique` | `paddle::experimental::_unique` | 返回参数类型不一致 | 头文件: `ATen/ops/_unique.h`<br>返回类型不一致 |
| 28 | `at::max_pool2d_with_indices` | `paddle::experimental::max_pool2d_with_indices` | 返回参数类型不一致 | 头文件: `ATen/ops/max_pool2d_with_indices.h`<br>返回类型不一致 |
| 29 | `at::max_pool3d_with_indices` | `paddle::experimental::max_pool3d_with_indices` | 返回参数类型不一致 | 头文件: `ATen/ops/max_pool3d_with_indices.h`<br>返回类型不一致 |

### 10. 组合替代实现

**简介：** 此类功能在 Paddle 中没有直接对应的单一 API，需要通过多个 Paddle API 组合来实现。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| - | - | - | - | 暂无 |

### 11. API 别名

**简介：** 此类 PyTorch API 在 Paddle 中有功能一致的实现，但 API 名称不同。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::_aminmax` | `paddle::experimental::aminmax` | API 别名 | 头文件: `ATen/ops/_aminmax.h`<br>返回类型不一致 |
| 2 | `at::_conj` | `paddle::experimental::conj` | API 别名 | 头文件: `ATen/ops/_conj.h`<br>参数类型和默认值相同，仅参数名不同 |
| 3 | `at::_fft_c2c` | `paddle::experimental::fft_c2c` | API 别名 | 头文件: `ATen/ops/_fft_c2c.h`<br>存在参数类型差异 |
| 4 | `at::_fft_c2r` | `paddle::experimental::fft_c2r` | API 别名 | 头文件: `ATen/ops/_fft_c2r.h`<br>PyTorch 4 个参数，Paddle 5 个参数 |
| 5 | `at::_fft_r2c` | `paddle::experimental::fft_r2c` | API 别名 | 头文件: `ATen/ops/_fft_r2c.h`<br>PyTorch 4 个参数，Paddle 5 个参数 |
| 6 | `at::_log_softmax` | `paddle::experimental::log_softmax` | API 别名 | 头文件: `ATen/ops/_log_softmax.h`<br>PyTorch 3 个参数，Paddle 2 个参数 |
| 7 | `at::_logcumsumexp` | `paddle::experimental::logcumsumexp` | API 别名 | 头文件: `ATen/ops/_logcumsumexp.h`<br>PyTorch 2 个参数，Paddle 5 个参数 |
| 8 | `at::_softmax` | `paddle::experimental::softmax` | API 别名 | 头文件: `ATen/ops/_softmax.h`<br>PyTorch 3 个参数，Paddle 2 个参数 |
| 9 | `at::_stack` | `paddle::experimental::stack` | API 别名 | 头文件: `ATen/ops/_stack.h`<br>存在参数类型差异 |
| 10 | `at::_standard_gamma` | `paddle::experimental::standard_gamma` | API 别名 | 头文件: `ATen/ops/_standard_gamma.h`<br>PyTorch 2 个参数，Paddle 1 个参数 |
| 11 | `at::_unique` | `paddle::experimental::unique` | API 别名 | 头文件: `ATen/ops/_unique.h`<br>返回类型不一致 |
| 12 | `at::conv_transpose2d` | `paddle::experimental::conv2d_transpose` | API 别名 | 头文件: `ATen/ops/conv_transpose2d.h`<br>PyTorch 8 个参数，Paddle 10 个参数 |
| 13 | `at::conv_transpose3d` | `paddle::experimental::conv3d_transpose` | API 别名 | 头文件: `ATen/ops/conv_transpose3d.h`<br>PyTorch 8 个参数，Paddle 10 个参数 |
| 14 | `at::grid_sampler` | `paddle::experimental::grid_sample` | API 别名 | 头文件: `ATen/ops/grid_sampler.h`<br>存在参数类型差异 |
| 15 | `at::log_sigmoid` | `paddle::experimental::logsigmoid` | API 别名 | 头文件: `ATen/ops/log_sigmoid.h`<br>参数类型和默认值相同，仅参数名不同 |
| 16 | `at::max_pool2d_with_indices` | `paddle::experimental::max_pool2d_with_index` | API 别名 | 头文件: `ATen/ops/max_pool2d_with_indices.h`<br>返回类型不一致 |
| 17 | `at::max_pool3d_with_indices` | `paddle::experimental::max_pool3d_with_index` | API 别名 | 头文件: `ATen/ops/max_pool3d_with_indices.h`<br>返回类型不一致 |
| 18 | `at::range` | `paddle::experimental::arange` | API 别名 | 头文件: `ATen/ops/range.h`<br>PyTorch 3 个参数，Paddle 5 个参数 |

### 12. 功能缺失

**简介：** 此类 PyTorch C++ API 在 Paddle 中暂时没有等效实现。

| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |
|------|-----------------|----------------|----------|------|
| 1 | `at::_adaptive_avg_pool2d` | - | 功能缺失 | 头文件: `ATen/ops/_adaptive_avg_pool2d.h` |
| 2 | `at::_adaptive_avg_pool3d` | - | 功能缺失 | 头文件: `ATen/ops/_adaptive_avg_pool3d.h` |
| 3 | `at::_add_batch_dim` | - | 功能缺失 | 头文件: `ATen/ops/_add_batch_dim.h` |
| 4 | `at::_add_relu` | - | 功能缺失 | 头文件: `ATen/ops/_add_relu.h` |
| 5 | `at::_addmm_activation` | - | 功能缺失 | 头文件: `ATen/ops/_addmm_activation.h` |
| 6 | `at::_amp_foreach_non_finite_check_and_unscale` | - | 功能缺失 | 头文件: `ATen/ops/_amp_foreach_non_finite_check_and_unscale.h` |
| 7 | `at::_amp_update_scale` | - | 功能缺失 | 头文件: `ATen/ops/_amp_update_scale.h` |
| 8 | `at::_assert_async` | - | 功能缺失 | 头文件: `ATen/ops/_assert_async.h` |
| 9 | `at::_assert_scalar` | - | 功能缺失 | 头文件: `ATen/ops/_assert_scalar.h` |
| 10 | `at::_autocast_to_full_precision` | - | 功能缺失 | 头文件: `ATen/ops/_autocast_to_full_precision.h` |
| 11 | `at::_autocast_to_reduced_precision` | - | 功能缺失 | 头文件: `ATen/ops/_autocast_to_reduced_precision.h` |
| 12 | `at::_batch_norm_impl_index` | - | 功能缺失 | 头文件: `ATen/ops/_batch_norm_impl_index.h` |
| 13 | `at::_batch_norm_no_update` | - | 功能缺失 | 头文件: `ATen/ops/_batch_norm_no_update.h` |
| 14 | `at::_batch_norm_with_update` | - | 功能缺失 | 头文件: `ATen/ops/_batch_norm_with_update.h` |
| 15 | `at::_cast_Byte` | - | 功能缺失 | 头文件: `ATen/ops/_cast_Byte.h` |
| 16 | `at::_cast_Char` | - | 功能缺失 | 头文件: `ATen/ops/_cast_Char.h` |
| 17 | `at::_cast_Double` | - | 功能缺失 | 头文件: `ATen/ops/_cast_Double.h` |
| 18 | `at::_cast_Float` | - | 功能缺失 | 头文件: `ATen/ops/_cast_Float.h` |
| 19 | `at::_cast_Half` | - | 功能缺失 | 头文件: `ATen/ops/_cast_Half.h` |
| 20 | `at::_cast_Int` | - | 功能缺失 | 头文件: `ATen/ops/_cast_Int.h` |
| 21 | `at::_cast_Long` | - | 功能缺失 | 头文件: `ATen/ops/_cast_Long.h` |
| 22 | `at::_cast_Short` | - | 功能缺失 | 头文件: `ATen/ops/_cast_Short.h` |
| 23 | `at::_cholesky_solve_helper` | - | 功能缺失 | 头文件: `ATen/ops/_cholesky_solve_helper.h` |
| 24 | `at::_choose_qparams_per_tensor` | - | 功能缺失 | 头文件: `ATen/ops/_choose_qparams_per_tensor.h` |
| 25 | `at::_chunk_cat` | - | 功能缺失 | 头文件: `ATen/ops/_chunk_cat.h` |
| 26 | `at::_coalesce` | - | 功能缺失 | 头文件: `ATen/ops/_coalesce.h` |
| 27 | `at::_coalesced` | - | 功能缺失 | 头文件: `ATen/ops/_coalesced.h` |
| 28 | `at::_compute_linear_combination` | - | 功能缺失 | 头文件: `ATen/ops/_compute_linear_combination.h` |
| 29 | `at::_conj_copy` | - | 功能缺失 | 头文件: `ATen/ops/_conj_copy.h` |
| 30 | `at::_conj_physical` | - | 功能缺失 | 头文件: `ATen/ops/_conj_physical.h` |
| 31 | `at::_conv_depthwise2d` | - | 功能缺失 | 头文件: `ATen/ops/_conv_depthwise2d.h` |
| 32 | `at::_convert_indices_from_coo_to_csr` | - | 功能缺失 | 头文件: `ATen/ops/_convert_indices_from_coo_to_csr.h` |
| 33 | `at::_convert_indices_from_csr_to_coo` | - | 功能缺失 | 头文件: `ATen/ops/_convert_indices_from_csr_to_coo.h` |
| 34 | `at::_convert_weight_to_int4pack` | - | 功能缺失 | 头文件: `ATen/ops/_convert_weight_to_int4pack.h` |
| 35 | `at::_convolution` | - | 功能缺失 | 头文件: `ATen/ops/_convolution.h` |
| 36 | `at::_convolution_mode` | - | 功能缺失 | 头文件: `ATen/ops/_convolution_mode.h` |
| 37 | `at::_copy_from` | - | 功能缺失 | 头文件: `ATen/ops/_copy_from.h` |
| 38 | `at::_copy_from_and_resize` | - | 功能缺失 | 头文件: `ATen/ops/_copy_from_and_resize.h` |
| 39 | `at::_cslt_compress` | - | 功能缺失 | 头文件: `ATen/ops/_cslt_compress.h` |
| 40 | `at::_ctc_loss` | - | 功能缺失 | 头文件: `ATen/ops/_ctc_loss.h` |
| 41 | `at::_cudnn_ctc_loss` | - | 功能缺失 | 头文件: `ATen/ops/_cudnn_ctc_loss.h` |
| 42 | `at::_cudnn_init_dropout_state` | - | 功能缺失 | 头文件: `ATen/ops/_cudnn_init_dropout_state.h` |
| 43 | `at::_cudnn_rnn` | - | 功能缺失 | 头文件: `ATen/ops/_cudnn_rnn.h` |
| 44 | `at::_cudnn_rnn_flatten_weight` | - | 功能缺失 | 头文件: `ATen/ops/_cudnn_rnn_flatten_weight.h` |
| 45 | `at::_cufft_clear_plan_cache` | - | 功能缺失 | 头文件: `ATen/ops/_cufft_clear_plan_cache.h` |
| 46 | `at::_cufft_get_plan_cache_max_size` | - | 功能缺失 | 头文件: `ATen/ops/_cufft_get_plan_cache_max_size.h` |
| 47 | `at::_cufft_get_plan_cache_size` | - | 功能缺失 | 头文件: `ATen/ops/_cufft_get_plan_cache_size.h` |
| 48 | `at::_cufft_set_plan_cache_max_size` | - | 功能缺失 | 头文件: `ATen/ops/_cufft_set_plan_cache_max_size.h` |
| 49 | `at::_cummax_helper` | - | 功能缺失 | 头文件: `ATen/ops/_cummax_helper.h` |
| 50 | `at::_cummin_helper` | - | 功能缺失 | 头文件: `ATen/ops/_cummin_helper.h` |
| 51 | `at::_debug_has_internal_overlap` | - | 功能缺失 | 头文件: `ATen/ops/_debug_has_internal_overlap.h` |
| 52 | `at::_dimI` | - | 功能缺失 | 头文件: `ATen/ops/_dimI.h` |
| 53 | `at::_dimV` | - | 功能缺失 | 头文件: `ATen/ops/_dimV.h` |
| 54 | `at::_dim_arange` | - | 功能缺失 | 头文件: `ATen/ops/_dim_arange.h` |
| 55 | `at::_dirichlet_grad` | - | 功能缺失 | 头文件: `ATen/ops/_dirichlet_grad.h` |
| 56 | `at::_dyn_quant_matmul_4bit` | - | 功能缺失 | 头文件: `ATen/ops/_dyn_quant_matmul_4bit.h` |
| 57 | `at::_dyn_quant_pack_4bit_weight` | - | 功能缺失 | 头文件: `ATen/ops/_dyn_quant_pack_4bit_weight.h` |
| 58 | `at::_efficientzerotensor` | - | 功能缺失 | 头文件: `ATen/ops/_efficientzerotensor.h` |
| 59 | `at::_embedding_bag` | - | 功能缺失 | 头文件: `ATen/ops/_embedding_bag.h` |
| 60 | `at::_euclidean_dist` | - | 功能缺失 | 头文件: `ATen/ops/_euclidean_dist.h` |
| 61 | `at::_fake_quantize_learnable_per_channel_affine` | - | 功能缺失 | 头文件: `ATen/ops/_fake_quantize_learnable_per_channel_affine.h` |
| 62 | `at::_fake_quantize_learnable_per_tensor_affine` | - | 功能缺失 | 头文件: `ATen/ops/_fake_quantize_learnable_per_tensor_affine.h` |
| 63 | `at::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams` | - | 功能缺失 | 头文件: `ATen/ops/_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.h` |
| 64 | `at::_fill_mem_eff_dropout_mask` | - | 功能缺失 | 头文件: `ATen/ops/_fill_mem_eff_dropout_mask.h` |
| 65 | `at::_foobar` | - | 功能缺失 | 头文件: `ATen/ops/_foobar.h` |
| 66 | `at::_foreach_abs` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_abs.h` |
| 67 | `at::_foreach_acos` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_acos.h` |
| 68 | `at::_foreach_add` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_add.h` |
| 69 | `at::_foreach_addcdiv` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_addcdiv.h` |
| 70 | `at::_foreach_addcmul` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_addcmul.h` |
| 71 | `at::_foreach_asin` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_asin.h` |
| 72 | `at::_foreach_atan` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_atan.h` |
| 73 | `at::_foreach_ceil` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_ceil.h` |
| 74 | `at::_foreach_clamp_max` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_clamp_max.h` |
| 75 | `at::_foreach_clamp_min` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_clamp_min.h` |
| 76 | `at::_foreach_copy` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_copy.h` |
| 77 | `at::_foreach_cos` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_cos.h` |
| 78 | `at::_foreach_cosh` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_cosh.h` |
| 79 | `at::_foreach_div` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_div.h` |
| 80 | `at::_foreach_erf` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_erf.h` |
| 81 | `at::_foreach_erfc` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_erfc.h` |
| 82 | `at::_foreach_exp` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_exp.h` |
| 83 | `at::_foreach_expm1` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_expm1.h` |
| 84 | `at::_foreach_floor` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_floor.h` |
| 85 | `at::_foreach_frac` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_frac.h` |
| 86 | `at::_foreach_lerp` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_lerp.h` |
| 87 | `at::_foreach_lgamma` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_lgamma.h` |
| 88 | `at::_foreach_log` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_log.h` |
| 89 | `at::_foreach_log10` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_log10.h` |
| 90 | `at::_foreach_log1p` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_log1p.h` |
| 91 | `at::_foreach_log2` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_log2.h` |
| 92 | `at::_foreach_max` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_max.h` |
| 93 | `at::_foreach_maximum` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_maximum.h` |
| 94 | `at::_foreach_minimum` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_minimum.h` |
| 95 | `at::_foreach_mul` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_mul.h` |
| 96 | `at::_foreach_neg` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_neg.h` |
| 97 | `at::_foreach_norm` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_norm.h` |
| 98 | `at::_foreach_pow` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_pow.h` |
| 99 | `at::_foreach_reciprocal` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_reciprocal.h` |
| 100 | `at::_foreach_round` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_round.h` |
| 101 | `at::_foreach_rsqrt` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_rsqrt.h` |
| 102 | `at::_foreach_sigmoid` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_sigmoid.h` |
| 103 | `at::_foreach_sign` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_sign.h` |
| 104 | `at::_foreach_sin` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_sin.h` |
| 105 | `at::_foreach_sinh` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_sinh.h` |
| 106 | `at::_foreach_sqrt` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_sqrt.h` |
| 107 | `at::_foreach_sub` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_sub.h` |
| 108 | `at::_foreach_tan` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_tan.h` |
| 109 | `at::_foreach_tanh` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_tanh.h` |
| 110 | `at::_foreach_trunc` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_trunc.h` |
| 111 | `at::_foreach_zero` | - | 功能缺失 | 头文件: `ATen/ops/_foreach_zero.h` |
| 112 | `at::_fused_adagrad` | - | 功能缺失 | 头文件: `ATen/ops/_fused_adagrad.h` |
| 113 | `at::_fused_adam` | - | 功能缺失 | 头文件: `ATen/ops/_fused_adam.h` |
| 114 | `at::_fused_adamw` | - | 功能缺失 | 头文件: `ATen/ops/_fused_adamw.h` |
| 115 | `at::_fused_dropout` | - | 功能缺失 | 头文件: `ATen/ops/_fused_dropout.h` |
| 116 | `at::_fused_moving_avg_obs_fq_helper` | - | 功能缺失 | 头文件: `ATen/ops/_fused_moving_avg_obs_fq_helper.h` |
| 117 | `at::_fused_rms_norm` | - | 功能缺失 | 头文件: `ATen/ops/_fused_rms_norm.h` |
| 118 | `at::_fused_sdp_choice` | - | 功能缺失 | 头文件: `ATen/ops/_fused_sdp_choice.h` |
| 119 | `at::_fused_sgd` | - | 功能缺失 | 头文件: `ATen/ops/_fused_sgd.h` |
| 120 | `at::_fw_primal` | - | 功能缺失 | 头文件: `ATen/ops/_fw_primal.h` |
| 121 | `at::_fw_primal_copy` | - | 功能缺失 | 头文件: `ATen/ops/_fw_primal_copy.h` |
| 122 | `at::_grouped_mm` | - | 功能缺失 | 头文件: `ATen/ops/_grouped_mm.h` |
| 123 | `at::_has_compatible_shallow_copy_type` | - | 功能缺失 | 头文件: `ATen/ops/_has_compatible_shallow_copy_type.h` |
| 124 | `at::_has_same_storage_numel` | - | 功能缺失 | 头文件: `ATen/ops/_has_same_storage_numel.h` |
| 125 | `at::_histogramdd_bin_edges` | - | 功能缺失 | 头文件: `ATen/ops/_histogramdd_bin_edges.h` |
| 126 | `at::_histogramdd_from_bin_cts` | - | 功能缺失 | 头文件: `ATen/ops/_histogramdd_from_bin_cts.h` |
| 127 | `at::_histogramdd_from_bin_tensors` | - | 功能缺失 | 头文件: `ATen/ops/_histogramdd_from_bin_tensors.h` |
| 128 | `at::_index_put_impl` | - | 功能缺失 | 头文件: `ATen/ops/_index_put_impl.h` |
| 129 | `at::_indices` | - | 功能缺失 | 头文件: `ATen/ops/_indices.h` |
| 130 | `at::_indices_copy` | - | 功能缺失 | 头文件: `ATen/ops/_indices_copy.h` |
| 131 | `at::_int_mm` | - | 功能缺失 | 头文件: `ATen/ops/_int_mm.h` |
| 132 | `at::_is_all_true` | - | 功能缺失 | 头文件: `ATen/ops/_is_all_true.h` |
| 133 | `at::_is_any_true` | - | 功能缺失 | 头文件: `ATen/ops/_is_any_true.h` |
| 134 | `at::_is_zerotensor` | - | 功能缺失 | 头文件: `ATen/ops/_is_zerotensor.h` |
| 135 | `at::_lazy_clone` | - | 功能缺失 | 头文件: `ATen/ops/_lazy_clone.h` |
| 136 | `at::_linalg_check_errors` | - | 功能缺失 | 头文件: `ATen/ops/_linalg_check_errors.h` |
| 137 | `at::_linalg_det` | - | 功能缺失 | 头文件: `ATen/ops/_linalg_det.h` |
| 138 | `at::_linalg_eigh` | - | 功能缺失 | 头文件: `ATen/ops/_linalg_eigh.h` |
| 139 | `at::_linalg_eigvals` | - | 功能缺失 | 头文件: `ATen/ops/_linalg_eigvals.h` |
| 140 | `at::_linalg_slogdet` | - | 功能缺失 | 头文件: `ATen/ops/_linalg_slogdet.h` |
| 141 | `at::_linalg_solve_ex` | - | 功能缺失 | 头文件: `ATen/ops/_linalg_solve_ex.h` |
| 142 | `at::_linalg_svd` | - | 功能缺失 | 头文件: `ATen/ops/_linalg_svd.h` |
| 143 | `at::_lu_with_info` | - | 功能缺失 | 头文件: `ATen/ops/_lu_with_info.h` |
| 144 | `at::_make_dep_token` | - | 功能缺失 | 头文件: `ATen/ops/_make_dep_token.h` |
| 145 | `at::_make_dual` | - | 功能缺失 | 头文件: `ATen/ops/_make_dual.h` |
| 146 | `at::_make_dual_copy` | - | 功能缺失 | 头文件: `ATen/ops/_make_dual_copy.h` |
| 147 | `at::_masked_scale` | - | 功能缺失 | 头文件: `ATen/ops/_masked_scale.h` |
| 148 | `at::_masked_softmax` | - | 功能缺失 | 头文件: `ATen/ops/_masked_softmax.h` |
| 149 | `at::_mixed_dtypes_linear` | - | 功能缺失 | 头文件: `ATen/ops/_mixed_dtypes_linear.h` |
| 150 | `at::_neg_view` | - | 功能缺失 | 头文件: `ATen/ops/_neg_view.h` |
| 151 | `at::_neg_view_copy` | - | 功能缺失 | 头文件: `ATen/ops/_neg_view_copy.h` |
| 152 | `at::_nested_compute_contiguous_strides_offsets` | - | 功能缺失 | 头文件: `ATen/ops/_nested_compute_contiguous_strides_offsets.h` |
| 153 | `at::_nested_from_padded` | - | 功能缺失 | 头文件: `ATen/ops/_nested_from_padded.h` |
| 154 | `at::_nested_from_padded_and_nested_example` | - | 功能缺失 | 头文件: `ATen/ops/_nested_from_padded_and_nested_example.h` |
| 155 | `at::_nested_from_padded_tensor` | - | 功能缺失 | 头文件: `ATen/ops/_nested_from_padded_tensor.h` |
| 156 | `at::_nested_get_jagged_dummy` | - | 功能缺失 | 头文件: `ATen/ops/_nested_get_jagged_dummy.h` |
| 157 | `at::_nested_get_lengths` | - | 功能缺失 | 头文件: `ATen/ops/_nested_get_lengths.h` |
| 158 | `at::_nested_get_max_seqlen` | - | 功能缺失 | 头文件: `ATen/ops/_nested_get_max_seqlen.h` |
| 159 | `at::_nested_get_min_seqlen` | - | 功能缺失 | 头文件: `ATen/ops/_nested_get_min_seqlen.h` |
| 160 | `at::_nested_get_offsets` | - | 功能缺失 | 头文件: `ATen/ops/_nested_get_offsets.h` |
| 161 | `at::_nested_get_ragged_idx` | - | 功能缺失 | 头文件: `ATen/ops/_nested_get_ragged_idx.h` |
| 162 | `at::_nested_get_values` | - | 功能缺失 | 头文件: `ATen/ops/_nested_get_values.h` |
| 163 | `at::_nested_get_values_copy` | - | 功能缺失 | 头文件: `ATen/ops/_nested_get_values_copy.h` |
| 164 | `at::_nested_tensor_from_mask` | - | 功能缺失 | 头文件: `ATen/ops/_nested_tensor_from_mask.h` |
| 165 | `at::_nested_tensor_from_mask_left_aligned` | - | 功能缺失 | 头文件: `ATen/ops/_nested_tensor_from_mask_left_aligned.h` |
| 166 | `at::_nested_tensor_from_tensor_list` | - | 功能缺失 | 头文件: `ATen/ops/_nested_tensor_from_tensor_list.h` |
| 167 | `at::_nested_tensor_size` | - | 功能缺失 | 头文件: `ATen/ops/_nested_tensor_size.h` |
| 168 | `at::_nested_tensor_softmax_with_shape` | - | 功能缺失 | 头文件: `ATen/ops/_nested_tensor_softmax_with_shape.h` |
| 169 | `at::_nested_tensor_storage_offsets` | - | 功能缺失 | 头文件: `ATen/ops/_nested_tensor_storage_offsets.h` |
| 170 | `at::_nested_tensor_strides` | - | 功能缺失 | 头文件: `ATen/ops/_nested_tensor_strides.h` |
| 171 | `at::_nested_view_from_buffer` | - | 功能缺失 | 头文件: `ATen/ops/_nested_view_from_buffer.h` |
| 172 | `at::_nested_view_from_buffer_copy` | - | 功能缺失 | 头文件: `ATen/ops/_nested_view_from_buffer_copy.h` |
| 173 | `at::_nested_view_from_jagged` | - | 功能缺失 | 头文件: `ATen/ops/_nested_view_from_jagged.h` |
| 174 | `at::_nested_view_from_jagged_copy` | - | 功能缺失 | 头文件: `ATen/ops/_nested_view_from_jagged_copy.h` |
| 175 | `at::_nnpack_available` | - | 功能缺失 | 头文件: `ATen/ops/_nnpack_available.h` |
| 176 | `at::_nnpack_spatial_convolution` | - | 功能缺失 | 头文件: `ATen/ops/_nnpack_spatial_convolution.h` |
| 177 | `at::_pack_padded_sequence` | - | 功能缺失 | 头文件: `ATen/ops/_pack_padded_sequence.h` |
| 178 | `at::_pad_circular` | - | 功能缺失 | 头文件: `ATen/ops/_pad_circular.h` |
| 179 | `at::_pad_enum` | - | 功能缺失 | 头文件: `ATen/ops/_pad_enum.h` |
| 180 | `at::_pad_packed_sequence` | - | 功能缺失 | 头文件: `ATen/ops/_pad_packed_sequence.h` |
| 181 | `at::_pin_memory` | - | 功能缺失 | 头文件: `ATen/ops/_pin_memory.h` |
| 182 | `at::_prelu_kernel` | - | 功能缺失 | 头文件: `ATen/ops/_prelu_kernel.h` |
| 183 | `at::_print` | - | 功能缺失 | 头文件: `ATen/ops/_print.h` |
| 184 | `at::_propagate_xla_data` | - | 功能缺失 | 头文件: `ATen/ops/_propagate_xla_data.h` |
| 185 | `at::_remove_batch_dim` | - | 功能缺失 | 头文件: `ATen/ops/_remove_batch_dim.h` |
| 186 | `at::_reshape_alias` | - | 功能缺失 | 头文件: `ATen/ops/_reshape_alias.h` |
| 187 | `at::_reshape_alias_copy` | - | 功能缺失 | 头文件: `ATen/ops/_reshape_alias_copy.h` |
| 188 | `at::_reshape_copy` | - | 功能缺失 | 头文件: `ATen/ops/_reshape_copy.h` |
| 189 | `at::_reshape_from_tensor` | - | 功能缺失 | 头文件: `ATen/ops/_reshape_from_tensor.h` |
| 190 | `at::_resize_output` | - | 功能缺失 | 头文件: `ATen/ops/_resize_output.h` |
| 191 | `at::_rowwise_prune` | - | 功能缺失 | 头文件: `ATen/ops/_rowwise_prune.h` |
| 192 | `at::_safe_softmax` | - | 功能缺失 | 头文件: `ATen/ops/_safe_softmax.h` |
| 193 | `at::_sample_dirichlet` | - | 功能缺失 | 头文件: `ATen/ops/_sample_dirichlet.h` |
| 194 | `at::_saturate_weight_to_fp16` | - | 功能缺失 | 头文件: `ATen/ops/_saturate_weight_to_fp16.h` |
| 195 | `at::_scaled_dot_product_cudnn_attention` | - | 功能缺失 | 头文件: `ATen/ops/_scaled_dot_product_cudnn_attention.h` |
| 196 | `at::_scaled_dot_product_efficient_attention` | - | 功能缺失 | 头文件: `ATen/ops/_scaled_dot_product_efficient_attention.h` |
| 197 | `at::_scaled_dot_product_flash_attention` | - | 功能缺失 | 头文件: `ATen/ops/_scaled_dot_product_flash_attention.h` |
| 198 | `at::_scaled_dot_product_fused_attention_overrideable` | - | 功能缺失 | 头文件: `ATen/ops/_scaled_dot_product_fused_attention_overrideable.h` |
| 199 | `at::_scaled_grouped_mm` | - | 功能缺失 | 头文件: `ATen/ops/_scaled_grouped_mm.h` |
| 200 | `at::_scaled_grouped_mm_v2` | - | 功能缺失 | 头文件: `ATen/ops/_scaled_grouped_mm_v2.h` |
| 201 | `at::_scaled_mm` | - | 功能缺失 | 头文件: `ATen/ops/_scaled_mm.h` |
| 202 | `at::_scaled_mm_v2` | - | 功能缺失 | 头文件: `ATen/ops/_scaled_mm_v2.h` |
| 203 | `at::_shape_as_tensor` | - | 功能缺失 | 头文件: `ATen/ops/_shape_as_tensor.h` |
| 204 | `at::_sobol_engine_draw` | - | 功能缺失 | 头文件: `ATen/ops/_sobol_engine_draw.h` |
| 205 | `at::_sobol_engine_ff` | - | 功能缺失 | 头文件: `ATen/ops/_sobol_engine_ff.h` |
| 206 | `at::_sobol_engine_initialize_state` | - | 功能缺失 | 头文件: `ATen/ops/_sobol_engine_initialize_state.h` |
| 207 | `at::_sobol_engine_scramble` | - | 功能缺失 | 头文件: `ATen/ops/_sobol_engine_scramble.h` |
| 208 | `at::_spdiags` | - | 功能缺失 | 头文件: `ATen/ops/_spdiags.h` |
| 209 | `at::_spsolve` | - | 功能缺失 | 头文件: `ATen/ops/_spsolve.h` |
| 210 | `at::_standard_gamma_grad` | - | 功能缺失 | 头文件: `ATen/ops/_standard_gamma_grad.h` |
| 211 | `at::_test_ambiguous_defaults` | - | 功能缺失 | 头文件: `ATen/ops/_test_ambiguous_defaults.h` |
| 212 | `at::_test_check_tensor` | - | 功能缺失 | 头文件: `ATen/ops/_test_check_tensor.h` |
| 213 | `at::_test_functorch_fallback` | - | 功能缺失 | 头文件: `ATen/ops/_test_functorch_fallback.h` |
| 214 | `at::_test_optional_filled_intlist` | - | 功能缺失 | 头文件: `ATen/ops/_test_optional_filled_intlist.h` |
| 215 | `at::_test_optional_floatlist` | - | 功能缺失 | 头文件: `ATen/ops/_test_optional_floatlist.h` |
| 216 | `at::_test_optional_intlist` | - | 功能缺失 | 头文件: `ATen/ops/_test_optional_intlist.h` |
| 217 | `at::_test_parallel_materialize` | - | 功能缺失 | 头文件: `ATen/ops/_test_parallel_materialize.h` |
| 218 | `at::_test_serialization_subcmul` | - | 功能缺失 | 头文件: `ATen/ops/_test_serialization_subcmul.h` |
| 219 | `at::_test_string_default` | - | 功能缺失 | 头文件: `ATen/ops/_test_string_default.h` |
| 220 | `at::_test_warn_in_autograd` | - | 功能缺失 | 头文件: `ATen/ops/_test_warn_in_autograd.h` |
| 221 | `at::_thnn_fused_gru_cell` | - | 功能缺失 | 头文件: `ATen/ops/_thnn_fused_gru_cell.h` |
| 222 | `at::_thnn_fused_lstm_cell` | - | 功能缺失 | 头文件: `ATen/ops/_thnn_fused_lstm_cell.h` |
| 223 | `at::_to_copy` | - | 功能缺失 | 头文件: `ATen/ops/_to_copy.h` |
| 224 | `at::_to_dense` | - | 功能缺失 | 头文件: `ATen/ops/_to_dense.h` |
| 225 | `at::_transform_bias_rescale_qkv` | - | 功能缺失 | 头文件: `ATen/ops/_transform_bias_rescale_qkv.h` |
| 226 | `at::_transformer_encoder_layer_fwd` | - | 功能缺失 | 头文件: `ATen/ops/_transformer_encoder_layer_fwd.h` |
| 227 | `at::_trilinear` | - | 功能缺失 | 头文件: `ATen/ops/_trilinear.h` |
| 228 | `at::_triton_multi_head_attention` | - | 功能缺失 | 头文件: `ATen/ops/_triton_multi_head_attention.h` |
| 229 | `at::_triton_scaled_dot_attention` | - | 功能缺失 | 头文件: `ATen/ops/_triton_scaled_dot_attention.h` |
| 230 | `at::_unique2` | - | 功能缺失 | 头文件: `ATen/ops/_unique2.h` |
| 231 | `at::_unpack_dual` | - | 功能缺失 | 头文件: `ATen/ops/_unpack_dual.h` |
| 232 | `at::_unsafe_index` | - | 功能缺失 | 头文件: `ATen/ops/_unsafe_index.h` |
| 233 | `at::_unsafe_index_put` | - | 功能缺失 | 头文件: `ATen/ops/_unsafe_index_put.h` |
| 234 | `at::_unsafe_masked_index` | - | 功能缺失 | 头文件: `ATen/ops/_unsafe_masked_index.h` |
| 235 | `at::_unsafe_masked_index_put_accumulate` | - | 功能缺失 | 头文件: `ATen/ops/_unsafe_masked_index_put_accumulate.h` |
| 236 | `at::_unsafe_view` | - | 功能缺失 | 头文件: `ATen/ops/_unsafe_view.h` |
| 237 | `at::_upsample_bicubic2d_aa` | - | 功能缺失 | 头文件: `ATen/ops/_upsample_bicubic2d_aa.h` |
| 238 | `at::_upsample_bilinear2d_aa` | - | 功能缺失 | 头文件: `ATen/ops/_upsample_bilinear2d_aa.h` |
| 239 | `at::_upsample_nearest_exact1d` | - | 功能缺失 | 头文件: `ATen/ops/_upsample_nearest_exact1d.h` |
| 240 | `at::_upsample_nearest_exact2d` | - | 功能缺失 | 头文件: `ATen/ops/_upsample_nearest_exact2d.h` |
| 241 | `at::_upsample_nearest_exact3d` | - | 功能缺失 | 头文件: `ATen/ops/_upsample_nearest_exact3d.h` |
| 242 | `at::_use_cudnn_ctc_loss` | - | 功能缺失 | 头文件: `ATen/ops/_use_cudnn_ctc_loss.h` |
| 243 | `at::_use_cudnn_rnn_flatten_weight` | - | 功能缺失 | 头文件: `ATen/ops/_use_cudnn_rnn_flatten_weight.h` |
| 244 | `at::_values_copy` | - | 功能缺失 | 头文件: `ATen/ops/_values_copy.h` |
| 245 | `at::_version` | - | 功能缺失 | 头文件: `ATen/ops/_version.h` |
| 246 | `at::_weight_int4pack_mm` | - | 功能缺失 | 头文件: `ATen/ops/_weight_int4pack_mm.h` |
| 247 | `at::_weight_int4pack_mm_with_scales_and_zeros` | - | 功能缺失 | 头文件: `ATen/ops/_weight_int4pack_mm_with_scales_and_zeros.h` |
| 248 | `at::_weight_int8pack_mm` | - | 功能缺失 | 头文件: `ATen/ops/_weight_int8pack_mm.h` |
| 249 | `at::_weight_norm` | - | 功能缺失 | 头文件: `ATen/ops/_weight_norm.h` |
| 250 | `at::_weight_norm_interface` | - | 功能缺失 | 头文件: `ATen/ops/_weight_norm_interface.h` |
| 251 | `at::_wrapped_linear_prepack` | - | 功能缺失 | 头文件: `ATen/ops/_wrapped_linear_prepack.h` |
| 252 | `at::absolute` | - | 功能缺失 | 头文件: `ATen/ops/absolute.h` |
| 253 | `at::adaptive_avg_pool1d` | - | 功能缺失 | 头文件: `ATen/ops/adaptive_avg_pool1d.h` |
| 254 | `at::adaptive_avg_pool2d` | - | 功能缺失 | 头文件: `ATen/ops/adaptive_avg_pool2d.h` |
| 255 | `at::adaptive_avg_pool3d` | - | 功能缺失 | 头文件: `ATen/ops/adaptive_avg_pool3d.h` |
| 256 | `at::adaptive_max_pool1d` | - | 功能缺失 | 头文件: `ATen/ops/adaptive_max_pool1d.h` |
| 257 | `at::adaptive_max_pool2d` | - | 功能缺失 | 头文件: `ATen/ops/adaptive_max_pool2d.h` |
| 258 | `at::adaptive_max_pool3d` | - | 功能缺失 | 头文件: `ATen/ops/adaptive_max_pool3d.h` |
| 259 | `at::addbmm` | - | 功能缺失 | 头文件: `ATen/ops/addbmm.h` |
| 260 | `at::addcdiv` | - | 功能缺失 | 头文件: `ATen/ops/addcdiv.h` |
| 261 | `at::addcmul` | - | 功能缺失 | 头文件: `ATen/ops/addcmul.h` |
| 262 | `at::addmv` | - | 功能缺失 | 头文件: `ATen/ops/addmv.h` |
| 263 | `at::addr` | - | 功能缺失 | 头文件: `ATen/ops/addr.h` |
| 264 | `at::adjoint` | - | 功能缺失 | 头文件: `ATen/ops/adjoint.h` |
| 265 | `at::affine_grid_generator` | - | 功能缺失 | 头文件: `ATen/ops/affine_grid_generator.h` |
| 266 | `at::alias` | - | 功能缺失 | 头文件: `ATen/ops/alias.h` |
| 267 | `at::alias_copy` | - | 功能缺失 | 头文件: `ATen/ops/alias_copy.h` |
| 268 | `at::align_as` | - | 功能缺失 | 头文件: `ATen/ops/align_as.h` |
| 269 | `at::align_tensors` | - | 功能缺失 | 头文件: `ATen/ops/align_tensors.h` |
| 270 | `at::align_to` | - | 功能缺失 | 头文件: `ATen/ops/align_to.h` |
| 271 | `at::alpha_dropout` | - | 功能缺失 | 头文件: `ATen/ops/alpha_dropout.h` |
| 272 | `at::and` | - | 功能缺失 | 头文件: `ATen/ops/and.h` |
| 273 | `at::arccos` | - | 功能缺失 | 头文件: `ATen/ops/arccos.h` |
| 274 | `at::arccosh` | - | 功能缺失 | 头文件: `ATen/ops/arccosh.h` |
| 275 | `at::arcsin` | - | 功能缺失 | 头文件: `ATen/ops/arcsin.h` |
| 276 | `at::arcsinh` | - | 功能缺失 | 头文件: `ATen/ops/arcsinh.h` |
| 277 | `at::arctan` | - | 功能缺失 | 头文件: `ATen/ops/arctan.h` |
| 278 | `at::arctan2` | - | 功能缺失 | 头文件: `ATen/ops/arctan2.h` |
| 279 | `at::arctanh` | - | 功能缺失 | 头文件: `ATen/ops/arctanh.h` |
| 280 | `at::argwhere` | - | 功能缺失 | 头文件: `ATen/ops/argwhere.h` |
| 281 | `at::as_strided_copy` | - | 功能缺失 | 头文件: `ATen/ops/as_strided_copy.h` |
| 282 | `at::as_strided_scatter` | - | 功能缺失 | 头文件: `ATen/ops/as_strided_scatter.h` |
| 283 | `at::atleast_1d` | - | 功能缺失 | 头文件: `ATen/ops/atleast_1d.h` |
| 284 | `at::atleast_2d` | - | 功能缺失 | 头文件: `ATen/ops/atleast_2d.h` |
| 285 | `at::atleast_3d` | - | 功能缺失 | 头文件: `ATen/ops/atleast_3d.h` |
| 286 | `at::avg_pool1d` | - | 功能缺失 | 头文件: `ATen/ops/avg_pool1d.h` |
| 287 | `at::avg_pool2d` | - | 功能缺失 | 头文件: `ATen/ops/avg_pool2d.h` |
| 288 | `at::avg_pool3d` | - | 功能缺失 | 头文件: `ATen/ops/avg_pool3d.h` |
| 289 | `at::bartlett_window` | - | 功能缺失 | 头文件: `ATen/ops/bartlett_window.h` |
| 290 | `at::batch_norm_elemt` | - | 功能缺失 | 头文件: `ATen/ops/batch_norm_elemt.h` |
| 291 | `at::batch_norm_gather_stats` | - | 功能缺失 | 头文件: `ATen/ops/batch_norm_gather_stats.h` |
| 292 | `at::batch_norm_gather_stats_with_counts` | - | 功能缺失 | 头文件: `ATen/ops/batch_norm_gather_stats_with_counts.h` |
| 293 | `at::batch_norm_stats` | - | 功能缺失 | 头文件: `ATen/ops/batch_norm_stats.h` |
| 294 | `at::batch_norm_update_stats` | - | 功能缺失 | 头文件: `ATen/ops/batch_norm_update_stats.h` |
| 295 | `at::binary_cross_entropy` | - | 功能缺失 | 头文件: `ATen/ops/binary_cross_entropy.h` |
| 296 | `at::binary_cross_entropy_with_logits` | - | 功能缺失 | 头文件: `ATen/ops/binary_cross_entropy_with_logits.h` |
| 297 | `at::blackman_window` | - | 功能缺失 | 头文件: `ATen/ops/blackman_window.h` |
| 298 | `at::block_diag` | - | 功能缺失 | 头文件: `ATen/ops/block_diag.h` |
| 299 | `at::broadcast_to` | - | 功能缺失 | 头文件: `ATen/ops/broadcast_to.h` |
| 300 | `at::bucketize` | - | 功能缺失 | 头文件: `ATen/ops/bucketize.h` |
| 301 | `at::can_cast` | - | 功能缺失 | 头文件: `ATen/ops/can_cast.h` |
| 302 | `at::cartesian_prod` | - | 功能缺失 | 头文件: `ATen/ops/cartesian_prod.h` |
| 303 | `at::cauchy` | - | 功能缺失 | 头文件: `ATen/ops/cauchy.h` |
| 304 | `at::ccol_indices` | - | 功能缺失 | 头文件: `ATen/ops/ccol_indices.h` |
| 305 | `at::ccol_indices_copy` | - | 功能缺失 | 头文件: `ATen/ops/ccol_indices_copy.h` |
| 306 | `at::cdist` | - | 功能缺失 | 头文件: `ATen/ops/cdist.h` |
| 307 | `at::chain_matmul` | - | 功能缺失 | 头文件: `ATen/ops/chain_matmul.h` |
| 308 | `at::chalf` | - | 功能缺失 | 头文件: `ATen/ops/chalf.h` |
| 309 | `at::cholesky_inverse` | - | 功能缺失 | 头文件: `ATen/ops/cholesky_inverse.h` |
| 310 | `at::choose_qparams_optimized` | - | 功能缺失 | 头文件: `ATen/ops/choose_qparams_optimized.h` |
| 311 | `at::clamp_max` | - | 功能缺失 | 头文件: `ATen/ops/clamp_max.h` |
| 312 | `at::clamp_min` | - | 功能缺失 | 头文件: `ATen/ops/clamp_min.h` |
| 313 | `at::clone` | - | 功能缺失 | 头文件: `ATen/ops/clone.h` |
| 314 | `at::col2im` | - | 功能缺失 | 头文件: `ATen/ops/col2im.h` |
| 315 | `at::col_indices` | - | 功能缺失 | 头文件: `ATen/ops/col_indices.h` |
| 316 | `at::col_indices_copy` | - | 功能缺失 | 头文件: `ATen/ops/col_indices_copy.h` |
| 317 | `at::column_stack` | - | 功能缺失 | 头文件: `ATen/ops/column_stack.h` |
| 318 | `at::combinations` | - | 功能缺失 | 头文件: `ATen/ops/combinations.h` |
| 319 | `at::concatenate` | - | 功能缺失 | 头文件: `ATen/ops/concatenate.h` |
| 320 | `at::conj_physical` | - | 功能缺失 | 头文件: `ATen/ops/conj_physical.h` |
| 321 | `at::constant_pad_nd` | - | 功能缺失 | 头文件: `ATen/ops/constant_pad_nd.h` |
| 322 | `at::contiguous` | - | 功能缺失 | 头文件: `ATen/ops/contiguous.h` |
| 323 | `at::conv1d` | - | 功能缺失 | 头文件: `ATen/ops/conv1d.h` |
| 324 | `at::conv_depthwise3d` | - | 功能缺失 | 头文件: `ATen/ops/conv_depthwise3d.h` |
| 325 | `at::conv_tbc` | - | 功能缺失 | 头文件: `ATen/ops/conv_tbc.h` |
| 326 | `at::conv_transpose1d` | - | 功能缺失 | 头文件: `ATen/ops/conv_transpose1d.h` |
| 327 | `at::convolution` | - | 功能缺失 | 头文件: `ATen/ops/convolution.h` |
| 328 | `at::convolution_overrideable` | - | 功能缺失 | 头文件: `ATen/ops/convolution_overrideable.h` |
| 329 | `at::copy` | - | 功能缺失 | 头文件: `ATen/ops/copy.h` |
| 330 | `at::corrcoef` | - | 功能缺失 | 头文件: `ATen/ops/corrcoef.h` |
| 331 | `at::cosine_embedding_loss` | - | 功能缺失 | 头文件: `ATen/ops/cosine_embedding_loss.h` |
| 332 | `at::cosine_similarity` | - | 功能缺失 | 头文件: `ATen/ops/cosine_similarity.h` |
| 333 | `at::count_nonzero` | - | 功能缺失 | 头文件: `ATen/ops/count_nonzero.h` |
| 334 | `at::cov` | - | 功能缺失 | 头文件: `ATen/ops/cov.h` |
| 335 | `at::cross_entropy_loss` | - | 功能缺失 | 头文件: `ATen/ops/cross_entropy_loss.h` |
| 336 | `at::crow_indices` | - | 功能缺失 | 头文件: `ATen/ops/crow_indices.h` |
| 337 | `at::crow_indices_copy` | - | 功能缺失 | 头文件: `ATen/ops/crow_indices_copy.h` |
| 338 | `at::ctc_loss` | - | 功能缺失 | 头文件: `ATen/ops/ctc_loss.h` |
| 339 | `at::cudnn_affine_grid_generator` | - | 功能缺失 | 头文件: `ATen/ops/cudnn_affine_grid_generator.h` |
| 340 | `at::cudnn_batch_norm` | - | 功能缺失 | 头文件: `ATen/ops/cudnn_batch_norm.h` |
| 341 | `at::cudnn_convolution` | - | 功能缺失 | 头文件: `ATen/ops/cudnn_convolution.h` |
| 342 | `at::cudnn_convolution_add_relu` | - | 功能缺失 | 头文件: `ATen/ops/cudnn_convolution_add_relu.h` |
| 343 | `at::cudnn_convolution_relu` | - | 功能缺失 | 头文件: `ATen/ops/cudnn_convolution_relu.h` |
| 344 | `at::cudnn_convolution_transpose` | - | 功能缺失 | 头文件: `ATen/ops/cudnn_convolution_transpose.h` |
| 345 | `at::cudnn_grid_sampler` | - | 功能缺失 | 头文件: `ATen/ops/cudnn_grid_sampler.h` |
| 346 | `at::cudnn_is_acceptable` | - | 功能缺失 | 头文件: `ATen/ops/cudnn_is_acceptable.h` |
| 347 | `at::cumulative_trapezoid` | - | 功能缺失 | 头文件: `ATen/ops/cumulative_trapezoid.h` |
| 348 | `at::deg2rad` | - | 功能缺失 | 头文件: `ATen/ops/deg2rad.h` |
| 349 | `at::dense_dim` | - | 功能缺失 | 头文件: `ATen/ops/dense_dim.h` |
| 350 | `at::dequantize` | - | 功能缺失 | 头文件: `ATen/ops/dequantize.h` |
| 351 | `at::detach_copy` | - | 功能缺失 | 头文件: `ATen/ops/detach_copy.h` |
| 352 | `at::diagflat` | - | 功能缺失 | 头文件: `ATen/ops/diagflat.h` |
| 353 | `at::diagonal_copy` | - | 功能缺失 | 头文件: `ATen/ops/diagonal_copy.h` |
| 354 | `at::diagonal_scatter` | - | 功能缺失 | 头文件: `ATen/ops/diagonal_scatter.h` |
| 355 | `at::diff` | - | 功能缺失 | 头文件: `ATen/ops/diff.h` |
| 356 | `at::div` | - | 功能缺失 | 头文件: `ATen/ops/div.h` |
| 357 | `at::dstack` | - | 功能缺失 | 头文件: `ATen/ops/dstack.h` |
| 358 | `at::einsum` | - | 功能缺失 | 头文件: `ATen/ops/einsum.h` |
| 359 | `at::embedding_bag` | - | 功能缺失 | 头文件: `ATen/ops/embedding_bag.h` |
| 360 | `at::embedding_renorm` | - | 功能缺失 | 头文件: `ATen/ops/embedding_renorm.h` |
| 361 | `at::empty_permuted` | - | 功能缺失 | 头文件: `ATen/ops/empty_permuted.h` |
| 362 | `at::eq` | - | 功能缺失 | 头文件: `ATen/ops/eq.h` |
| 363 | `at::erfc` | - | 功能缺失 | 头文件: `ATen/ops/erfc.h` |
| 364 | `at::exp2` | - | 功能缺失 | 头文件: `ATen/ops/exp2.h` |
| 365 | `at::expand_copy` | - | 功能缺失 | 头文件: `ATen/ops/expand_copy.h` |
| 366 | `at::exponential` | - | 功能缺失 | 头文件: `ATen/ops/exponential.h` |
| 367 | `at::fake_quantize_per_channel_affine` | - | 功能缺失 | 头文件: `ATen/ops/fake_quantize_per_channel_affine.h` |
| 368 | `at::fake_quantize_per_channel_affine_cachemask` | - | 功能缺失 | 头文件: `ATen/ops/fake_quantize_per_channel_affine_cachemask.h` |
| 369 | `at::fake_quantize_per_tensor_affine` | - | 功能缺失 | 头文件: `ATen/ops/fake_quantize_per_tensor_affine.h` |
| 370 | `at::fake_quantize_per_tensor_affine_cachemask` | - | 功能缺失 | 头文件: `ATen/ops/fake_quantize_per_tensor_affine_cachemask.h` |
| 371 | `at::fbgemm_linear_fp16_weight` | - | 功能缺失 | 头文件: `ATen/ops/fbgemm_linear_fp16_weight.h` |
| 372 | `at::fbgemm_linear_fp16_weight_fp32_activation` | - | 功能缺失 | 头文件: `ATen/ops/fbgemm_linear_fp16_weight_fp32_activation.h` |
| 373 | `at::fbgemm_linear_int8_weight` | - | 功能缺失 | 头文件: `ATen/ops/fbgemm_linear_int8_weight.h` |
| 374 | `at::fbgemm_linear_int8_weight_fp32_activation` | - | 功能缺失 | 头文件: `ATen/ops/fbgemm_linear_int8_weight_fp32_activation.h` |
| 375 | `at::fbgemm_linear_quantize_weight` | - | 功能缺失 | 头文件: `ATen/ops/fbgemm_linear_quantize_weight.h` |
| 376 | `at::fbgemm_pack_gemm_matrix_fp16` | - | 功能缺失 | 头文件: `ATen/ops/fbgemm_pack_gemm_matrix_fp16.h` |
| 377 | `at::feature_alpha_dropout` | - | 功能缺失 | 头文件: `ATen/ops/feature_alpha_dropout.h` |
| 378 | `at::feature_dropout` | - | 功能缺失 | 头文件: `ATen/ops/feature_dropout.h` |
| 379 | `at::fft_fft` | - | 功能缺失 | 头文件: `ATen/ops/fft_fft.h` |
| 380 | `at::fft_fft2` | - | 功能缺失 | 头文件: `ATen/ops/fft_fft2.h` |
| 381 | `at::fft_fftfreq` | - | 功能缺失 | 头文件: `ATen/ops/fft_fftfreq.h` |
| 382 | `at::fft_fftn` | - | 功能缺失 | 头文件: `ATen/ops/fft_fftn.h` |
| 383 | `at::fft_fftshift` | - | 功能缺失 | 头文件: `ATen/ops/fft_fftshift.h` |
| 384 | `at::fft_hfft` | - | 功能缺失 | 头文件: `ATen/ops/fft_hfft.h` |
| 385 | `at::fft_hfft2` | - | 功能缺失 | 头文件: `ATen/ops/fft_hfft2.h` |
| 386 | `at::fft_hfftn` | - | 功能缺失 | 头文件: `ATen/ops/fft_hfftn.h` |
| 387 | `at::fft_ifft` | - | 功能缺失 | 头文件: `ATen/ops/fft_ifft.h` |
| 388 | `at::fft_ifft2` | - | 功能缺失 | 头文件: `ATen/ops/fft_ifft2.h` |
| 389 | `at::fft_ifftn` | - | 功能缺失 | 头文件: `ATen/ops/fft_ifftn.h` |
| 390 | `at::fft_ifftshift` | - | 功能缺失 | 头文件: `ATen/ops/fft_ifftshift.h` |
| 391 | `at::fft_ihfft` | - | 功能缺失 | 头文件: `ATen/ops/fft_ihfft.h` |
| 392 | `at::fft_ihfft2` | - | 功能缺失 | 头文件: `ATen/ops/fft_ihfft2.h` |
| 393 | `at::fft_ihfftn` | - | 功能缺失 | 头文件: `ATen/ops/fft_ihfftn.h` |
| 394 | `at::fft_irfft` | - | 功能缺失 | 头文件: `ATen/ops/fft_irfft.h` |
| 395 | `at::fft_irfft2` | - | 功能缺失 | 头文件: `ATen/ops/fft_irfft2.h` |
| 396 | `at::fft_irfftn` | - | 功能缺失 | 头文件: `ATen/ops/fft_irfftn.h` |
| 397 | `at::fft_rfft` | - | 功能缺失 | 头文件: `ATen/ops/fft_rfft.h` |
| 398 | `at::fft_rfft2` | - | 功能缺失 | 头文件: `ATen/ops/fft_rfft2.h` |
| 399 | `at::fft_rfftfreq` | - | 功能缺失 | 头文件: `ATen/ops/fft_rfftfreq.h` |
| 400 | `at::fft_rfftn` | - | 功能缺失 | 头文件: `ATen/ops/fft_rfftn.h` |
| 401 | `at::fix` | - | 功能缺失 | 头文件: `ATen/ops/fix.h` |
| 402 | `at::flatten_dense_tensors` | - | 功能缺失 | 头文件: `ATen/ops/flatten_dense_tensors.h` |
| 403 | `at::fliplr` | - | 功能缺失 | 头文件: `ATen/ops/fliplr.h` |
| 404 | `at::flipud` | - | 功能缺失 | 头文件: `ATen/ops/flipud.h` |
| 405 | `at::float_power` | - | 功能缺失 | 头文件: `ATen/ops/float_power.h` |
| 406 | `at::fmod` | - | 功能缺失 | 头文件: `ATen/ops/fmod.h` |
| 407 | `at::frac` | - | 功能缺失 | 头文件: `ATen/ops/frac.h` |
| 408 | `at::frexp` | - | 功能缺失 | 头文件: `ATen/ops/frexp.h` |
| 409 | `at::from_file` | - | 功能缺失 | 头文件: `ATen/ops/from_file.h` |
| 410 | `at::fused_moving_avg_obs_fake_quant` | - | 功能缺失 | 头文件: `ATen/ops/fused_moving_avg_obs_fake_quant.h` |
| 411 | `at::gcd` | - | 功能缺失 | 头文件: `ATen/ops/gcd.h` |
| 412 | `at::ge` | - | 功能缺失 | 头文件: `ATen/ops/ge.h` |
| 413 | `at::geometric` | - | 功能缺失 | 头文件: `ATen/ops/geometric.h` |
| 414 | `at::geqrf` | - | 功能缺失 | 头文件: `ATen/ops/geqrf.h` |
| 415 | `at::ger` | - | 功能缺失 | 头文件: `ATen/ops/ger.h` |
| 416 | `at::glu` | - | 功能缺失 | 头文件: `ATen/ops/glu.h` |
| 417 | `at::glu_jvp` | - | 功能缺失 | 头文件: `ATen/ops/glu_jvp.h` |
| 418 | `at::gradient` | - | 功能缺失 | 头文件: `ATen/ops/gradient.h` |
| 419 | `at::greater` | - | 功能缺失 | 头文件: `ATen/ops/greater.h` |
| 420 | `at::grid_sampler_2d` | - | 功能缺失 | 头文件: `ATen/ops/grid_sampler_2d.h` |
| 421 | `at::grid_sampler_3d` | - | 功能缺失 | 头文件: `ATen/ops/grid_sampler_3d.h` |
| 422 | `at::gru_cell` | - | 功能缺失 | 头文件: `ATen/ops/gru_cell.h` |
| 423 | `at::gt` | - | 功能缺失 | 头文件: `ATen/ops/gt.h` |
| 424 | `at::hamming_window` | - | 功能缺失 | 头文件: `ATen/ops/hamming_window.h` |
| 425 | `at::hann_window` | - | 功能缺失 | 头文件: `ATen/ops/hann_window.h` |
| 426 | `at::hash_tensor` | - | 功能缺失 | 头文件: `ATen/ops/hash_tensor.h` |
| 427 | `at::hinge_embedding_loss` | - | 功能缺失 | 头文件: `ATen/ops/hinge_embedding_loss.h` |
| 428 | `at::histc` | - | 功能缺失 | 头文件: `ATen/ops/histc.h` |
| 429 | `at::histogramdd` | - | 功能缺失 | 头文件: `ATen/ops/histogramdd.h` |
| 430 | `at::hspmm` | - | 功能缺失 | 头文件: `ATen/ops/hspmm.h` |
| 431 | `at::hstack` | - | 功能缺失 | 头文件: `ATen/ops/hstack.h` |
| 432 | `at::hypot` | - | 功能缺失 | 头文件: `ATen/ops/hypot.h` |
| 433 | `at::igamma` | - | 功能缺失 | 头文件: `ATen/ops/igamma.h` |
| 434 | `at::igammac` | - | 功能缺失 | 头文件: `ATen/ops/igammac.h` |
| 435 | `at::im2col` | - | 功能缺失 | 头文件: `ATen/ops/im2col.h` |
| 436 | `at::index_copy` | - | 功能缺失 | 头文件: `ATen/ops/index_copy.h` |
| 437 | `at::index_reduce` | - | 功能缺失 | 头文件: `ATen/ops/index_reduce.h` |
| 438 | `at::indices` | - | 功能缺失 | 头文件: `ATen/ops/indices.h` |
| 439 | `at::indices_copy` | - | 功能缺失 | 头文件: `ATen/ops/indices_copy.h` |
| 440 | `at::inner` | - | 功能缺失 | 头文件: `ATen/ops/inner.h` |
| 441 | `at::int_repr` | - | 功能缺失 | 头文件: `ATen/ops/int_repr.h` |
| 442 | `at::is_complex` | - | 功能缺失 | 头文件: `ATen/ops/is_complex.h` |
| 443 | `at::is_conj` | - | 功能缺失 | 头文件: `ATen/ops/is_conj.h` |
| 444 | `at::is_distributed` | - | 功能缺失 | 头文件: `ATen/ops/is_distributed.h` |
| 445 | `at::is_floating_point` | - | 功能缺失 | 头文件: `ATen/ops/is_floating_point.h` |
| 446 | `at::is_inference` | - | 功能缺失 | 头文件: `ATen/ops/is_inference.h` |
| 447 | `at::is_leaf` | - | 功能缺失 | 头文件: `ATen/ops/is_leaf.h` |
| 448 | `at::is_neg` | - | 功能缺失 | 头文件: `ATen/ops/is_neg.h` |
| 449 | `at::is_nonzero` | - | 功能缺失 | 头文件: `ATen/ops/is_nonzero.h` |
| 450 | `at::is_pinned` | - | 功能缺失 | 头文件: `ATen/ops/is_pinned.h` |
| 451 | `at::is_same_size` | - | 功能缺失 | 头文件: `ATen/ops/is_same_size.h` |
| 452 | `at::is_set_to` | - | 功能缺失 | 头文件: `ATen/ops/is_set_to.h` |
| 453 | `at::is_signed` | - | 功能缺失 | 头文件: `ATen/ops/is_signed.h` |
| 454 | `at::isin` | - | 功能缺失 | 头文件: `ATen/ops/isin.h` |
| 455 | `at::isneginf` | - | 功能缺失 | 头文件: `ATen/ops/isneginf.h` |
| 456 | `at::isposinf` | - | 功能缺失 | 头文件: `ATen/ops/isposinf.h` |
| 457 | `at::isreal` | - | 功能缺失 | 头文件: `ATen/ops/isreal.h` |
| 458 | `at::istft` | - | 功能缺失 | 头文件: `ATen/ops/istft.h` |
| 459 | `at::kaiser_window` | - | 功能缺失 | 头文件: `ATen/ops/kaiser_window.h` |
| 460 | `at::kl_div` | - | 功能缺失 | 头文件: `ATen/ops/kl_div.h` |
| 461 | `at::l1_loss` | - | 功能缺失 | 头文件: `ATen/ops/l1_loss.h` |
| 462 | `at::lcm` | - | 功能缺失 | 头文件: `ATen/ops/lcm.h` |
| 463 | `at::ldexp` | - | 功能缺失 | 头文件: `ATen/ops/ldexp.h` |
| 464 | `at::le` | - | 功能缺失 | 头文件: `ATen/ops/le.h` |
| 465 | `at::less` | - | 功能缺失 | 头文件: `ATen/ops/less.h` |
| 466 | `at::lift` | - | 功能缺失 | 头文件: `ATen/ops/lift.h` |
| 467 | `at::lift_fresh` | - | 功能缺失 | 头文件: `ATen/ops/lift_fresh.h` |
| 468 | `at::lift_fresh_copy` | - | 功能缺失 | 头文件: `ATen/ops/lift_fresh_copy.h` |
| 469 | `at::linalg_cholesky` | - | 功能缺失 | 头文件: `ATen/ops/linalg_cholesky.h` |
| 470 | `at::linalg_cholesky_ex` | - | 功能缺失 | 头文件: `ATen/ops/linalg_cholesky_ex.h` |
| 471 | `at::linalg_cond` | - | 功能缺失 | 头文件: `ATen/ops/linalg_cond.h` |
| 472 | `at::linalg_cross` | - | 功能缺失 | 头文件: `ATen/ops/linalg_cross.h` |
| 473 | `at::linalg_det` | - | 功能缺失 | 头文件: `ATen/ops/linalg_det.h` |
| 474 | `at::linalg_diagonal` | - | 功能缺失 | 头文件: `ATen/ops/linalg_diagonal.h` |
| 475 | `at::linalg_eig` | - | 功能缺失 | 头文件: `ATen/ops/linalg_eig.h` |
| 476 | `at::linalg_eigh` | - | 功能缺失 | 头文件: `ATen/ops/linalg_eigh.h` |
| 477 | `at::linalg_eigvals` | - | 功能缺失 | 头文件: `ATen/ops/linalg_eigvals.h` |
| 478 | `at::linalg_eigvalsh` | - | 功能缺失 | 头文件: `ATen/ops/linalg_eigvalsh.h` |
| 479 | `at::linalg_householder_product` | - | 功能缺失 | 头文件: `ATen/ops/linalg_householder_product.h` |
| 480 | `at::linalg_inv` | - | 功能缺失 | 头文件: `ATen/ops/linalg_inv.h` |
| 481 | `at::linalg_inv_ex` | - | 功能缺失 | 头文件: `ATen/ops/linalg_inv_ex.h` |
| 482 | `at::linalg_ldl_factor` | - | 功能缺失 | 头文件: `ATen/ops/linalg_ldl_factor.h` |
| 483 | `at::linalg_ldl_factor_ex` | - | 功能缺失 | 头文件: `ATen/ops/linalg_ldl_factor_ex.h` |
| 484 | `at::linalg_ldl_solve` | - | 功能缺失 | 头文件: `ATen/ops/linalg_ldl_solve.h` |
| 485 | `at::linalg_lstsq` | - | 功能缺失 | 头文件: `ATen/ops/linalg_lstsq.h` |
| 486 | `at::linalg_lu` | - | 功能缺失 | 头文件: `ATen/ops/linalg_lu.h` |
| 487 | `at::linalg_lu_factor` | - | 功能缺失 | 头文件: `ATen/ops/linalg_lu_factor.h` |
| 488 | `at::linalg_lu_factor_ex` | - | 功能缺失 | 头文件: `ATen/ops/linalg_lu_factor_ex.h` |
| 489 | `at::linalg_lu_solve` | - | 功能缺失 | 头文件: `ATen/ops/linalg_lu_solve.h` |
| 490 | `at::linalg_matmul` | - | 功能缺失 | 头文件: `ATen/ops/linalg_matmul.h` |
| 491 | `at::linalg_matrix_exp` | - | 功能缺失 | 头文件: `ATen/ops/linalg_matrix_exp.h` |
| 492 | `at::linalg_matrix_norm` | - | 功能缺失 | 头文件: `ATen/ops/linalg_matrix_norm.h` |
| 493 | `at::linalg_matrix_power` | - | 功能缺失 | 头文件: `ATen/ops/linalg_matrix_power.h` |
| 494 | `at::linalg_matrix_rank` | - | 功能缺失 | 头文件: `ATen/ops/linalg_matrix_rank.h` |
| 495 | `at::linalg_multi_dot` | - | 功能缺失 | 头文件: `ATen/ops/linalg_multi_dot.h` |
| 496 | `at::linalg_norm` | - | 功能缺失 | 头文件: `ATen/ops/linalg_norm.h` |
| 497 | `at::linalg_pinv` | - | 功能缺失 | 头文件: `ATen/ops/linalg_pinv.h` |
| 498 | `at::linalg_qr` | - | 功能缺失 | 头文件: `ATen/ops/linalg_qr.h` |
| 499 | `at::linalg_slogdet` | - | 功能缺失 | 头文件: `ATen/ops/linalg_slogdet.h` |
| 500 | `at::linalg_solve` | - | 功能缺失 | 头文件: `ATen/ops/linalg_solve.h` |
| 501 | `at::linalg_solve_ex` | - | 功能缺失 | 头文件: `ATen/ops/linalg_solve_ex.h` |
| 502 | `at::linalg_solve_triangular` | - | 功能缺失 | 头文件: `ATen/ops/linalg_solve_triangular.h` |
| 503 | `at::linalg_svd` | - | 功能缺失 | 头文件: `ATen/ops/linalg_svd.h` |
| 504 | `at::linalg_svdvals` | - | 功能缺失 | 头文件: `ATen/ops/linalg_svdvals.h` |
| 505 | `at::linalg_tensorinv` | - | 功能缺失 | 头文件: `ATen/ops/linalg_tensorinv.h` |
| 506 | `at::linalg_tensorsolve` | - | 功能缺失 | 头文件: `ATen/ops/linalg_tensorsolve.h` |
| 507 | `at::linalg_vander` | - | 功能缺失 | 头文件: `ATen/ops/linalg_vander.h` |
| 508 | `at::linalg_vecdot` | - | 功能缺失 | 头文件: `ATen/ops/linalg_vecdot.h` |
| 509 | `at::linalg_vector_norm` | - | 功能缺失 | 头文件: `ATen/ops/linalg_vector_norm.h` |
| 510 | `at::linear` | - | 功能缺失 | 头文件: `ATen/ops/linear.h` |
| 511 | `at::log_normal` | - | 功能缺失 | 头文件: `ATen/ops/log_normal.h` |
| 512 | `at::logaddexp` | - | 功能缺失 | 头文件: `ATen/ops/logaddexp.h` |
| 513 | `at::logaddexp2` | - | 功能缺失 | 头文件: `ATen/ops/logaddexp2.h` |
| 514 | `at::logdet` | - | 功能缺失 | 头文件: `ATen/ops/logdet.h` |
| 515 | `at::lshift` | - | 功能缺失 | 头文件: `ATen/ops/lshift.h` |
| 516 | `at::lstm_cell` | - | 功能缺失 | 头文件: `ATen/ops/lstm_cell.h` |
| 517 | `at::lt` | - | 功能缺失 | 头文件: `ATen/ops/lt.h` |
| 518 | `at::mH` | - | 功能缺失 | 头文件: `ATen/ops/mH.h` |
| 519 | `at::mT` | - | 功能缺失 | 头文件: `ATen/ops/mT.h` |
| 520 | `at::margin_ranking_loss` | - | 功能缺失 | 头文件: `ATen/ops/margin_ranking_loss.h` |
| 521 | `at::matrix_H` | - | 功能缺失 | 头文件: `ATen/ops/matrix_H.h` |
| 522 | `at::matrix_exp` | - | 功能缺失 | 头文件: `ATen/ops/matrix_exp.h` |
| 523 | `at::max_pool1d` | - | 功能缺失 | 头文件: `ATen/ops/max_pool1d.h` |
| 524 | `at::max_pool1d_with_indices` | - | 功能缺失 | 头文件: `ATen/ops/max_pool1d_with_indices.h` |
| 525 | `at::max_pool2d` | - | 功能缺失 | 头文件: `ATen/ops/max_pool2d.h` |
| 526 | `at::max_pool3d` | - | 功能缺失 | 头文件: `ATen/ops/max_pool3d.h` |
| 527 | `at::max_unpool2d` | - | 功能缺失 | 头文件: `ATen/ops/max_unpool2d.h` |
| 528 | `at::max_unpool3d` | - | 功能缺失 | 头文件: `ATen/ops/max_unpool3d.h` |
| 529 | `at::miopen_batch_norm` | - | 功能缺失 | 头文件: `ATen/ops/miopen_batch_norm.h` |
| 530 | `at::miopen_convolution` | - | 功能缺失 | 头文件: `ATen/ops/miopen_convolution.h` |
| 531 | `at::miopen_convolution_add_relu` | - | 功能缺失 | 头文件: `ATen/ops/miopen_convolution_add_relu.h` |
| 532 | `at::miopen_convolution_relu` | - | 功能缺失 | 头文件: `ATen/ops/miopen_convolution_relu.h` |
| 533 | `at::miopen_convolution_transpose` | - | 功能缺失 | 头文件: `ATen/ops/miopen_convolution_transpose.h` |
| 534 | `at::miopen_depthwise_convolution` | - | 功能缺失 | 头文件: `ATen/ops/miopen_depthwise_convolution.h` |
| 535 | `at::miopen_rnn` | - | 功能缺失 | 头文件: `ATen/ops/miopen_rnn.h` |
| 536 | `at::mkldnn_adaptive_avg_pool2d` | - | 功能缺失 | 头文件: `ATen/ops/mkldnn_adaptive_avg_pool2d.h` |
| 537 | `at::mkldnn_convolution` | - | 功能缺失 | 头文件: `ATen/ops/mkldnn_convolution.h` |
| 538 | `at::mkldnn_linear` | - | 功能缺失 | 头文件: `ATen/ops/mkldnn_linear.h` |
| 539 | `at::mkldnn_max_pool2d` | - | 功能缺失 | 头文件: `ATen/ops/mkldnn_max_pool2d.h` |
| 540 | `at::mkldnn_max_pool3d` | - | 功能缺失 | 头文件: `ATen/ops/mkldnn_max_pool3d.h` |
| 541 | `at::mkldnn_reorder_conv2d_weight` | - | 功能缺失 | 头文件: `ATen/ops/mkldnn_reorder_conv2d_weight.h` |
| 542 | `at::mkldnn_reorder_conv3d_weight` | - | 功能缺失 | 头文件: `ATen/ops/mkldnn_reorder_conv3d_weight.h` |
| 543 | `at::mkldnn_rnn_layer` | - | 功能缺失 | 头文件: `ATen/ops/mkldnn_rnn_layer.h` |
| 544 | `at::mm` | - | 功能缺失 | 头文件: `ATen/ops/mm.h` |
| 545 | `at::moveaxis` | - | 功能缺失 | 头文件: `ATen/ops/moveaxis.h` |
| 546 | `at::movedim` | - | 功能缺失 | 头文件: `ATen/ops/movedim.h` |
| 547 | `at::mse_loss` | - | 功能缺失 | 头文件: `ATen/ops/mse_loss.h` |
| 548 | `at::msort` | - | 功能缺失 | 头文件: `ATen/ops/msort.h` |
| 549 | `at::mul` | - | 功能缺失 | 头文件: `ATen/ops/mul.h` |
| 550 | `at::multi_margin_loss` | - | 功能缺失 | 头文件: `ATen/ops/multi_margin_loss.h` |
| 551 | `at::multilabel_margin_loss` | - | 功能缺失 | 头文件: `ATen/ops/multilabel_margin_loss.h` |
| 552 | `at::mvlgamma` | - | 功能缺失 | 头文件: `ATen/ops/mvlgamma.h` |
| 553 | `at::nan_to_num` | - | 功能缺失 | 头文件: `ATen/ops/nan_to_num.h` |
| 554 | `at::nanmean` | - | 功能缺失 | 头文件: `ATen/ops/nanmean.h` |
| 555 | `at::nanquantile` | - | 功能缺失 | 头文件: `ATen/ops/nanquantile.h` |
| 556 | `at::native_batch_norm` | - | 功能缺失 | 头文件: `ATen/ops/native_batch_norm.h` |
| 557 | `at::native_channel_shuffle` | - | 功能缺失 | 头文件: `ATen/ops/native_channel_shuffle.h` |
| 558 | `at::native_dropout` | - | 功能缺失 | 头文件: `ATen/ops/native_dropout.h` |
| 559 | `at::native_group_norm` | - | 功能缺失 | 头文件: `ATen/ops/native_group_norm.h` |
| 560 | `at::native_layer_norm` | - | 功能缺失 | 头文件: `ATen/ops/native_layer_norm.h` |
| 561 | `at::native_norm` | - | 功能缺失 | 头文件: `ATen/ops/native_norm.h` |
| 562 | `at::ne` | - | 功能缺失 | 头文件: `ATen/ops/ne.h` |
| 563 | `at::neg` | - | 功能缺失 | 头文件: `ATen/ops/neg.h` |
| 564 | `at::negative` | - | 功能缺失 | 头文件: `ATen/ops/negative.h` |
| 565 | `at::nested_to_padded_tensor` | - | 功能缺失 | 头文件: `ATen/ops/nested_to_padded_tensor.h` |
| 566 | `at::new_empty_strided` | - | 功能缺失 | 头文件: `ATen/ops/new_empty_strided.h` |
| 567 | `at::nll_loss2d` | - | 功能缺失 | 头文件: `ATen/ops/nll_loss2d.h` |
| 568 | `at::nll_loss_nd` | - | 功能缺失 | 头文件: `ATen/ops/nll_loss_nd.h` |
| 569 | `at::nonzero_numpy` | - | 功能缺失 | 头文件: `ATen/ops/nonzero_numpy.h` |
| 570 | `at::nonzero_static` | - | 功能缺失 | 头文件: `ATen/ops/nonzero_static.h` |
| 571 | `at::norm_except_dim` | - | 功能缺失 | 头文件: `ATen/ops/norm_except_dim.h` |
| 572 | `at::normal` | - | 功能缺失 | 头文件: `ATen/ops/normal.h` |
| 573 | `at::nuclear_norm` | - | 功能缺失 | 头文件: `ATen/ops/nuclear_norm.h` |
| 574 | `at::numpy_T` | - | 功能缺失 | 头文件: `ATen/ops/numpy_T.h` |
| 575 | `at::or` | - | 功能缺失 | 头文件: `ATen/ops/or.h` |
| 576 | `at::orgqr` | - | 功能缺失 | 头文件: `ATen/ops/orgqr.h` |
| 577 | `at::ormqr` | - | 功能缺失 | 头文件: `ATen/ops/ormqr.h` |
| 578 | `at::outer` | - | 功能缺失 | 头文件: `ATen/ops/outer.h` |
| 579 | `at::output_nr` | - | 功能缺失 | 头文件: `ATen/ops/output_nr.h` |
| 580 | `at::pad_sequence` | - | 功能缺失 | 头文件: `ATen/ops/pad_sequence.h` |
| 581 | `at::pairwise_distance` | - | 功能缺失 | 头文件: `ATen/ops/pairwise_distance.h` |
| 582 | `at::pdist` | - | 功能缺失 | 头文件: `ATen/ops/pdist.h` |
| 583 | `at::permute_copy` | - | 功能缺失 | 头文件: `ATen/ops/permute_copy.h` |
| 584 | `at::pin_memory` | - | 功能缺失 | 头文件: `ATen/ops/pin_memory.h` |
| 585 | `at::pinverse` | - | 功能缺失 | 头文件: `ATen/ops/pinverse.h` |
| 586 | `at::poisson_nll_loss` | - | 功能缺失 | 头文件: `ATen/ops/poisson_nll_loss.h` |
| 587 | `at::polar` | - | 功能缺失 | 头文件: `ATen/ops/polar.h` |
| 588 | `at::positive` | - | 功能缺失 | 头文件: `ATen/ops/positive.h` |
| 589 | `at::promote_types` | - | 功能缺失 | 头文件: `ATen/ops/promote_types.h` |
| 590 | `at::put` | - | 功能缺失 | 头文件: `ATen/ops/put.h` |
| 591 | `at::q_per_channel_axis` | - | 功能缺失 | 头文件: `ATen/ops/q_per_channel_axis.h` |
| 592 | `at::q_per_channel_scales` | - | 功能缺失 | 头文件: `ATen/ops/q_per_channel_scales.h` |
| 593 | `at::q_per_channel_zero_points` | - | 功能缺失 | 头文件: `ATen/ops/q_per_channel_zero_points.h` |
| 594 | `at::q_scale` | - | 功能缺失 | 头文件: `ATen/ops/q_scale.h` |
| 595 | `at::q_zero_point` | - | 功能缺失 | 头文件: `ATen/ops/q_zero_point.h` |
| 596 | `at::qscheme` | - | 功能缺失 | 头文件: `ATen/ops/qscheme.h` |
| 597 | `at::quantile` | - | 功能缺失 | 头文件: `ATen/ops/quantile.h` |
| 598 | `at::quantize_per_channel` | - | 功能缺失 | 头文件: `ATen/ops/quantize_per_channel.h` |
| 599 | `at::quantize_per_tensor` | - | 功能缺失 | 头文件: `ATen/ops/quantize_per_tensor.h` |
| 600 | `at::quantize_per_tensor_dynamic` | - | 功能缺失 | 头文件: `ATen/ops/quantize_per_tensor_dynamic.h` |
| 601 | `at::quantized_batch_norm` | - | 功能缺失 | 头文件: `ATen/ops/quantized_batch_norm.h` |
| 602 | `at::quantized_gru_cell` | - | 功能缺失 | 头文件: `ATen/ops/quantized_gru_cell.h` |
| 603 | `at::quantized_lstm_cell` | - | 功能缺失 | 头文件: `ATen/ops/quantized_lstm_cell.h` |
| 604 | `at::quantized_max_pool1d` | - | 功能缺失 | 头文件: `ATen/ops/quantized_max_pool1d.h` |
| 605 | `at::quantized_max_pool2d` | - | 功能缺失 | 头文件: `ATen/ops/quantized_max_pool2d.h` |
| 606 | `at::quantized_max_pool3d` | - | 功能缺失 | 头文件: `ATen/ops/quantized_max_pool3d.h` |
| 607 | `at::quantized_rnn_relu_cell` | - | 功能缺失 | 头文件: `ATen/ops/quantized_rnn_relu_cell.h` |
| 608 | `at::quantized_rnn_tanh_cell` | - | 功能缺失 | 头文件: `ATen/ops/quantized_rnn_tanh_cell.h` |
| 609 | `at::rad2deg` | - | 功能缺失 | 头文件: `ATen/ops/rad2deg.h` |
| 610 | `at::rand` | - | 功能缺失 | 头文件: `ATen/ops/rand.h` |
| 611 | `at::rand_like` | - | 功能缺失 | 头文件: `ATen/ops/rand_like.h` |
| 612 | `at::randint_like` | - | 功能缺失 | 头文件: `ATen/ops/randint_like.h` |
| 613 | `at::randn` | - | 功能缺失 | 头文件: `ATen/ops/randn.h` |
| 614 | `at::randn_like` | - | 功能缺失 | 头文件: `ATen/ops/randn_like.h` |
| 615 | `at::ravel` | - | 功能缺失 | 头文件: `ATen/ops/ravel.h` |
| 616 | `at::refine_names` | - | 功能缺失 | 头文件: `ATen/ops/refine_names.h` |
| 617 | `at::reflection_pad1d` | - | 功能缺失 | 头文件: `ATen/ops/reflection_pad1d.h` |
| 618 | `at::reflection_pad2d` | - | 功能缺失 | 头文件: `ATen/ops/reflection_pad2d.h` |
| 619 | `at::reflection_pad3d` | - | 功能缺失 | 头文件: `ATen/ops/reflection_pad3d.h` |
| 620 | `at::repeat` | - | 功能缺失 | 头文件: `ATen/ops/repeat.h` |
| 621 | `at::replication_pad1d` | - | 功能缺失 | 头文件: `ATen/ops/replication_pad1d.h` |
| 622 | `at::replication_pad2d` | - | 功能缺失 | 头文件: `ATen/ops/replication_pad2d.h` |
| 623 | `at::replication_pad3d` | - | 功能缺失 | 头文件: `ATen/ops/replication_pad3d.h` |
| 624 | `at::requires_grad` | - | 功能缺失 | 头文件: `ATen/ops/requires_grad.h` |
| 625 | `at::reshape_as` | - | 功能缺失 | 头文件: `ATen/ops/reshape_as.h` |
| 626 | `at::resize_as` | - | 功能缺失 | 头文件: `ATen/ops/resize_as.h` |
| 627 | `at::resolve_conj` | - | 功能缺失 | 头文件: `ATen/ops/resolve_conj.h` |
| 628 | `at::resolve_neg` | - | 功能缺失 | 头文件: `ATen/ops/resolve_neg.h` |
| 629 | `at::result_type` | - | 功能缺失 | 头文件: `ATen/ops/result_type.h` |
| 630 | `at::retain_grad` | - | 功能缺失 | 头文件: `ATen/ops/retain_grad.h` |
| 631 | `at::retains_grad` | - | 功能缺失 | 头文件: `ATen/ops/retains_grad.h` |
| 632 | `at::rnn_relu` | - | 功能缺失 | 头文件: `ATen/ops/rnn_relu.h` |
| 633 | `at::rnn_relu_cell` | - | 功能缺失 | 头文件: `ATen/ops/rnn_relu_cell.h` |
| 634 | `at::rnn_tanh` | - | 功能缺失 | 头文件: `ATen/ops/rnn_tanh.h` |
| 635 | `at::rnn_tanh_cell` | - | 功能缺失 | 头文件: `ATen/ops/rnn_tanh_cell.h` |
| 636 | `at::rot90` | - | 功能缺失 | 头文件: `ATen/ops/rot90.h` |
| 637 | `at::row_indices` | - | 功能缺失 | 头文件: `ATen/ops/row_indices.h` |
| 638 | `at::row_indices_copy` | - | 功能缺失 | 头文件: `ATen/ops/row_indices_copy.h` |
| 639 | `at::row_stack` | - | 功能缺失 | 头文件: `ATen/ops/row_stack.h` |
| 640 | `at::rrelu_with_noise` | - | 功能缺失 | 头文件: `ATen/ops/rrelu_with_noise.h` |
| 641 | `at::rshift` | - | 功能缺失 | 头文件: `ATen/ops/rshift.h` |
| 642 | `at::rsub` | - | 功能缺失 | 头文件: `ATen/ops/rsub.h` |
| 643 | `at::scalar_tensor` | - | 功能缺失 | 头文件: `ATen/ops/scalar_tensor.h` |
| 644 | `at::scaled_dot_product_attention` | - | 功能缺失 | 头文件: `ATen/ops/scaled_dot_product_attention.h` |
| 645 | `at::scatter_add` | - | 功能缺失 | 头文件: `ATen/ops/scatter_add.h` |
| 646 | `at::scatter_reduce` | - | 功能缺失 | 头文件: `ATen/ops/scatter_reduce.h` |
| 647 | `at::segment_reduce` | - | 功能缺失 | 头文件: `ATen/ops/segment_reduce.h` |
| 648 | `at::select_copy` | - | 功能缺失 | 头文件: `ATen/ops/select_copy.h` |
| 649 | `at::select_scatter` | - | 功能缺失 | 头文件: `ATen/ops/select_scatter.h` |
| 650 | `at::set_data` | - | 功能缺失 | 头文件: `ATen/ops/set_data.h` |
| 651 | `at::sgn` | - | 功能缺失 | 头文件: `ATen/ops/sgn.h` |
| 652 | `at::signbit` | - | 功能缺失 | 头文件: `ATen/ops/signbit.h` |
| 653 | `at::sinc` | - | 功能缺失 | 头文件: `ATen/ops/sinc.h` |
| 654 | `at::size` | - | 功能缺失 | 头文件: `ATen/ops/size.h` |
| 655 | `at::slice_copy` | - | 功能缺失 | 头文件: `ATen/ops/slice_copy.h` |
| 656 | `at::slice_inverse` | - | 功能缺失 | 头文件: `ATen/ops/slice_inverse.h` |
| 657 | `at::slice_scatter` | - | 功能缺失 | 头文件: `ATen/ops/slice_scatter.h` |
| 658 | `at::slow_conv3d` | - | 功能缺失 | 头文件: `ATen/ops/slow_conv3d.h` |
| 659 | `at::slow_conv_dilated2d` | - | 功能缺失 | 头文件: `ATen/ops/slow_conv_dilated2d.h` |
| 660 | `at::slow_conv_dilated3d` | - | 功能缺失 | 头文件: `ATen/ops/slow_conv_dilated3d.h` |
| 661 | `at::slow_conv_transpose2d` | - | 功能缺失 | 头文件: `ATen/ops/slow_conv_transpose2d.h` |
| 662 | `at::slow_conv_transpose3d` | - | 功能缺失 | 头文件: `ATen/ops/slow_conv_transpose3d.h` |
| 663 | `at::smm` | - | 功能缺失 | 头文件: `ATen/ops/smm.h` |
| 664 | `at::smooth_l1_loss` | - | 功能缺失 | 头文件: `ATen/ops/smooth_l1_loss.h` |
| 665 | `at::soft_margin_loss` | - | 功能缺失 | 头文件: `ATen/ops/soft_margin_loss.h` |
| 666 | `at::sort` | - | 功能缺失 | 头文件: `ATen/ops/sort.h` |
| 667 | `at::sparse_bsc_tensor` | - | 功能缺失 | 头文件: `ATen/ops/sparse_bsc_tensor.h` |
| 668 | `at::sparse_bsr_tensor` | - | 功能缺失 | 头文件: `ATen/ops/sparse_bsr_tensor.h` |
| 669 | `at::sparse_compressed_tensor` | - | 功能缺失 | 头文件: `ATen/ops/sparse_compressed_tensor.h` |
| 670 | `at::sparse_csc_tensor` | - | 功能缺失 | 头文件: `ATen/ops/sparse_csc_tensor.h` |
| 671 | `at::sparse_dim` | - | 功能缺失 | 头文件: `ATen/ops/sparse_dim.h` |
| 672 | `at::sparse_mask` | - | 功能缺失 | 头文件: `ATen/ops/sparse_mask.h` |
| 673 | `at::sparse_resize` | - | 功能缺失 | 头文件: `ATen/ops/sparse_resize.h` |
| 674 | `at::sparse_resize_and_clear` | - | 功能缺失 | 头文件: `ATen/ops/sparse_resize_and_clear.h` |
| 675 | `at::sparse_sampled_addmm` | - | 功能缺失 | 头文件: `ATen/ops/sparse_sampled_addmm.h` |
| 676 | `at::special_airy_ai` | - | 功能缺失 | 头文件: `ATen/ops/special_airy_ai.h` |
| 677 | `at::special_bessel_j0` | - | 功能缺失 | 头文件: `ATen/ops/special_bessel_j0.h` |
| 678 | `at::special_bessel_j1` | - | 功能缺失 | 头文件: `ATen/ops/special_bessel_j1.h` |
| 679 | `at::special_bessel_y0` | - | 功能缺失 | 头文件: `ATen/ops/special_bessel_y0.h` |
| 680 | `at::special_bessel_y1` | - | 功能缺失 | 头文件: `ATen/ops/special_bessel_y1.h` |
| 681 | `at::special_chebyshev_polynomial_t` | - | 功能缺失 | 头文件: `ATen/ops/special_chebyshev_polynomial_t.h` |
| 682 | `at::special_chebyshev_polynomial_u` | - | 功能缺失 | 头文件: `ATen/ops/special_chebyshev_polynomial_u.h` |
| 683 | `at::special_chebyshev_polynomial_v` | - | 功能缺失 | 头文件: `ATen/ops/special_chebyshev_polynomial_v.h` |
| 684 | `at::special_chebyshev_polynomial_w` | - | 功能缺失 | 头文件: `ATen/ops/special_chebyshev_polynomial_w.h` |
| 685 | `at::special_digamma` | - | 功能缺失 | 头文件: `ATen/ops/special_digamma.h` |
| 686 | `at::special_entr` | - | 功能缺失 | 头文件: `ATen/ops/special_entr.h` |
| 687 | `at::special_erf` | - | 功能缺失 | 头文件: `ATen/ops/special_erf.h` |
| 688 | `at::special_erfc` | - | 功能缺失 | 头文件: `ATen/ops/special_erfc.h` |
| 689 | `at::special_erfcx` | - | 功能缺失 | 头文件: `ATen/ops/special_erfcx.h` |
| 690 | `at::special_erfinv` | - | 功能缺失 | 头文件: `ATen/ops/special_erfinv.h` |
| 691 | `at::special_exp2` | - | 功能缺失 | 头文件: `ATen/ops/special_exp2.h` |
| 692 | `at::special_expit` | - | 功能缺失 | 头文件: `ATen/ops/special_expit.h` |
| 693 | `at::special_expm1` | - | 功能缺失 | 头文件: `ATen/ops/special_expm1.h` |
| 694 | `at::special_gammainc` | - | 功能缺失 | 头文件: `ATen/ops/special_gammainc.h` |
| 695 | `at::special_gammaincc` | - | 功能缺失 | 头文件: `ATen/ops/special_gammaincc.h` |
| 696 | `at::special_gammaln` | - | 功能缺失 | 头文件: `ATen/ops/special_gammaln.h` |
| 697 | `at::special_hermite_polynomial_h` | - | 功能缺失 | 头文件: `ATen/ops/special_hermite_polynomial_h.h` |
| 698 | `at::special_hermite_polynomial_he` | - | 功能缺失 | 头文件: `ATen/ops/special_hermite_polynomial_he.h` |
| 699 | `at::special_i0` | - | 功能缺失 | 头文件: `ATen/ops/special_i0.h` |
| 700 | `at::special_i0e` | - | 功能缺失 | 头文件: `ATen/ops/special_i0e.h` |
| 701 | `at::special_i1` | - | 功能缺失 | 头文件: `ATen/ops/special_i1.h` |
| 702 | `at::special_i1e` | - | 功能缺失 | 头文件: `ATen/ops/special_i1e.h` |
| 703 | `at::special_laguerre_polynomial_l` | - | 功能缺失 | 头文件: `ATen/ops/special_laguerre_polynomial_l.h` |
| 704 | `at::special_legendre_polynomial_p` | - | 功能缺失 | 头文件: `ATen/ops/special_legendre_polynomial_p.h` |
| 705 | `at::special_log1p` | - | 功能缺失 | 头文件: `ATen/ops/special_log1p.h` |
| 706 | `at::special_log_ndtr` | - | 功能缺失 | 头文件: `ATen/ops/special_log_ndtr.h` |
| 707 | `at::special_log_softmax` | - | 功能缺失 | 头文件: `ATen/ops/special_log_softmax.h` |
| 708 | `at::special_logit` | - | 功能缺失 | 头文件: `ATen/ops/special_logit.h` |
| 709 | `at::special_logsumexp` | - | 功能缺失 | 头文件: `ATen/ops/special_logsumexp.h` |
| 710 | `at::special_modified_bessel_i0` | - | 功能缺失 | 头文件: `ATen/ops/special_modified_bessel_i0.h` |
| 711 | `at::special_modified_bessel_i1` | - | 功能缺失 | 头文件: `ATen/ops/special_modified_bessel_i1.h` |
| 712 | `at::special_modified_bessel_k0` | - | 功能缺失 | 头文件: `ATen/ops/special_modified_bessel_k0.h` |
| 713 | `at::special_modified_bessel_k1` | - | 功能缺失 | 头文件: `ATen/ops/special_modified_bessel_k1.h` |
| 714 | `at::special_multigammaln` | - | 功能缺失 | 头文件: `ATen/ops/special_multigammaln.h` |
| 715 | `at::special_ndtr` | - | 功能缺失 | 头文件: `ATen/ops/special_ndtr.h` |
| 716 | `at::special_ndtri` | - | 功能缺失 | 头文件: `ATen/ops/special_ndtri.h` |
| 717 | `at::special_polygamma` | - | 功能缺失 | 头文件: `ATen/ops/special_polygamma.h` |
| 718 | `at::special_psi` | - | 功能缺失 | 头文件: `ATen/ops/special_psi.h` |
| 719 | `at::special_round` | - | 功能缺失 | 头文件: `ATen/ops/special_round.h` |
| 720 | `at::special_scaled_modified_bessel_k0` | - | 功能缺失 | 头文件: `ATen/ops/special_scaled_modified_bessel_k0.h` |
| 721 | `at::special_scaled_modified_bessel_k1` | - | 功能缺失 | 头文件: `ATen/ops/special_scaled_modified_bessel_k1.h` |
| 722 | `at::special_shifted_chebyshev_polynomial_t` | - | 功能缺失 | 头文件: `ATen/ops/special_shifted_chebyshev_polynomial_t.h` |
| 723 | `at::special_shifted_chebyshev_polynomial_u` | - | 功能缺失 | 头文件: `ATen/ops/special_shifted_chebyshev_polynomial_u.h` |
| 724 | `at::special_shifted_chebyshev_polynomial_v` | - | 功能缺失 | 头文件: `ATen/ops/special_shifted_chebyshev_polynomial_v.h` |
| 725 | `at::special_shifted_chebyshev_polynomial_w` | - | 功能缺失 | 头文件: `ATen/ops/special_shifted_chebyshev_polynomial_w.h` |
| 726 | `at::special_sinc` | - | 功能缺失 | 头文件: `ATen/ops/special_sinc.h` |
| 727 | `at::special_softmax` | - | 功能缺失 | 头文件: `ATen/ops/special_softmax.h` |
| 728 | `at::special_spherical_bessel_j0` | - | 功能缺失 | 头文件: `ATen/ops/special_spherical_bessel_j0.h` |
| 729 | `at::special_xlog1py` | - | 功能缺失 | 头文件: `ATen/ops/special_xlog1py.h` |
| 730 | `at::special_xlogy` | - | 功能缺失 | 头文件: `ATen/ops/special_xlogy.h` |
| 731 | `at::special_zeta` | - | 功能缺失 | 头文件: `ATen/ops/special_zeta.h` |
| 732 | `at::split_copy` | - | 功能缺失 | 头文件: `ATen/ops/split_copy.h` |
| 733 | `at::split_with_sizes_copy` | - | 功能缺失 | 头文件: `ATen/ops/split_with_sizes_copy.h` |
| 734 | `at::squeeze_copy` | - | 功能缺失 | 头文件: `ATen/ops/squeeze_copy.h` |
| 735 | `at::sspaddmm` | - | 功能缺失 | 头文件: `ATen/ops/sspaddmm.h` |
| 736 | `at::std_mean` | - | 功能缺失 | 头文件: `ATen/ops/std_mean.h` |
| 737 | `at::stride` | - | 功能缺失 | 头文件: `ATen/ops/stride.h` |
| 738 | `at::sub` | - | 功能缺失 | 头文件: `ATen/ops/sub.h` |
| 739 | `at::sum_to_size` | - | 功能缺失 | 头文件: `ATen/ops/sum_to_size.h` |
| 740 | `at::swapaxes` | - | 功能缺失 | 头文件: `ATen/ops/swapaxes.h` |
| 741 | `at::swapdims` | - | 功能缺失 | 头文件: `ATen/ops/swapdims.h` |
| 742 | `at::sym_constrain_range` | - | 功能缺失 | 头文件: `ATen/ops/sym_constrain_range.h` |
| 743 | `at::sym_constrain_range_for_size` | - | 功能缺失 | 头文件: `ATen/ops/sym_constrain_range_for_size.h` |
| 744 | `at::sym_is_contiguous` | - | 功能缺失 | 头文件: `ATen/ops/sym_is_contiguous.h` |
| 745 | `at::sym_numel` | - | 功能缺失 | 头文件: `ATen/ops/sym_numel.h` |
| 746 | `at::sym_size` | - | 功能缺失 | 头文件: `ATen/ops/sym_size.h` |
| 747 | `at::sym_storage_offset` | - | 功能缺失 | 头文件: `ATen/ops/sym_storage_offset.h` |
| 748 | `at::sym_stride` | - | 功能缺失 | 头文件: `ATen/ops/sym_stride.h` |
| 749 | `at::t_copy` | - | 功能缺失 | 头文件: `ATen/ops/t_copy.h` |
| 750 | `at::take` | - | 功能缺失 | 头文件: `ATen/ops/take.h` |
| 751 | `at::take_along_dim` | - | 功能缺失 | 头文件: `ATen/ops/take_along_dim.h` |
| 752 | `at::tensordot` | - | 功能缺失 | 头文件: `ATen/ops/tensordot.h` |
| 753 | `at::thnn_conv2d` | - | 功能缺失 | 头文件: `ATen/ops/thnn_conv2d.h` |
| 754 | `at::threshold` | - | 功能缺失 | 头文件: `ATen/ops/threshold.h` |
| 755 | `at::to_dense` | - | 功能缺失 | 头文件: `ATen/ops/to_dense.h` |
| 756 | `at::to_padded_tensor` | - | 功能缺失 | 头文件: `ATen/ops/to_padded_tensor.h` |
| 757 | `at::transpose_copy` | - | 功能缺失 | 头文件: `ATen/ops/transpose_copy.h` |
| 758 | `at::trapezoid` | - | 功能缺失 | 头文件: `ATen/ops/trapezoid.h` |
| 759 | `at::trapz` | - | 功能缺失 | 头文件: `ATen/ops/trapz.h` |
| 760 | `at::triplet_margin_loss` | - | 功能缺失 | 头文件: `ATen/ops/triplet_margin_loss.h` |
| 761 | `at::true_divide` | - | 功能缺失 | 头文件: `ATen/ops/true_divide.h` |
| 762 | `at::type_as` | - | 功能缺失 | 头文件: `ATen/ops/type_as.h` |
| 763 | `at::unbind_copy` | - | 功能缺失 | 头文件: `ATen/ops/unbind_copy.h` |
| 764 | `at::unflatten_dense_tensors` | - | 功能缺失 | 头文件: `ATen/ops/unflatten_dense_tensors.h` |
| 765 | `at::unfold_copy` | - | 功能缺失 | 头文件: `ATen/ops/unfold_copy.h` |
| 766 | `at::unique_dim` | - | 功能缺失 | 头文件: `ATen/ops/unique_dim.h` |
| 767 | `at::unique_dim_consecutive` | - | 功能缺失 | 头文件: `ATen/ops/unique_dim_consecutive.h` |
| 768 | `at::unsafe_chunk` | - | 功能缺失 | 头文件: `ATen/ops/unsafe_chunk.h` |
| 769 | `at::unsqueeze_copy` | - | 功能缺失 | 头文件: `ATen/ops/unsqueeze_copy.h` |
| 770 | `at::upsample_bicubic2d` | - | 功能缺失 | 头文件: `ATen/ops/upsample_bicubic2d.h` |
| 771 | `at::upsample_bilinear2d` | - | 功能缺失 | 头文件: `ATen/ops/upsample_bilinear2d.h` |
| 772 | `at::upsample_linear1d` | - | 功能缺失 | 头文件: `ATen/ops/upsample_linear1d.h` |
| 773 | `at::upsample_nearest1d` | - | 功能缺失 | 头文件: `ATen/ops/upsample_nearest1d.h` |
| 774 | `at::upsample_nearest2d` | - | 功能缺失 | 头文件: `ATen/ops/upsample_nearest2d.h` |
| 775 | `at::upsample_nearest3d` | - | 功能缺失 | 头文件: `ATen/ops/upsample_nearest3d.h` |
| 776 | `at::upsample_trilinear3d` | - | 功能缺失 | 头文件: `ATen/ops/upsample_trilinear3d.h` |
| 777 | `at::values` | - | 功能缺失 | 头文件: `ATen/ops/values.h` |
| 778 | `at::values_copy` | - | 功能缺失 | 头文件: `ATen/ops/values_copy.h` |
| 779 | `at::vander` | - | 功能缺失 | 头文件: `ATen/ops/vander.h` |
| 780 | `at::var_mean` | - | 功能缺失 | 头文件: `ATen/ops/var_mean.h` |
| 781 | `at::vdot` | - | 功能缺失 | 头文件: `ATen/ops/vdot.h` |
| 782 | `at::view_as_complex` | - | 功能缺失 | 头文件: `ATen/ops/view_as_complex.h` |
| 783 | `at::view_as_complex_copy` | - | 功能缺失 | 头文件: `ATen/ops/view_as_complex_copy.h` |
| 784 | `at::view_as_real` | - | 功能缺失 | 头文件: `ATen/ops/view_as_real.h` |
| 785 | `at::view_as_real_copy` | - | 功能缺失 | 头文件: `ATen/ops/view_as_real_copy.h` |
| 786 | `at::view_copy` | - | 功能缺失 | 头文件: `ATen/ops/view_copy.h` |
| 787 | `at::vstack` | - | 功能缺失 | 头文件: `ATen/ops/vstack.h` |
| 788 | `at::xlogy` | - | 功能缺失 | 头文件: `ATen/ops/xlogy.h` |
| 789 | `at::xor` | - | 功能缺失 | 头文件: `ATen/ops/xor.h` |
| 790 | `at::zero` | - | 功能缺失 | 头文件: `ATen/ops/zero.h` |

## 统计

- **API 完全一致**: 66 个
- **仅 API 调用方式不一致**: 1 个
- **仅参数名不一致**: 72 个
- **paddle 参数更多**: 46 个
- **参数默认值不一致**: 3 个
- **torch 参数更多**: 22 个
- **输入参数用法不一致**: 0 个
- **输入参数类型不一致**: 49 个
- **返回参数类型不一致**: 29 个
- **组合替代实现**: 0 个
- **API 别名**: 18 个
- **功能缺失**: 790 个
- **API 别名映射数**: 18 个
- **libtorch 主 ops 总数**: 1082 个
- **实际参与映射的 ops 数**: 1096 个
