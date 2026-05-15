## [paddle 参数更多]at::tril_indices

### PyTorch C++ API
```cpp
at::tril_indices(row, col, offset=0, options=at::kLong)
```

### Paddle C++ API
```cpp
paddle::experimental::tril_indices(rows, cols, offset, dtype, place={})
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| row | - | Paddle 无此参数，PyTorch 有 `row`。 |
| col | - | Paddle 无此参数，PyTorch 有 `col`。 |
| offset | offset | 参数名一致。 |
| options | - | Paddle 无此参数，PyTorch 有 `options`。 |
| - | rows | PyTorch 无此参数，Paddle 有 `rows`。 |
| - | cols | PyTorch 无此参数，Paddle 有 `cols`。 |
| - | dtype | PyTorch 无此参数，Paddle 有 `dtype`。 |
| - | place | PyTorch 无此参数，Paddle 有 `place`。 |
