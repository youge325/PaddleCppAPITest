## [返回参数类型不一致]at::_unique

### PyTorch C++ API
```cpp
at::_unique(self, sorted=true, return_inverse=false) -> ::tuple<Tensor,Tensor>
```

### Paddle C++ API
```cpp
paddle::experimental::unique(x, return_index, return_inverse, return_counts, axis, dtype=DataType::INT64) -> tuple<Tensor, Tensor, Tensor, Tensor>
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| ::tuple<Tensor,Tensor> | tuple<Tensor, Tensor, Tensor, Tensor> | 返回类型不一致。 |
