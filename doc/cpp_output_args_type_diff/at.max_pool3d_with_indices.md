## [返回参数类型不一致]at::max_pool3d_with_indices

### PyTorch C++ API
```cpp
at::max_pool3d_with_indices(self, kernel_size, stride={}, padding=0, dilation=1, ceil_mode=false) -> ::tuple<Tensor,Tensor>
```

### Paddle C++ API
```cpp
paddle::experimental::max_pool3d_with_index(x, kernel_size, strides={1, 1, 1}, paddings={0, 0, 0}, dilations={1, 1, 1}, global_pooling=false, adaptive=false, ceil_mode=false) -> tuple<Tensor, Tensor>
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| ::tuple<Tensor,Tensor> | tuple<Tensor, Tensor> | 返回类型不一致。 |
