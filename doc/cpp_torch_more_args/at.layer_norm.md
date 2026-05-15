## [torch 参数更多]at::layer_norm

### PyTorch C++ API
```cpp
at::layer_norm(input, normalized_shape, weight={}, bias={}, eps=1e-05, cudnn_enable=true)
```

### Paddle C++ API
```cpp
paddle::experimental::layer_norm(x, scale, bias, epsilon=1e-5, begin_norm_axis=1)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| input | x | 仅参数名不一致，`input` 对应 `x`。 |
| normalized_shape | - | Paddle 无此参数，PyTorch 有 `normalized_shape`。 |
| weight | - | Paddle 无此参数，PyTorch 有 `weight`。 |
| bias | bias | 参数名一致。 |
| eps | - | Paddle 无此参数，PyTorch 有 `eps`。 |
| cudnn_enable | - | Paddle 无此参数，PyTorch 有 `cudnn_enable`。 |
| - | scale | PyTorch 无此参数，Paddle 有 `scale`。 |
| - | epsilon | PyTorch 无此参数，Paddle 有 `epsilon`。 |
| - | begin_norm_axis | PyTorch 无此参数，Paddle 有 `begin_norm_axis`。 |
