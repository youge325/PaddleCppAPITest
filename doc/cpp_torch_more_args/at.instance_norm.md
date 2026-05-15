## [torch 参数更多]at::instance_norm

### PyTorch C++ API
```cpp
at::instance_norm(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled)
```

### Paddle C++ API
```cpp
paddle::experimental::instance_norm(x, scale, bias, epsilon=1e-5)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| input | x | 仅参数名不一致，`input` 对应 `x`。 |
| weight | - | Paddle 无此参数，PyTorch 有 `weight`。 |
| bias | bias | 参数名一致。 |
| running_mean | - | Paddle 无此参数，PyTorch 有 `running_mean`。 |
| running_var | - | Paddle 无此参数，PyTorch 有 `running_var`。 |
| use_input_stats | - | Paddle 无此参数，PyTorch 有 `use_input_stats`。 |
| momentum | - | Paddle 无此参数，PyTorch 有 `momentum`。 |
| eps | - | Paddle 无此参数，PyTorch 有 `eps`。 |
| cudnn_enabled | - | Paddle 无此参数，PyTorch 有 `cudnn_enabled`。 |
| - | scale | PyTorch 无此参数，Paddle 有 `scale`。 |
| - | epsilon | PyTorch 无此参数，Paddle 有 `epsilon`。 |
