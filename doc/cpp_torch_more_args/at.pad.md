## [torch 参数更多]at::pad

### PyTorch C++ API
```cpp
at::pad(self, pad, mode="constant", value=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::pad(x, paddings, pad_value)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| pad | - | Paddle 无此参数，PyTorch 有 `pad`。 |
| mode | - | Paddle 无此参数，PyTorch 有 `mode`。 |
| value | - | Paddle 无此参数，PyTorch 有 `value`。 |
| - | paddings | PyTorch 无此参数，Paddle 有 `paddings`。 |
| - | pad_value | PyTorch 无此参数，Paddle 有 `pad_value`。 |
