## [paddle 参数更多]at::uniform

### PyTorch C++ API
```cpp
at::uniform(self, from=0, to=1, generator=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::uniform(shape, dtype, min, max, seed, place={})
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | - | Paddle 无此参数，PyTorch 有 `self`。 |
| from | - | Paddle 无此参数，PyTorch 有 `from`。 |
| to | - | Paddle 无此参数，PyTorch 有 `to`。 |
| generator | - | Paddle 无此参数，PyTorch 有 `generator`。 |
| - | shape | PyTorch 无此参数，Paddle 有 `shape`。 |
| - | dtype | PyTorch 无此参数，Paddle 有 `dtype`。 |
| - | min | PyTorch 无此参数，Paddle 有 `min`。 |
| - | max | PyTorch 无此参数，Paddle 有 `max`。 |
| - | seed | PyTorch 无此参数，Paddle 有 `seed`。 |
| - | place | PyTorch 无此参数，Paddle 有 `place`。 |
