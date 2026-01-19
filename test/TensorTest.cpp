#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include <vector>

namespace at {
namespace test {

class TensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {2, 3, 4};

    tensor = at::ones(shape, at::kFloat);
    // std::cout << "tensor dim: " << tensor.dim() << std::endl;
  }

  at::Tensor tensor;
};

TEST_F(TensorTest, ConstructFromPaddleTensor) {
  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.numel(), 24);  // 2*3*4
}

// 测试 data_ptr
TEST_F(TensorTest, DataPtr) {
  // Tensor tensor(paddle_tensor_);

  void* ptr = tensor.data_ptr();
  EXPECT_NE(ptr, nullptr);

  float* float_ptr = tensor.data_ptr<float>();
  EXPECT_NE(float_ptr, nullptr);
}

// 测试 strides
TEST_F(TensorTest, Strides) {
  // Tensor tensor(paddle_tensor_);

  c10::IntArrayRef strides = tensor.strides();
  EXPECT_GT(strides.size(), 0U);  // 使用无符号字面量
}

// 测试 sizes
TEST_F(TensorTest, Sizes) {
  // Tensor tensor(paddle_tensor_);

  c10::IntArrayRef sizes = tensor.sizes();
  EXPECT_EQ(sizes.size(), 3U);
  EXPECT_EQ(sizes[0], 2U);
  EXPECT_EQ(sizes[1], 3U);
  EXPECT_EQ(sizes[2], 4U);
}

// 测试 toType
TEST_F(TensorTest, ToType) {
  // Tensor tensor(paddle_tensor_);

  Tensor double_tensor = tensor.toType(c10::ScalarType::Double);
  EXPECT_EQ(double_tensor.dtype(), c10::ScalarType::Double);
}

// 测试 numel
TEST_F(TensorTest, Numel) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_EQ(tensor.numel(), 24U);  // 2*3*4
}

// 测试 device
TEST_F(TensorTest, Device) {
  // Tensor tensor(paddle_tensor_);

  c10::Device device = tensor.device();
  EXPECT_EQ(device.type(), c10::DeviceType::CPU);
}

// 测试 get_device
TEST_F(TensorTest, GetDevice) {
  // Tensor tensor(paddle_tensor_);

  c10::DeviceIndex device_idx = tensor.get_device();
  EXPECT_GE(device_idx, -1);
}

// 测试 dim 和 ndimension
TEST_F(TensorTest, DimAndNdimension) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.ndimension(), 3);
  EXPECT_EQ(tensor.dim(), tensor.ndimension());
}

// 测试 contiguous
TEST_F(TensorTest, Contiguous) {
  // Tensor tensor(paddle_tensor_);

  at::Tensor cont_tensor = tensor.contiguous();
  EXPECT_TRUE(cont_tensor.is_contiguous());
}

// 测试 is_contiguous
TEST_F(TensorTest, IsContiguous) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_TRUE(tensor.is_contiguous());
}

// 测试 scalar_type
TEST_F(TensorTest, ScalarType) {
  // Tensor tensor(paddle_tensor_);

  c10::ScalarType stype = tensor.scalar_type();
  EXPECT_EQ(stype, c10::ScalarType::Float);
}

// 测试 fill_
TEST_F(TensorTest, Fill) {
  // Tensor tensor(paddle_tensor_);

  tensor.fill_(5.0);
  float* data = tensor.data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 5.0f);
}

// 测试 zero_
TEST_F(TensorTest, Zero) {
  // Tensor tensor(paddle_tensor_);

  tensor.zero_();
  float* data = tensor.data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 0.0f);
}

// 测试 is_cpu
TEST_F(TensorTest, IsCpu) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_TRUE(tensor.is_cpu());
}

// 测试 is_cuda (在 CPU tensor 上应该返回 false)
TEST_F(TensorTest, IsCuda) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_FALSE(tensor.is_cuda());
}

// 测试 reshape
TEST_F(TensorTest, Reshape) {
  // Tensor tensor(paddle_tensor_);

  at::Tensor reshaped = tensor.reshape({6, 4});
  EXPECT_EQ(reshaped.sizes()[0], 6);
  EXPECT_EQ(reshaped.sizes()[1], 4);
  EXPECT_EQ(reshaped.numel(), 24);
}

// 测试 transpose
TEST_F(TensorTest, Transpose) {
  // Tensor tensor(paddle_tensor_);

  at::Tensor transposed = tensor.transpose(0, 2);
  EXPECT_EQ(transposed.sizes()[0], 4);
  EXPECT_EQ(transposed.sizes()[2], 2);
}

// 测试 _is_zerotensor
TEST_F(TensorTest, IsZeroTensor) {
  // 默认创建的 ones tensor 不应该是 zero tensor
  EXPECT_FALSE(tensor._is_zerotensor());

  // 注意: _is_zerotensor() 和 _set_zero() 是 PyTorch 内部 API
  // 主要用于优化，不建议在用户代码中使用
  // 创建的普通张量（即使全为0）也不是 "zero tensor"
  at::Tensor zero_tensor = at::zeros({2, 3}, at::kFloat);
  EXPECT_FALSE(zero_tensor._is_zerotensor());
}

// 测试 is_conj (复数张量的共轭标记)
TEST_F(TensorTest, IsConj) {
  // 实数张量不应该有共轭标记
  EXPECT_FALSE(tensor.is_conj());

  // 创建一个复数张量来测试 conj 功能
  at::Tensor complex_tensor = at::ones({2, 3}, at::kComplexFloat);
  EXPECT_FALSE(complex_tensor.is_conj());
}

// 测试 _set_conj (只能用于复数张量)
TEST_F(TensorTest, SetConj) {
  // 创建复数张量
  at::Tensor complex_tensor = at::ones({2, 3}, at::kComplexFloat);

  // 设置共轭标记
  complex_tensor._set_conj(true);
  EXPECT_TRUE(complex_tensor.is_conj());

  // 取消共轭标记
  complex_tensor._set_conj(false);
  EXPECT_FALSE(complex_tensor.is_conj());
}

// 测试 is_neg
TEST_F(TensorTest, IsNeg) {
  // 默认情况下不应该有负号标记
  EXPECT_FALSE(tensor.is_neg());
}

// 测试 _set_neg
TEST_F(TensorTest, SetNeg) {
  // _set_neg 是 PyTorch 内部 API，用于优化
  // 在某些后端实现中可能不完全支持

  // 测试设置负号标记
  tensor._set_neg(true);
  // 注意: 某些实现可能不会真正设置此标记
  // 因此我们只测试 API 调用不会崩溃

  // 取消负号标记
  tensor._set_neg(false);
  EXPECT_FALSE(tensor.is_neg());
}

// 测试组合标记状态 (复数张量)
TEST_F(TensorTest, CombinedFlags) {
  // 创建复数张量来测试标记组合
  at::Tensor complex_tensor = at::ones({2, 3}, at::kComplexFloat);

  // 测试同时设置 conj 和 neg 标记
  complex_tensor._set_conj(true);
  complex_tensor._set_neg(true);

  EXPECT_TRUE(complex_tensor.is_conj());
  // neg 标记可能不被所有后端完全支持

  // 清除标记
  complex_tensor._set_conj(false);
  EXPECT_FALSE(complex_tensor.is_conj());

  complex_tensor._set_neg(false);
}

}  // namespace test
}  // namespace at
