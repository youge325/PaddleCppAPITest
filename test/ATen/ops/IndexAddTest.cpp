#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/index_add.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

static void write_index_add_result_to_file(FileManerger* file,
                                           const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
  switch (result.scalar_type()) {
    case at::kFloat: {
      float* data = result.data_ptr<float>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        *file << std::to_string(data[i]) << " ";
      }
      break;
    }
    case at::kDouble: {
      double* data = result.data_ptr<double>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        *file << std::to_string(data[i]) << " ";
      }
      break;
    }
    case at::kInt: {
      int32_t* data = result.data_ptr<int32_t>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        *file << std::to_string(data[i]) << " ";
      }
      break;
    }
    case at::kLong: {
      int64_t* data = result.data_ptr<int64_t>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        *file << std::to_string(data[i]) << " ";
      }
      break;
    }
    default: {
      *file << "unsupported_dtype ";
      break;
    }
  }
}

class IndexAddTest : public ::testing::Test {
 protected:
  void SetUp() override {
    base = at::zeros({5, 3}, at::kFloat);
    idx = at::empty({3}, at::kLong);
    int64_t* idx_data = idx.data_ptr<int64_t>();
    idx_data[0] = 0;
    idx_data[1] = 2;
    idx_data[2] = 4;
    source = at::full({3, 3}, 1.0f, at::kFloat);
  }
  at::Tensor base;
  at::Tensor idx;
  at::Tensor source;
};

// ========== 基础功能 ==========

TEST_F(IndexAddTest, BasicIndexAdd) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "BasicIndexAdd ";
  at::Tensor result = base.index_add(0, idx, source);
  write_index_add_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Shape 覆盖 ==========

// 1D tensor
TEST_F(IndexAddTest, OneDimTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OneDimTensor ";
  at::Tensor t = at::zeros({5}, at::kFloat);
  at::Tensor i = at::empty({2}, at::kLong);
  i.data_ptr<int64_t>()[0] = 1;
  i.data_ptr<int64_t>()[1] = 3;
  at::Tensor s = at::full({2}, 2.0f, at::kFloat);
  at::Tensor result = t.index_add(0, i, s);
  write_index_add_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 小 shape {2, 3}
TEST_F(IndexAddTest, SmallShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SmallShape ";
  at::Tensor t = at::zeros({2, 3}, at::kFloat);
  at::Tensor i = at::empty({2}, at::kLong);
  i.data_ptr<int64_t>()[0] = 0;
  i.data_ptr<int64_t>()[1] = 1;
  at::Tensor s = at::full({2, 3}, 1.0f, at::kFloat);
  at::Tensor result = t.index_add(0, i, s);
  write_index_add_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 大 shape {100, 50}
TEST_F(IndexAddTest, LargeShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LargeShape ";
  at::Tensor t = at::zeros({100, 50}, at::kFloat);
  at::Tensor i = at::arange(10, at::kLong);
  at::Tensor s = at::ones({10, 50}, at::kFloat);
  at::Tensor result = t.index_add(0, i, s);
  write_index_add_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 3D tensor
TEST_F(IndexAddTest, ThreeDimTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ThreeDimTensor ";
  at::Tensor t = at::zeros({2, 3, 4}, at::kFloat);
  at::Tensor i = at::empty({2}, at::kLong);
  i.data_ptr<int64_t>()[0] = 0;
  i.data_ptr<int64_t>()[1] = 2;
  // source 在 dim=1 上的大小必须等于 index 的大小
  at::Tensor s = at::ones({2, 2, 4}, at::kFloat);
  at::Tensor result = t.index_add(1, i, s);
  write_index_add_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Dtype 覆盖 ==========

// float64
TEST_F(IndexAddTest, Float64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Float64Dtype ";
  at::Tensor t = at::zeros({5}, at::kDouble);
  at::Tensor i = at::empty({2}, at::kLong);
  i.data_ptr<int64_t>()[0] = 1;
  i.data_ptr<int64_t>()[1] = 3;
  at::Tensor s = at::full({2}, 1.5, at::kDouble);
  at::Tensor result = t.index_add(0, i, s);
  write_index_add_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int32
TEST_F(IndexAddTest, Int32Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int32Dtype ";
  at::Tensor t = at::zeros({5}, at::kInt);
  at::Tensor i = at::empty({2}, at::kLong);
  i.data_ptr<int64_t>()[0] = 0;
  i.data_ptr<int64_t>()[1] = 4;
  at::Tensor s = at::full({2}, 7, at::kInt);
  at::Tensor result = t.index_add(0, i, s);
  write_index_add_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int64
TEST_F(IndexAddTest, Int64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int64Dtype ";
  at::Tensor t = at::zeros({5}, at::kLong);
  at::Tensor i = at::empty({2}, at::kLong);
  i.data_ptr<int64_t>()[0] = 1;
  i.data_ptr<int64_t>()[1] = 3;
  at::Tensor s = at::full({2}, 10L, at::kLong);
  at::Tensor result = t.index_add(0, i, s);
  write_index_add_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== 值域覆盖 ==========

// 正数
TEST_F(IndexAddTest, PositiveValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PositiveValues ";
  at::Tensor t = at::zeros({5}, at::kFloat);
  at::Tensor i = at::empty({2}, at::kLong);
  i.data_ptr<int64_t>()[0] = 0;
  i.data_ptr<int64_t>()[1] = 2;
  at::Tensor s = at::full({2}, 3.5f, at::kFloat);
  at::Tensor result = t.index_add(0, i, s);
  write_index_add_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 负数
TEST_F(IndexAddTest, NegativeValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NegativeValues ";
  at::Tensor t = at::zeros({5}, at::kFloat);
  at::Tensor i = at::empty({2}, at::kLong);
  i.data_ptr<int64_t>()[0] = 1;
  i.data_ptr<int64_t>()[1] = 3;
  at::Tensor s = at::full({2}, -2.0f, at::kFloat);
  at::Tensor result = t.index_add(0, i, s);
  write_index_add_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 含零
TEST_F(IndexAddTest, ZeroValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZeroValues ";
  at::Tensor t = at::zeros({5}, at::kFloat);
  at::Tensor i = at::empty({2}, at::kLong);
  i.data_ptr<int64_t>()[0] = 0;
  i.data_ptr<int64_t>()[1] = 2;
  at::Tensor s = at::full({2}, 0.0f, at::kFloat);
  at::Tensor result = t.index_add(0, i, s);
  write_index_add_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== API 变体 ==========

// 原地操作 index_add_()
TEST_F(IndexAddTest, InplaceIndexAdd) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "InplaceIndexAdd ";
  at::Tensor t = at::zeros({5}, at::kFloat);
  at::Tensor i = at::empty({2}, at::kLong);
  i.data_ptr<int64_t>()[0] = 1;
  i.data_ptr<int64_t>()[1] = 3;
  at::Tensor s = at::full({2}, 2.0f, at::kFloat);
  void* original_ptr = t.data_ptr();
  t.index_add_(0, i, s);
  file << std::to_string(t.data_ptr() == original_ptr) << " ";
  write_index_add_result_to_file(&file, t);
  file << "\n";
  file.saveFile();
}

// 方法调用 t.index_add()
TEST_F(IndexAddTest, MethodIndexAdd) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MethodIndexAdd ";
  at::Tensor result = base.index_add(0, idx, source);
  write_index_add_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// alpha 参数
TEST_F(IndexAddTest, AlphaParameter) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AlphaParameter ";
  at::Tensor t = at::zeros({5}, at::kFloat);
  at::Tensor i = at::empty({2}, at::kLong);
  i.data_ptr<int64_t>()[0] = 1;
  i.data_ptr<int64_t>()[1] = 3;
  at::Tensor s = at::full({2}, 2.0f, at::kFloat);
  at::Tensor result = t.index_add(0, i, s, 3);
  write_index_add_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
