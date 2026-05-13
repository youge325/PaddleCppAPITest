#include <ATen/ATen.h>
#include <ATen/ops/index_add.h>
#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

class IndexAddTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

static at::Tensor make_long_index(const std::vector<int64_t>& values) {
  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  return at::from_blob(const_cast<int64_t*>(values.data()),
                       {static_cast<int64_t>(values.size())},
                       options)
      .clone();
}

static at::Tensor make_int_index(const std::vector<int32_t>& values) {
  auto options = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
  return at::from_blob(const_cast<int32_t*>(values.data()),
                       {static_cast<int64_t>(values.size())},
                       options)
      .clone();
}

static void write_result_to_file(FileManerger* file, const at::Tensor& r) {
  *file << std::to_string(r.dim()) << " ";
  *file << std::to_string(r.numel()) << " ";
  for (int64_t i = 0; i < r.dim(); ++i) {
    *file << std::to_string(r.sizes()[i]) << " ";
  }
  if (r.numel() == 0) {
    *file << "empty ";
    return;
  }
  at::Tensor c = r.contiguous();
  if (c.scalar_type() == at::kFloat) {
    float* d = c.data_ptr<float>();
    for (int64_t i = 0; i < c.numel(); ++i)
      *file << std::to_string(d[i]) << " ";
  } else if (c.scalar_type() == at::kDouble) {
    double* d = c.data_ptr<double>();
    for (int64_t i = 0; i < c.numel(); ++i)
      *file << std::to_string(d[i]) << " ";
  } else if (c.scalar_type() == at::kInt) {
    int* d = c.data_ptr<int>();
    for (int64_t i = 0; i < c.numel(); ++i)
      *file << std::to_string(d[i]) << " ";
  } else if (c.scalar_type() == at::kLong) {
    int64_t* d = c.data_ptr<int64_t>();
    for (int64_t i = 0; i < c.numel(); ++i)
      *file << std::to_string(d[i]) << " ";
  }
}

TEST_F(IndexAddTest, FreeFunctionFloat) {
  at::Tensor self = at::zeros({5}, at::kFloat);
  at::Tensor index = make_long_index({0, 2, 4});
  at::Tensor source = at::full({3}, 2.0f, at::kFloat);

  at::Tensor result = at::index_add(self, 0, index, source);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "FreeFunctionFloat ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, MethodOutOfPlaceFloat) {
  at::Tensor self = at::ones({4}, at::kFloat);
  at::Tensor index = make_long_index({0, 1});
  at::Tensor source = at::full({2}, 3.0f, at::kFloat);

  at::Tensor result = self.index_add(0, index, source);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MethodOutOfPlaceFloat ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, MethodInplaceFloat) {
  at::Tensor self = at::ones({4}, at::kFloat);
  at::Tensor index = make_long_index({0, 2});
  at::Tensor source = at::full({2}, 4.0f, at::kFloat);

  self.index_add_(0, index, source);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MethodInplaceFloat ";
  write_result_to_file(&file, self);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, DtypeDouble) {
  at::Tensor self = at::zeros({4}, at::kDouble);
  at::Tensor index = make_long_index({0, 2});
  at::Tensor source = at::full({2}, 1.5, at::kDouble);

  at::Tensor result = at::index_add(self, 0, index, source);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DtypeDouble ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, DtypeIntAlphaIntegral) {
  at::Tensor self = at::zeros({4}, at::kInt);
  at::Tensor index = make_long_index({0, 1, 2});
  at::Tensor source = at::full({3}, 3, at::kInt);

  at::Tensor result = at::index_add(self, 0, index, source, at::Scalar(2));

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DtypeIntAlphaIntegral ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, DtypeLongAlphaIntegral) {
  at::Tensor self = at::zeros({3}, at::kLong);
  at::Tensor index = make_long_index({0, 2});
  at::Tensor source = at::full({2}, 5, at::kLong);

  at::Tensor result = at::index_add(self, 0, index, source, at::Scalar(3));

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DtypeLongAlphaIntegral ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, AlphaPositiveFloat) {
  at::Tensor self = at::zeros({3}, at::kFloat);
  at::Tensor index = make_long_index({0, 1, 2});
  at::Tensor source = at::full({3}, 1.5f, at::kFloat);

  at::Tensor result = at::index_add(self, 0, index, source, at::Scalar(2.0));

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AlphaPositiveFloat ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, AlphaNegativeFloat) {
  at::Tensor self = at::full({3}, 10.0f, at::kFloat);
  at::Tensor index = make_long_index({0, 1, 2});
  at::Tensor source = at::full({3}, 2.0f, at::kFloat);

  at::Tensor result = at::index_add(self, 0, index, source, at::Scalar(-1.0));

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AlphaNegativeFloat ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, AlphaZero) {
  at::Tensor self = at::full({3}, 7.0f, at::kFloat);
  at::Tensor index = make_long_index({0, 1, 2});
  at::Tensor source = at::full({3}, 99.0f, at::kFloat);

  at::Tensor result = at::index_add(self, 0, index, source, at::Scalar(0.0));

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AlphaZero ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, DimZero2D) {
  at::Tensor self = at::zeros({3, 4}, at::kFloat);
  at::Tensor index = make_long_index({0, 2});
  at::Tensor source = at::full({2, 4}, 1.0f, at::kFloat);

  at::Tensor result = at::index_add(self, 0, index, source);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DimZero2D ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, DimLast2D) {
  at::Tensor self = at::zeros({3, 4}, at::kFloat);
  at::Tensor index = make_long_index({1, 3});
  at::Tensor source = at::full({3, 2}, 2.0f, at::kFloat);

  at::Tensor result = at::index_add(self, 1, index, source);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DimLast2D ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, DimNegative) {
  at::Tensor self = at::zeros({2, 3, 4}, at::kFloat);
  at::Tensor index = make_long_index({0, 2});
  at::Tensor source = at::full({2, 3, 2}, 3.0f, at::kFloat);

  at::Tensor result = at::index_add(self, -1, index, source);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DimNegative ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, IndexInt32Accepted) {
  at::Tensor self = at::zeros({4}, at::kFloat);
  at::Tensor index = make_int_index({0, 3});
  at::Tensor source = at::full({2}, 1.0f, at::kFloat);

  at::Tensor result = at::index_add(self, 0, index, source);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexInt32Accepted ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, IndexSingleElement) {
  at::Tensor self = at::zeros({4}, at::kFloat);
  at::Tensor index = make_long_index({2});
  at::Tensor source = at::full({1}, 9.0f, at::kFloat);

  at::Tensor result = at::index_add(self, 0, index, source);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexSingleElement ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, IndexRepeatsAccumulate) {
  at::Tensor self = at::zeros({3}, at::kFloat);
  at::Tensor index = make_long_index({0, 0, 0});
  at::Tensor source = at::full({3}, 2.0f, at::kFloat);

  at::Tensor result = at::index_add(self, 0, index, source);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexRepeatsAccumulate ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, ShapeMedium2D) {
  at::Tensor self = at::ones({8, 16}, at::kFloat);
  std::vector<int64_t> idx_vals;
  for (int64_t i = 0; i < 4; ++i) idx_vals.push_back(i * 2);
  at::Tensor index = make_long_index(idx_vals);
  at::Tensor source = at::full({4, 16}, 0.5f, at::kFloat);

  at::Tensor result = at::index_add(self, 0, index, source);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ShapeMedium2D ";
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(result.sizes()[i]) << " ";
  }
  at::Tensor cont = result.contiguous();
  float* d = cont.data_ptr<float>();
  file << std::to_string(d[0]) << " ";
  file << std::to_string(d[cont.numel() - 1]) << " ";
  file << std::to_string(cont.sum().item<float>()) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, IndexDtypeFloatThrows) {
  at::Tensor self = at::zeros({4}, at::kFloat);
  at::Tensor index = at::zeros({2}, at::kFloat);
  at::Tensor source = at::zeros({2}, at::kFloat);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexDtypeFloatThrows ";
  try {
    (void)at::index_add(self, 0, index, source);
    file << "no_throw ";
  } catch (const std::exception&) {
    file << "exception: ";
  } catch (...) {
    file << "exception: ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, IndexOutOfBoundsThrows) {
  at::Tensor self = at::ones({4}, at::kFloat);
  at::Tensor index = make_long_index({0, 99});
  at::Tensor source = at::zeros({2}, at::kFloat);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexOutOfBoundsThrows ";
  try {
    (void)at::index_add(self, 0, index, source);
    file << "no_throw ";
  } catch (const std::exception&) {
    file << "exception: ";
  } catch (...) {
    file << "exception: ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, ShapeMismatchThrows) {
  at::Tensor self = at::ones({4, 3}, at::kFloat);
  at::Tensor index = make_long_index({0, 1, 2});
  at::Tensor source = at::zeros({3, 5}, at::kFloat);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ShapeMismatchThrows ";
  try {
    (void)at::index_add(self, 0, index, source);
    file << "no_throw ";
  } catch (const std::exception&) {
    file << "exception: ";
  } catch (...) {
    file << "exception: ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(IndexAddTest, IntegerSelfFloatAlphaNoThrow) {
  at::Tensor self = at::zeros({3}, at::kLong);
  at::Tensor index = make_long_index({0, 1});
  at::Tensor source = at::ones({2}, at::kLong);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IntegerSelfFloatAlphaNoThrow ";
  try {
    (void)at::index_add(self, 0, index, source, at::Scalar(1.5));
    file << "no_throw ";
  } catch (const std::exception&) {
    file << "exception: ";
  } catch (...) {
    file << "exception: ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
