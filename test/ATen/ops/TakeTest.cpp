#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/take.h>
#include <ATen/ops/zeros.h>
#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

class TakeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

static void write_result_to_file(FileManerger* file, const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  if (result.scalar_type() == at::kFloat) {
    float* data = result.data_ptr<float>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (result.scalar_type() == at::kDouble) {
    double* data = result.data_ptr<double>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (result.scalar_type() == at::kInt) {
    int* data = result.data_ptr<int>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (result.scalar_type() == at::kLong) {
    int64_t* data = result.data_ptr<int64_t>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  }
}

static at::Tensor make_test_tensor(at::ScalarType dtype) {
  if (dtype == at::kFloat) {
    auto t = at::zeros({3, 4}, at::kFloat);
    float* data = t.data_ptr<float>();
    for (int i = 0; i < 12; ++i) data[i] = static_cast<float>(i);
    return t;
  } else if (dtype == at::kDouble) {
    auto t = at::zeros({3, 4}, at::kDouble);
    double* data = t.data_ptr<double>();
    for (int i = 0; i < 12; ++i) data[i] = static_cast<double>(i);
    return t;
  } else if (dtype == at::kInt) {
    auto t = at::zeros({3, 4}, at::kInt);
    int* data = t.data_ptr<int>();
    for (int i = 0; i < 12; ++i) data[i] = i;
    return t;
  } else if (dtype == at::kLong) {
    auto t = at::zeros({3, 4}, at::kLong);
    int64_t* data = t.data_ptr<int64_t>();
    for (int i = 0; i < 12; ++i) data[i] = i;
    return t;
  }
  return at::zeros({3, 4}, at::kFloat);
}

static at::Tensor make_index_tensor(const std::vector<int64_t>& values) {
  auto t = at::empty({static_cast<int64_t>(values.size())},
                     at::TensorOptions().dtype(at::kLong));
  int64_t* data = t.data_ptr<int64_t>();
  for (size_t i = 0; i < values.size(); ++i) data[i] = values[i];
  return t;
}

// Shape: small 1D index
TEST_F(TakeTest, TakeFloatSmall) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = make_index_tensor({0, 5, 11});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "TakeFloatSmall ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeDoubleSmall) {
  at::Tensor t = make_test_tensor(at::kDouble);
  at::Tensor index = make_index_tensor({0, 5, 11});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeDoubleSmall ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeIntSmall) {
  at::Tensor t = make_test_tensor(at::kInt);
  at::Tensor index = make_index_tensor({0, 5, 11});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeIntSmall ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeLongSmall) {
  at::Tensor t = make_test_tensor(at::kLong);
  at::Tensor index = make_index_tensor({0, 5, 11});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeLongSmall ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Shape: multi-dimensional index
TEST_F(TakeTest, TakeFloatMultiDimIndex) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = at::empty({2, 2}, at::TensorOptions().dtype(at::kLong));
  int64_t* data = index.data_ptr<int64_t>();
  data[0] = 0;
  data[1] = 3;
  data[2] = 7;
  data[3] = 10;
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeFloatMultiDimIndex ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Shape: scalar index (0-dim)
TEST_F(TakeTest, TakeFloatScalarIndex) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = at::empty({}, at::TensorOptions().dtype(at::kLong));
  index.data_ptr<int64_t>()[0] = 7;
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeFloatScalarIndex ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Shape: empty index (boundary)
TEST_F(TakeTest, TakeFloatEmptyIndex) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = at::empty({0}, at::TensorOptions().dtype(at::kLong));
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeFloatEmptyIndex ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Shape: duplicate indices
TEST_F(TakeTest, TakeFloatDuplicateIndices) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = make_index_tensor({1, 1, 3, 1});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeFloatDuplicateIndices ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Exception: index out of range
TEST_F(TakeTest, TakeExceptionOutOfRange) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = make_index_tensor({100});  // out of range

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeExceptionOutOfRange ";
  try {
    at::Tensor result = at::take(t, index);
    write_result_to_file(&file, result);
  } catch (const std::exception& e) {
    file << "exception: ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
