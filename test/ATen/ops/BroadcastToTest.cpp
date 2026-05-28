#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

class BroadcastToTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

<<<<<<< HEAD
// Compute element offset from flat index using strides (strides-aware access)
static inline int64_t compute_offset_from_flat_index(int64_t flat_idx,
                                                     const at::Tensor& tensor) {
  int64_t offset = 0;
  int64_t remainder = flat_idx;
  for (int64_t d = tensor.dim() - 1; d >= 0; --d) {
    int64_t coord = remainder % tensor.sizes()[d];
    remainder /= tensor.sizes()[d];
    offset += coord * tensor.strides()[d];
  }
  return offset;
}

// Write tensor metadata (dim, numel, sizes, strides) and all element values
// Uses strides-aware access to faithfully reflect the underlying layout.
// If Paddle and PyTorch produce different strides, result_cmp will DIFFER,
// and the difference should be recorded as a known mismatch.
=======
>>>>>>> d0b418e ([Cpp API Compatibility] Add broadcast_to cross-framework test and update mapping doc)
static void write_broadcast_to_result_to_file(FileManerger* file,
                                              const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
<<<<<<< HEAD
  // Record strides so layout differences are detected by result_cmp
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.strides()[i]) << " ";
  }
=======
>>>>>>> d0b418e ([Cpp API Compatibility] Add broadcast_to cross-framework test and update mapping doc)
  if (result.numel() == 0) {
    *file << "empty ";
    return;
  }
<<<<<<< HEAD
  switch (result.scalar_type()) {
    case at::kFloat: {
      float* data = result.data_ptr<float>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        int64_t offset = compute_offset_from_flat_index(i, result);
        *file << std::to_string(data[offset]) << " ";
      }
      break;
    }
    case at::kDouble: {
      double* data = result.data_ptr<double>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        int64_t offset = compute_offset_from_flat_index(i, result);
        *file << std::to_string(data[offset]) << " ";
      }
      break;
    }
    case at::kInt: {
      int32_t* data = result.data_ptr<int32_t>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        int64_t offset = compute_offset_from_flat_index(i, result);
        *file << std::to_string(data[offset]) << " ";
      }
      break;
    }
    case at::kLong: {
      int64_t* data = result.data_ptr<int64_t>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        int64_t offset = compute_offset_from_flat_index(i, result);
        *file << std::to_string(data[offset]) << " ";
      }
      break;
    }
    default:
      *file << "unsupported_dtype ";
      break;
  }
=======
  at::Tensor cont = result.contiguous();
  float* data = cont.data_ptr<float>();
  *file << std::to_string(data[0]) << " ";
  *file << std::to_string(data[cont.numel() - 1]) << " ";
  *file << std::to_string(cont.sum().item<float>()) << " ";
>>>>>>> d0b418e ([Cpp API Compatibility] Add broadcast_to cross-framework test and update mapping doc)
}

// ======================== Shape coverage ========================

// Small shape test
TEST_F(BroadcastToTest, BroadcastToSmall) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "BroadcastToSmall ";
  at::Tensor small = at::ones({1, 3}, at::kFloat);
  at::Tensor result = small.broadcast_to({2, 3});
  write_broadcast_to_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Large shape test
TEST_F(BroadcastToTest, BroadcastToLarge) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToLarge ";
  at::Tensor large = at::ones({1, 1, 128}, at::kFloat);
  at::Tensor result = large.broadcast_to({64, 32, 128});
  write_broadcast_to_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Boundary: empty tensor
TEST_F(BroadcastToTest, BroadcastToBoundaryEmpty) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToBoundaryEmpty ";
  at::Tensor t = at::ones({1, 0}, at::kFloat);
  at::Tensor result = t.broadcast_to({2, 0});
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << "empty ";
  file << "\n";
  file.saveFile();
}

// Boundary: rank less (input rank < target rank)
TEST_F(BroadcastToTest, BroadcastToBoundaryRankLess) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToBoundaryRankLess ";
  at::Tensor t = at::ones({1}, at::kFloat);
  at::Tensor result = t.broadcast_to({2, 3});
  write_broadcast_to_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ======================== Dtype coverage ========================

TEST_F(BroadcastToTest, BroadcastToDtypeFloat) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToDtypeFloat ";
  at::Tensor t = at::ones({1, 2}, at::kFloat);
  at::Tensor result = t.broadcast_to({3, 2});
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_broadcast_to_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(BroadcastToTest, BroadcastToDtypeDouble) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToDtypeDouble ";
  at::Tensor t = at::ones({1, 2}, at::kDouble);
  at::Tensor result = t.broadcast_to({3, 2});
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
<<<<<<< HEAD
  write_broadcast_to_result_to_file(&file, result);
=======
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(result.sizes()[i]) << " ";
  }
  at::Tensor cont = result.contiguous();
  double* data = cont.data_ptr<double>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[cont.numel() - 1]) << " ";
>>>>>>> d0b418e ([Cpp API Compatibility] Add broadcast_to cross-framework test and update mapping doc)
  file << "\n";
  file.saveFile();
}

TEST_F(BroadcastToTest, BroadcastToDtypeInt) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToDtypeInt ";
  at::Tensor t = at::ones({1, 2}, at::kInt);
  at::Tensor result = t.broadcast_to({3, 2});
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
<<<<<<< HEAD
  write_broadcast_to_result_to_file(&file, result);
=======
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(result.sizes()[i]) << " ";
  }
  at::Tensor cont = result.contiguous();
  int* data = cont.data_ptr<int>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[cont.numel() - 1]) << " ";
>>>>>>> d0b418e ([Cpp API Compatibility] Add broadcast_to cross-framework test and update mapping doc)
  file << "\n";
  file.saveFile();
}

TEST_F(BroadcastToTest, BroadcastToDtypeLong) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToDtypeLong ";
  at::Tensor t = at::ones({1, 2}, at::kLong);
  at::Tensor result = t.broadcast_to({3, 2});
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
<<<<<<< HEAD
  write_broadcast_to_result_to_file(&file, result);
=======
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(result.sizes()[i]) << " ";
  }
  at::Tensor cont = result.contiguous();
  int64_t* data = cont.data_ptr<int64_t>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[cont.numel() - 1]) << " ";
>>>>>>> d0b418e ([Cpp API Compatibility] Add broadcast_to cross-framework test and update mapping doc)
  file << "\n";
  file.saveFile();
}

// ======================== Exception coverage ========================

TEST_F(BroadcastToTest, BroadcastToInvalidNonSingleton) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToInvalidNonSingleton ";

  try {
    at::Tensor t = at::ones({2, 3}, at::kFloat);
    at::Tensor result = t.broadcast_to({2, 4});
    write_broadcast_to_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

TEST_F(BroadcastToTest, BroadcastToHighRankToLowRank) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToHighRankToLowRank ";

  try {
    at::Tensor t = at::ones({2, 3, 4}, at::kFloat);
    at::Tensor result = t.broadcast_to({3, 4});
    write_broadcast_to_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

// ======================== Function form ========================

TEST_F(BroadcastToTest, BroadcastToFunction) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToFunction ";
  at::Tensor t = at::ones({1, 2}, at::kFloat);
  at::Tensor result = at::broadcast_to(t, {3, 2});
  write_broadcast_to_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
