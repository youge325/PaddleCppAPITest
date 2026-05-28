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

static void write_broadcast_to_result_to_file(FileManerger* file,
                                              const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
  if (result.numel() == 0) {
    *file << "empty ";
    return;
  }
  at::Tensor cont = result.contiguous();
  float* data = cont.data_ptr<float>();
  *file << std::to_string(data[0]) << " ";
  *file << std::to_string(data[cont.numel() - 1]) << " ";
  *file << std::to_string(cont.sum().item<float>()) << " ";
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
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(result.sizes()[i]) << " ";
  }
  at::Tensor cont = result.contiguous();
  double* data = cont.data_ptr<double>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[cont.numel() - 1]) << " ";
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
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(result.sizes()[i]) << " ";
  }
  at::Tensor cont = result.contiguous();
  int* data = cont.data_ptr<int>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[cont.numel() - 1]) << " ";
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
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(result.sizes()[i]) << " ";
  }
  at::Tensor cont = result.contiguous();
  int64_t* data = cont.data_ptr<int64_t>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[cont.numel() - 1]) << " ";
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
