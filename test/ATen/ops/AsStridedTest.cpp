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

class AsStridedTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(AsStridedTest, AsStrided) {
  FileManerger file(g_custom_param.get());
  file.createFile();
  file << "AsStrided ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor strided = tensor.as_strided({3, 4, 2}, {2, 1, 6});
  file << std::to_string(strided.sizes()[0]) << " ";
  file << std::to_string(strided.sizes()[1]) << " ";
  file << std::to_string(strided.sizes()[2]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(AsStridedTest, AsStridedInplace) {
  FileManerger file(g_custom_param.get());
  file.openAppend();
  file << "AsStridedInplace ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  tensor.as_strided_({3, 4, 2}, {2, 1, 6});
  file << std::to_string(tensor.sizes()[0]) << " ";
  file << std::to_string(tensor.sizes()[1]) << " ";
  file << std::to_string(tensor.sizes()[2]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(AsStridedTest, AsStridedScatter) {
  FileManerger file(g_custom_param.get());
  file.openAppend();
  file << "AsStridedScatter ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor src = at::ones({3, 4, 2}, at::kFloat).fill_(2.0f);
  at::Tensor result = tensor.as_strided_scatter(src, {3, 4, 2}, {2, 1, 6});
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.dim()) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(AsStridedTest, AsStridedScatterPreservesInputShape) {
  FileManerger file(g_custom_param.get());
  file.openAppend();
  file << "AsStridedScatterPreservesInputShape ";
  at::Tensor tensor = at::arange(12, at::kFloat);
  at::Tensor src = at::full({2, 3}, 99.0f, at::kFloat);
  at::Tensor result = tensor.as_strided_scatter(src, {2, 3}, {3, 1});
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  float* data = result.data_ptr<float>();
  for (int i = 0; i < 6; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(AsStridedTest, AsStridedScatterWithOffset) {
  FileManerger file(g_custom_param.get());
  file.openAppend();
  file << "AsStridedScatterWithOffset ";
  at::Tensor tensor = at::arange(12, at::kFloat);
  at::Tensor src = at::full({2, 2}, 88.0f, at::kFloat);
  at::Tensor result = tensor.as_strided_scatter(src, {2, 2}, {2, 1}, 2);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[2]) << " ";
  file << std::to_string(data[3]) << " ";
  file << std::to_string(data[5]) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
