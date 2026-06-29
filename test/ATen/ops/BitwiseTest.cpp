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

class BitwiseTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(BitwiseTest, BitwiseRightShift) {
  FileManerger file(g_custom_param.get());
  file.createFile();
  file << "BitwiseRightShift ";
  at::Tensor input = at::ones({2, 3}, at::kInt).fill_(8);
  at::Tensor result = input.bitwise_right_shift(2);
  int* data = result.data_ptr<int>();
  file << std::to_string(data[0]) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
