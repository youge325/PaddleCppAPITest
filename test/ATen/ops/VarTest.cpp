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

class VarTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(VarTest, VarDim) {
  FileManerger file(g_custom_param.get());
  file.createFile();
  file << "VarDim ";
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  input.fill_(1.0f);
  input.data_ptr<float>()[1] = 3.0f;
  at::Tensor result = input.var(1);
  file << std::to_string(result.dim()) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(VarTest, VarAll) {
  FileManerger file(g_custom_param.get());
  file.openAppend();
  file << "VarAll ";
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  input.fill_(1.0f);
  input.data_ptr<float>()[1] = 3.0f;
  at::Tensor result = input.var(true);
  file << std::to_string(result.dim()) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(VarTest, VarDims) {
  FileManerger file(g_custom_param.get());
  file.openAppend();
  file << "VarDims ";
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  input.fill_(1.0f);
  input.data_ptr<float>()[1] = 3.0f;
  at::Tensor result = input.var({1}, true, true);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(VarTest, VarCorrection) {
  FileManerger file(g_custom_param.get());
  file.openAppend();
  file << "VarCorrection ";
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  input.fill_(1.0f);
  input.data_ptr<float>()[1] = 3.0f;
  at::Tensor result = input.var({1}, 1.0, true);
  file << std::to_string(result.dim()) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
