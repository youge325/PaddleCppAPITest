#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

TEST(TensorBodyTest, PtrTest) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor t = at::ones({2, 3}, options);

  const float* const_ptr = t.const_data_ptr<float>();
  EXPECT_NE(const_ptr, nullptr);

  const float* const_type_ptr = t.const_data_ptr<const float>();
  EXPECT_NE(const_type_ptr, nullptr);

  const void* void_const_ptr = t.const_data_ptr();
  EXPECT_NE(void_const_ptr, nullptr);

  float* data_ptr = t.data_ptr<float>();
  EXPECT_NE(data_ptr, nullptr);

  void* void_data_ptr = t.data_ptr();
  EXPECT_NE(void_data_ptr, nullptr);

  float* mut_ptr = t.mutable_data_ptr<float>();
  EXPECT_NE(mut_ptr, nullptr);

  void* void_mut_ptr = t.mutable_data_ptr();
  EXPECT_NE(void_mut_ptr, nullptr);

  // We should write to file to check values
  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.createFile();
  file << "PtrTest ";
  file << "const_ptr[0]: " + std::to_string(const_ptr[0]) + "\n";
  file << "const_type_ptr[0]: " + std::to_string(const_type_ptr[0]) + "\n";
  file << "void_const_ptr_equal: "
       << std::to_string(void_const_ptr == static_cast<const void*>(const_ptr))
       << "\n";
  file << "data_ptr_equal: "
       << std::to_string(static_cast<void*>(data_ptr) == void_data_ptr) << "\n";
  file << "const_type_ptr_equal: "
       << std::to_string(static_cast<const void*>(const_type_ptr) ==
                         static_cast<const void*>(const_ptr))
       << "\n";

  mut_ptr[0] = 5.0f;
  file << "mut_ptr[0]: " + std::to_string(mut_ptr[0]) + "\n";

  file << "\n";
  file.saveFile();
  // Type mismatch crash check?
  // const int* int_ptr = t.const_data_ptr<int>(); // Might throw
}
