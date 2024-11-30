// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#include <gtest/gtest.h>
#include "gemm.h"

namespace swiftware::hpp {
  TEST(MMTest, SmallTest) {
    int m = 2;
    int n = 2;
    int k = 2;
    // TODO generate random sparse matrices
    float A[4] = {1, 2, 3, 4};
    float B[4] = {1, 2, 3, 4};
    float C[4] = {0, 0, 0, 0};
    swiftware::hpp::gemmEfficientParallel(m, n, k, A, B, C, swiftware::hpp::ScheduleParams(32, 32, 8, 1));
    float expected[4] = {7, 10, 15, 22};
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        EXPECT_EQ(C[i * n + j], expected[i * n + j]);
      }
    }
  }

  // TODO add more tests for GEMM OpenCL

}