// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab

#include <gtest/gtest.h>

#include <string>
#include <tuple>

#include "gemm.h"
#include "utils.h"

namespace swiftware::hpp {

using MMTestParams = std::tuple<int, int, int, int, int, int, int, std::string>;

// lets parameterize the test
// m, n, k, t1, t2, cs, nt, name
class MMTest : public ::testing::TestWithParam<MMTestParams> {
protected:
    int m, n, k, t1, t2, cs, nt;
    std::string name;
    DenseMatrix *A, *B, *C, *E;

    static void gemmBaseline(int m, int n, int k, const float* A,
                             const float* B, float* C) {
        for (int i = 0; i < m; ++i) {
            for (int l = 0; l < k; ++l) {
                for (int j = 0; j < n; ++j) {
                    C[i * n + j] += A[i * k + l] * B[l * n + j];
                }
            }
        }
    }

    void SetUp() override {
        // Code here will be called immediately after the constructor (right
        // before each test).
        std::tie(m, n, k, t1, t2, cs, nt, name) = GetParam();
        std::cout << "Running " << name << std::endl;
        A = new DenseMatrix(m, k);
        B = new DenseMatrix(k, n);
        C = new DenseMatrix(m, n);
        E = new DenseMatrix(m, n);

        for (int i = 0; i < m * k; ++i) {
            A->data[i] = (float)(i + 1.0) / (m * k);
        }
        for (int i = 0; i < k * n; ++i) {
            B->data[i] = (float)(i + 1.0) / (k * n);
        }
        for (int i = 0; i < m * n; ++i) {
            E->data[i] = 0.0;
            C->data[i] = 0.0;
        }
    }

    void TearDown() override {
        // Code here will be called immediately after each test (right before
        // the destructor).
        delete A;
        delete B;
        delete C;
        delete E;
    }
};

TEST_P(MMTest, TestSingleRowDecomp) {
    gemmBaseline(m, n, k, A->data.data(), B->data.data(), E->data.data());
    float temp = gemmGpuSingleRowDecomp(
        m, n, k, A->data.data(), B->data.data(), C->data.data(),
        swiftware::hpp::ScheduleParams(1024, t2, cs, nt));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_LT(abs((C->data[i * n + j] - E->data[i * n + j]) /
                          E->data[i * n + j]),
                      1e-3);
        }
    }
}

TEST_P(MMTest, Test1DTile) {
    gemmBaseline(m, n, k, A->data.data(), B->data.data(), E->data.data());
    float temp = gemmGpuSingleRowDecomp(
        m, n, k, A->data.data(), B->data.data(), C->data.data(),
        swiftware::hpp::ScheduleParams(1024, t2, cs, nt));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_LT(abs((C->data[i * n + j] - E->data[i * n + j]) /
                          E->data[i * n + j]),
                      1e-3);
        }
    }
}

TEST_P(MMTest, Test2DTile) {
    gemmBaseline(m, n, k, A->data.data(), B->data.data(), E->data.data());
    float temp = gemmGpuSingleRowDecomp(
        m, n, k, A->data.data(), B->data.data(), C->data.data(),
        swiftware::hpp::ScheduleParams(1024, t2, cs, nt));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_LT(abs((C->data[i * n + j] - E->data[i * n + j]) /
                          E->data[i * n + j]),
                      1e-3);
        }
    }
}

// m, n, k, blockDim1, t2, cs, nt, srPercentage, name
INSTANTIATE_TEST_SUITE_P(
    MMTest, MMTest,
    ::testing::Values(
        MMTestParams(512, 512, 512, 64, 32, 8, 1, "SameSizePwrOf2"),
        MMTestParams(4096, 4096, 4096, 64, 32, 8, 1, "FullSize"),
        MMTestParams(500, 500, 500, 64, 32, 8, 1, "SameSizeNonPwrOf2"),
        MMTestParams(10000, 10000, 50, 64, 32, 8, 1, "TallSkinny"),
        MMTestParams(50, 50, 10000, 64, 32, 8, 1, "ShortWide"),
        MMTestParams(1024, 1024, 1024, 64, 32, 8, 1, "Square")));

}  // namespace swiftware::hpp