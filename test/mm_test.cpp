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

TEST_P(MMTest, TestMultiRowDecomp) {
    gemmBaseline(m, n, k, A->data.data(), B->data.data(), E->data.data());
    float temp = gemmGpuMultiRowDecomp(
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

TEST_P(MMTest, TestSingleElementDecomp) {
    gemmBaseline(m, n, k, A->data.data(), B->data.data(), E->data.data());
    float temp = gemmGpuSingleElementDecomp(
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
    float temp = gemmGpuTwoDimTile(
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
        MMTestParams(10000, 10000, 64, 64, 32, 8, 1, "TallSkinny"),
        MMTestParams(512, 512, 128, 64, 32, 8, 1, "TallSkinny"),
        MMTestParams(10000, 10000, 256, 64, 32, 8, 1, "TallSkinny"),
        MMTestParams(50, 50, 10000, 64, 32, 8, 1, "ShortWide"),
        MMTestParams(2048, 2048, 2048, 64, 32, 8, 1, "MediumSquare"),
        MMTestParams(256, 256, 256, 64, 32, 8, 1, "SmallSquare"),
        MMTestParams(128, 128, 128, 64, 32, 8, 1, "TinySquare"),
        MMTestParams(1024, 512, 2048, 64, 32, 8, 1, "RectangularMix1"),
        MMTestParams(2048, 1024, 512, 64, 32, 8, 1, "RectangularMix2"),
        MMTestParams(4096, 512, 1024, 64, 32, 8, 1, "RectangularMix3"),
        MMTestParams(8192, 256, 512, 64, 32, 8, 1, "VeryTallSkinny"),
        MMTestParams(256, 8192, 512, 64, 32, 8, 1, "VeryShortWide"),
        MMTestParams(777, 888, 999, 64, 32, 8, 1, "OddSizes"),
        MMTestParams(15000, 15000, 32, 64, 32, 8, 1, "ExtremelyTallSkinny"),
        MMTestParams(32, 32, 15000, 64, 32, 8, 1, "ExtremelyShortWide"),
        MMTestParams(1, 1, 1, 64, 32, 8, 1, "SingleElement"),
        MMTestParams(1, 1000, 1, 64, 32, 8, 1, "SingleRowColumn"),
        MMTestParams(1000, 1, 1, 64, 32, 8, 1, "SingleColumnRow"),
        MMTestParams(1, 1, 1000, 64, 32, 8, 1, "SingleRowColumnLong"),
        MMTestParams(3, 5, 7, 64, 32, 8, 1, "SmallPrimes"),
        MMTestParams(2047, 2047, 2047, 64, 32, 8, 1, "AlmostPowerTwo"),
        MMTestParams(1024, 1024, 1024, 64, 32, 8, 1, "Square")));

}  // namespace swiftware::hpp