// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab

#include <immintrin.h>
#include <omp.h>

#include <cstring>

#include "gemm.h"

namespace swiftware::hpp {

static int nearestMultiple(int n, int s) {
    if (n % s == 0) {
        return n;
    }

    return n + (s - n % s);
}

void gemmEfficientParallel(int m, int n, int k, const float* A, const float* B,
                           float* C, ScheduleParams Sp) {
    // if A is short and wide
    if (m * 100 <= k) {
        // Process in blocks for better cache utilization
        const int BLOCK_SIZE = Sp.TileSize1;  // Tune this value
        int i, j, l, ii, ll, jj;

        for (ii = 0; ii < m; ii += BLOCK_SIZE) {
            for (ll = 0; ll < k; ll += BLOCK_SIZE) {
#pragma omp parallel for schedule(static)
                for (jj = 0; jj < n; jj += BLOCK_SIZE) {
                    const int iLim = std::min(ii + BLOCK_SIZE, m);
                    const int kLim = std::min(ll + BLOCK_SIZE, k);
                    const int jLim = std::min(jj + BLOCK_SIZE, n);
                    for (i = ii; i < iLim; i++) {
                        for (l = ll; l < kLim; l++) {
                            auto aVec = _mm256_broadcast_ss(&A[i * k + l]);

                            for (j = jj; j < jLim - 7; j += 8) {
                                auto bVec = _mm256_loadu_ps(&B[l * n + j]);
                                auto cVec = _mm256_loadu_ps(&C[i * n + j]);
                                auto prodVec = _mm256_mul_ps(aVec, bVec);
                                cVec = _mm256_add_ps(cVec, prodVec);
                                _mm256_storeu_ps(&C[i * n + j], cVec);
                            }

                            // Handle remaining elements
                            for (; j < jLim; j++) {
                                C[i * n + j] += A[i * k + l] * B[l * n + j];
                            }
                        }
                    }
                }
            }
        }
        return;  // Skip the tiled implementation
    }

    if (m >= k * 100) {
        const int BLOCK_SIZE = Sp.TileSize1;  // Tune this value
        int i, j, l, ii, ll, jj;
        int iLim;
        int kLim;
        int jLim;

        // Since m >> k, parallelize over m dimension and keep k,n blocks
        // together Reorder loops to maximize cache reuse of B matrix

#pragma omp parallel default(none) private(i, j, l, ii, ll, jj, iLim, kLim, \
                                               jLim)                        \
    shared(A, B, C, m, n, k, BLOCK_SIZE)
        {
            for (ii = omp_get_thread_num() * BLOCK_SIZE; ii < m;
                 ii += BLOCK_SIZE * omp_get_num_threads()) {
                iLim = std::min(ii + BLOCK_SIZE, m);

                for (ll = 0; ll < k; ll += BLOCK_SIZE) {
                    kLim = std::min(ll + BLOCK_SIZE, k);

                    for (i = ii; i < iLim; i++) {
                        for (l = ll; l < kLim; l++) {
                            auto aVec = _mm256_broadcast_ss(&A[i * k + l]);

                            for (j = 0; j < n - 7; j += 8) {
                                auto bVec = _mm256_loadu_ps(&B[l * n + j]);
                                auto cVec = _mm256_loadu_ps(&C[i * n + j]);
                                auto prodVec = _mm256_mul_ps(aVec, bVec);
                                cVec = _mm256_add_ps(cVec, prodVec);
                                _mm256_storeu_ps(&C[i * n + j], cVec);
                            }

                            // Handle remaining elements
                            for (; j < n; j++) {
                                C[i * n + j] += A[i * k + l] * B[l * n + j];
                            }
                        }
                    }
                }
            }
        }
        return;  // Skip the tiled implementation
    }

    const int s1 = Sp.TileSize1;
    const int s2 = Sp.TileSize2;
    const size_t ARRAY_SIZE = s1 * s1;

    const int mAdj = nearestMultiple(m, s1);
    const int nAdj = nearestMultiple(n, s1);
    const int kAdj = nearestMultiple(k, s1);

    float* Apad;
    float* Bpad;
    float* Cpad;

    Apad = static_cast<float*>(_mm_malloc(mAdj * kAdj * sizeof(float), 64));
    Bpad = static_cast<float*>(_mm_malloc(kAdj * nAdj * sizeof(float), 64));
    Cpad = static_cast<float*>(_mm_malloc(mAdj * nAdj * sizeof(float), 64));

    std::fill(Cpad, Cpad + mAdj * nAdj, 0.0f);
    std::fill(Apad, Apad + mAdj * kAdj, 0.0f);
    std::fill(Bpad, Bpad + kAdj * nAdj, 0.0f);

    for (int i = 0; i < m; i++) {
        memcpy((void*)(Apad + i * kAdj), (void*)(A + i * k), k * sizeof(float));
    }

    for (int i = 0; i < k; i++) {
        memcpy((void*)(Bpad + i * nAdj), (void*)(B + i * n), n * sizeof(float));
    }

    // Step 2: loop block-wise
    const size_t numCRowBlocks = mAdj / s1;
    const size_t numCColBlocks = nAdj / s1;
    const size_t numABBlocks = kAdj / s1;

#pragma omp parallel num_threads(Sp.NumThreads)
    {
        auto* Bblock = (float*)(_mm_malloc(sizeof(float) * ARRAY_SIZE, 64));
        auto* Cblock = (float*)(_mm_malloc(ARRAY_SIZE * sizeof(float), 64));

#pragma omp for collapse(2) schedule(static, Sp.ChunkSize)
        for (int cr = 0; cr < numCRowBlocks; ++cr) {
            for (int cc = 0; cc < numCColBlocks; ++cc) {
                std::fill(Cblock, Cblock + ARRAY_SIZE, 0.0f);

                // Calculate the C block
                for (int ac = 0, br = 0; ac < numABBlocks; ++ac, ++br) {
                    // Get the B block
                    for (int r = 0; r < s1; ++r) {
                        for (int c = 0; c < s1; c += 8) {
                            auto load_vec =
                                _mm256_loadu_ps(&Bpad[(((br * s1) + r) * nAdj) +
                                                      (cc * s1 + c)]);
                            _mm256_store_ps(&Bblock[r * s1 + c], load_vec);
                        }
                    }

                    // A is stored row-wise and B is stored column-wise
                    for (int i = 0; i < s1; i += 8) {
                        for (int l = 0; l < s1; l++) {
                            auto aVec0 = _mm256_broadcast_ss(
                                &Apad[(((cr * s1) + i) * kAdj) +
                                      (ac * s1 + l)]);
                            auto aVec1 = _mm256_broadcast_ss(
                                &Apad[(((cr * s1) + i + 1) * kAdj) +
                                      (ac * s1 + l)]);
                            auto aVec2 = _mm256_broadcast_ss(
                                &Apad[(((cr * s1) + i + 2) * kAdj) +
                                      (ac * s1 + l)]);
                            auto aVec3 = _mm256_broadcast_ss(
                                &Apad[(((cr * s1) + i + 3) * kAdj) +
                                      (ac * s1 + l)]);
                            auto aVec4 = _mm256_broadcast_ss(
                                &Apad[(((cr * s1) + i + 4) * kAdj) +
                                      (ac * s1 + l)]);
                            auto aVec5 = _mm256_broadcast_ss(
                                &Apad[(((cr * s1) + i + 5) * kAdj) +
                                      (ac * s1 + l)]);
                            auto aVec6 = _mm256_broadcast_ss(
                                &Apad[(((cr * s1) + i + 6) * kAdj) +
                                      (ac * s1 + l)]);
                            auto aVec7 = _mm256_broadcast_ss(
                                &Apad[(((cr * s1) + i + 7) * kAdj) +
                                      (ac * s1 + l)]);

                            for (int j = 0; j < s1; j += 8) {
                                auto bVec = _mm256_load_ps(&Bblock[l * s1 + j]);

                                auto cVec0 =
                                    _mm256_load_ps(&Cblock[i * s1 + j]);
                                auto cVec1 =
                                    _mm256_load_ps(&Cblock[(i + 1) * s1 + j]);
                                auto cVec2 =
                                    _mm256_load_ps(&Cblock[(i + 2) * s1 + j]);
                                auto cVec3 =
                                    _mm256_load_ps(&Cblock[(i + 3) * s1 + j]);
                                auto cVec4 =
                                    _mm256_load_ps(&Cblock[(i + 4) * s1 + j]);
                                auto cVec5 =
                                    _mm256_load_ps(&Cblock[(i + 5) * s1 + j]);
                                auto cVec6 =
                                    _mm256_load_ps(&Cblock[(i + 6) * s1 + j]);
                                auto cVec7 =
                                    _mm256_load_ps(&Cblock[(i + 7) * s1 + j]);

                                auto prodVec0 = _mm256_mul_ps(aVec0, bVec);
                                auto prodVec1 = _mm256_mul_ps(aVec1, bVec);
                                auto prodVec2 = _mm256_mul_ps(aVec2, bVec);
                                auto prodVec3 = _mm256_mul_ps(aVec3, bVec);
                                auto prodVec4 = _mm256_mul_ps(aVec4, bVec);
                                auto prodVec5 = _mm256_mul_ps(aVec5, bVec);
                                auto prodVec6 = _mm256_mul_ps(aVec6, bVec);
                                auto prodVec7 = _mm256_mul_ps(aVec7, bVec);

                                cVec0 = _mm256_add_ps(cVec0, prodVec0);
                                cVec1 = _mm256_add_ps(cVec1, prodVec1);
                                cVec2 = _mm256_add_ps(cVec2, prodVec2);
                                cVec3 = _mm256_add_ps(cVec3, prodVec3);
                                cVec4 = _mm256_add_ps(cVec4, prodVec4);
                                cVec5 = _mm256_add_ps(cVec5, prodVec5);
                                cVec6 = _mm256_add_ps(cVec6, prodVec6);
                                cVec7 = _mm256_add_ps(cVec7, prodVec7);

                                _mm256_store_ps(&Cblock[i * s1 + j], cVec0);
                                _mm256_store_ps(&Cblock[(i + 1) * s1 + j],
                                                cVec1);
                                _mm256_store_ps(&Cblock[(i + 2) * s1 + j],
                                                cVec2);
                                _mm256_store_ps(&Cblock[(i + 3) * s1 + j],
                                                cVec3);
                                _mm256_store_ps(&Cblock[(i + 4) * s1 + j],
                                                cVec4);
                                _mm256_store_ps(&Cblock[(i + 5) * s1 + j],
                                                cVec5);
                                _mm256_store_ps(&Cblock[(i + 6) * s1 + j],
                                                cVec6);
                                _mm256_store_ps(&Cblock[(i + 7) * s1 + j],
                                                cVec7);
                            }
                        }
                    }
                }

                // Store Cblock values back to C
                for (int r = 0; r < s1; ++r) {
                    for (int c = 0; c < s1; c += 8) {
                        auto store_vec = _mm256_load_ps(&Cblock[(r * s1) + c]);
                        _mm256_storeu_ps(
                            &Cpad[(((cr * s1) + r) * nAdj) + (cc * s1 + c)],
                            store_vec);
                    }
                }
            }
        }

        _mm_free(Bblock);
        _mm_free(Cblock);
    }

    for (int i = 0; i < m; ++i) {
        memcpy((void*)(C + i * n), (void*)(Cpad + i * nAdj), n * sizeof(float));
    }

    _mm_free((void*)Apad);
    _mm_free((void*)Bpad);
    _mm_free(Cpad);
}

}  // namespace swiftware::hpp
