#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include <fmt/core.h>

#include "calc.h"
#include "cmp.h"
#include "error.h"
#include "timer.h"

constexpr auto M = 1024;
constexpr auto K = 1024;
constexpr auto N = 1024;
constexpr auto REPEAT_TIME = 4;

constexpr auto EPS = 1e-3f;

namespace host {
auto matmul_host(const std::vector<float>& a, const std::vector<float>& b)
    -> std::tuple<std::vector<float>, float> {
    auto c = std::vector<float>(M * N);
    auto elapsed = util::time_cpu(REPEAT_TIME, [&]() {
        std::fill(c.begin(), c.end(), 0.0f);
        for (auto i = 0; i < M; ++i) {
            for (auto k = 0; k < K; ++k) {
                auto a_ik = a[i * K + k];
                for (auto j = 0; j < N; ++j) {
                    c[i * N + j] += a_ik * b[k * N + j];
                }
            }
        }
    });
    return {std::move(c), elapsed};
}
}

namespace device {
constexpr auto TILE = 32;
static_assert(M % TILE == 0 && N % TILE == 0 && K % TILE == 0, "M, N and K must be multiples of TILE");

template <typename Kernel>
auto matmul_device(Kernel kernel, dim3 grid, dim3 block, const float* a_d, const float* b_d, float* c_d)
    -> std::tuple<std::vector<float>, float> {
    auto elapsed = util::time_cuda(REPEAT_TIME, [&]() {
        kernel<<<grid, block>>>(a_d, b_d, c_d, M, N, K);
        CHECK_ERR(cudaGetLastError());
        CHECK_ERR(cudaDeviceSynchronize());
    });

    auto result = std::vector<float>(M * N);
    CHECK_ERR(cudaMemcpy(result.data(), c_d, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return {std::move(result), elapsed};
}

__global__ void matmul_naive(const float* a, const float* b, float* c, int m, int n, int k) {
    const auto row = blockDim.y * blockIdx.y + threadIdx.y;
    const auto col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= m || col >= n) return;

    auto acc = 0.0f;
    for (auto i = 0; i < k; ++i) {
        acc = fmaf(a[row * k + i], b[i * n + col], acc);
    }
    c[row * n + col] = acc;
}

__global__ void matmul_tiled(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float a_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE];

    const auto row = blockIdx.y * TILE + threadIdx.y;
    const auto col = blockIdx.x * TILE + threadIdx.x;
    auto acc = 0.0f;

    for (auto t = 0; t < k / TILE; ++t) {
        a_tile[threadIdx.y][threadIdx.x] = a[row * k + t * TILE + threadIdx.x];
        b_tile[threadIdx.y][threadIdx.x] = b[(t * TILE + threadIdx.y) * n + col];
        __syncthreads();

        for (auto i = 0; i < TILE; ++i) {
            acc = fmaf(a_tile[threadIdx.y][i], b_tile[i][threadIdx.x], acc);
        }
        __syncthreads();
    }

    c[row * n + col] = acc;
}
}

int main() {
    auto a_h = std::vector<float>(M * K);
    auto b_h = std::vector<float>(K * N);
    for (auto i = 0; i < M * K; ++i) {
        a_h[i] = static_cast<float>((i * 2654435761u) % 1000) / 1000.0f;
    }
    for (auto i = 0; i < K * N; ++i) {
        b_h[i] = static_cast<float>((i * 40503u) % 1000) / 1000.0f;
    }

    float* a_d = nullptr;
    float* b_d = nullptr;
    float* c_d = nullptr;
    CHECK_ERR(cudaMalloc((void**)&a_d, a_h.size() * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&b_d, b_h.size() * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&c_d, M * N * sizeof(float)));
    CHECK_ERR(cudaMemcpy(a_d, a_h.data(), a_h.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_ERR(cudaMemcpy(b_d, b_h.data(), b_h.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto [host_result, cpu_elapsed] = host::matmul_host(a_h, b_h);
    fmt::println("[host] elapsed_time={} ms", cpu_elapsed);

    const auto thread_per_block = 32;
    auto naive_block = dim3(thread_per_block, thread_per_block, 1);
    auto naive_grid = dim3(util::ceil_div(N, thread_per_block), util::ceil_div(M, thread_per_block), 1);

    auto [device_result_naive, naive_elapsed] =
        device::matmul_device(device::matmul_naive, naive_grid, naive_block, a_d, b_d, c_d);
    auto [naive_ok, naive_max_idx] = util::cmp_vec(device_result_naive, host_result, EPS);
    auto naive_max_diff = std::fabs(host_result[naive_max_idx] - device_result_naive[naive_max_idx]);
    fmt::println("[device::naive] elapsed_time={} ms, result=[{}, {}], diff={:.10e}", naive_elapsed,
                host_result[naive_max_idx], device_result_naive[naive_max_idx], naive_max_diff);
    if (!naive_ok) return -1;

    auto tiled_block = dim3(device::TILE, device::TILE, 1);
    auto tiled_grid = dim3(N / device::TILE, M / device::TILE, 1);

    auto [device_result_tiled, tiled_elapsed] =
        device::matmul_device(device::matmul_tiled, tiled_grid, tiled_block, a_d, b_d, c_d);
    auto [tiled_ok, tiled_max_idx] = util::cmp_vec(device_result_tiled, host_result, EPS);
    auto tiled_max_diff = std::fabs(host_result[tiled_max_idx] - device_result_tiled[tiled_max_idx]);
    fmt::println("[device::tiled] elapsed_time={} ms, result=[{}, {}], diff={:.10e}", tiled_elapsed,
                host_result[tiled_max_idx], device_result_tiled[tiled_max_idx], tiled_max_diff);
    if (!tiled_ok) return -1;

    CHECK_ERR(cudaFree(a_d));
    CHECK_ERR(cudaFree(b_d));
    CHECK_ERR(cudaFree(c_d));

    return 0;
}
