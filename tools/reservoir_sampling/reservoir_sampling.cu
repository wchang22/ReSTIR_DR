#include <cstdio>
#include <iostream>
#include <exception>
#include <string>

struct Reservoir {
    float* weight_sum;
    uint32_t* pixel_idx;
    float* u;
    float* v;
    float* w;
    float* sample_value_r;
    float* sample_value_g;
    float* sample_value_b;
};

struct Sample {
    float* weight;
    float* u;
    float* v;
    float* w;
    float* sample_value_r;
    float* sample_value_g;
    float* sample_value_b;
    uint32_t* pixel_idx;
    float* rand_num;
    bool* valid;
};

constexpr size_t N_samples = 100;
constexpr size_t N_reservoirs = 5;
constexpr size_t N_samples_per_reservoir = N_samples / N_reservoirs;

__device__ void lock(uint32_t* lock) {
    while (atomicExch(lock, 1) == 1) {}
}

__device__ void unlock(uint32_t* lock) {
    atomicExch(lock, 0);
}

extern "C" __noinline__
__device__ void reservoir_sampling(
    uint32_t idx,
    float rand_num,
    uint32_t* locks,
    uint64_t* weight_sum_index_ptr,

    uint32_t* target_pixel_idx_ptr,
    float* target_u_ptr,
    float* target_v_ptr,
    float* target_w_ptr,
    float* target_sample_value_r_ptr,
    float* target_sample_value_g_ptr,
    float* target_sample_value_b_ptr,

    float sample_weight,
    uint32_t sample_pixel_idx,
    float sample_u,
    float sample_v,
    float sample_w,
    float sample_value_r,
    float sample_value_g,
    float sample_value_b
) {
    uint32_t tIdx = blockIdx.x * blockDim.x + threadIdx.x;

    bool select_sample = false;
    while (true) {
        uint64_t weight_sum_index = weight_sum_index_ptr[idx];
        float target_weight_sum = __uint_as_float(weight_sum_index);
        uint32_t target_index = weight_sum_index >> 32;

        float new_weight_sum = target_weight_sum + sample_weight;
        select_sample = rand_num * target_weight_sum < sample_weight;
        uint64_t new_index = select_sample ? tIdx : target_index;
        uint64_t new_weight_sum_index = (new_index << 32) | ((uint64_t) __float_as_uint(new_weight_sum));

        uint64_t old = atomicCAS((unsigned long long*) weight_sum_index_ptr + idx,
            (unsigned long long) weight_sum_index,
            (unsigned long long) new_weight_sum_index);
        if (old == weight_sum_index) {
            break;
        }
    }

    if (select_sample) {
        lock(locks + idx);
        uint32_t target_index = weight_sum_index_ptr[idx] >> 32;
        if (target_index == tIdx) {
            target_u_ptr[idx] = sample_u;
            target_v_ptr[idx] = sample_v;
            target_w_ptr[idx] = sample_w;
            target_sample_value_r_ptr[idx] = sample_value_r;
            target_sample_value_g_ptr[idx] = sample_value_g;
            target_sample_value_b_ptr[idx] = sample_value_b;
            target_pixel_idx_ptr[idx] = sample_pixel_idx;
        }
        unlock(locks + idx);
    }
}

extern "C"
__global__ void my_kernel(Reservoir reservoir, Sample sample, uint32_t* locks, uint64_t* weight_sum_index, uint32_t* idx) {
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!sample.valid[tIdx]) {
        return;
    }

    reservoir_sampling(
        idx[tIdx],
        sample.rand_num[tIdx],
        locks,
        weight_sum_index,

        reservoir.pixel_idx,
        reservoir.u,
        reservoir.v,
        reservoir.w,
        reservoir.sample_value_r,
        reservoir.sample_value_g,
        reservoir.sample_value_b,

        sample.weight[tIdx],
        sample.pixel_idx[tIdx],
        sample.u[tIdx],
        sample.v[tIdx],
        sample.w[tIdx],
        sample.sample_value_r[tIdx],
        sample.sample_value_g[tIdx],
        sample.sample_value_b[tIdx]
    );
}

int main() {
    Reservoir reservoir;
    Sample sample;
    uint32_t* locks;
    uint32_t* idx;
    uint64_t* weight_sum_index;

    cudaMallocManaged(&reservoir.weight_sum, N_reservoirs*sizeof(float));
    cudaMallocManaged(&reservoir.pixel_idx, N_reservoirs*sizeof(uint32_t));
    cudaMallocManaged(&reservoir.u, N_reservoirs*sizeof(float));
    cudaMallocManaged(&reservoir.v, N_reservoirs*sizeof(float));
    cudaMallocManaged(&reservoir.w, N_reservoirs*sizeof(float));
    cudaMallocManaged(&reservoir.sample_value_r, N_reservoirs*sizeof(float));
    cudaMallocManaged(&reservoir.sample_value_g, N_reservoirs*sizeof(float));
    cudaMallocManaged(&reservoir.sample_value_b, N_reservoirs*sizeof(float));

    cudaMallocManaged(&locks, N_reservoirs*sizeof(uint32_t));
    cudaMallocManaged(&idx, N_samples*sizeof(uint32_t));
    cudaMallocManaged(&weight_sum_index, N_reservoirs*sizeof(uint64_t));

    cudaMallocManaged(&sample.weight, N_samples*sizeof(float));
    cudaMallocManaged(&sample.pixel_idx, N_samples*sizeof(uint32_t));
    cudaMallocManaged(&sample.u, N_samples*sizeof(float));
    cudaMallocManaged(&sample.v, N_samples*sizeof(float));
    cudaMallocManaged(&sample.w, N_samples*sizeof(float));
    cudaMallocManaged(&sample.sample_value_r, N_samples*sizeof(float));
    cudaMallocManaged(&sample.sample_value_g, N_samples*sizeof(float));
    cudaMallocManaged(&sample.sample_value_b, N_samples*sizeof(float));
    cudaMallocManaged(&sample.rand_num, N_samples*sizeof(float));
    cudaMallocManaged(&sample.valid, N_samples*sizeof(bool));

    for (uint32_t i = 0; i < N_reservoirs; i++) {
        reservoir.weight_sum[i] = {};
        reservoir.pixel_idx[i] = {};
        reservoir.u[i] = {};
        reservoir.v[i] = {};
        reservoir.w[i] = {};
        reservoir.sample_value_r[i] = {};
        reservoir.sample_value_g[i] = {};
        reservoir.sample_value_b[i] = {};

        locks[i] = 0;
        weight_sum_index[i] = 0;
    }

    int invalid_samples[] = { 2, 31, 40, 72, 94 };

    for (uint32_t i = 0; i < N_samples; i++) {
        idx[i] = i / N_samples_per_reservoir;
        float f = i;
        sample.weight[i] = f,
        sample.rand_num[i] = 0.5,
        sample.u[i] = f;
        sample.v[i] = f + 1;
        sample.w[i] = f + 2;
        sample.sample_value_r[i] = f + 3;
        sample.sample_value_g[i] = f + 4;
        sample.sample_value_b[i] = f + 5;
        sample.pixel_idx[i] = i,
        sample.valid[i] = 1;
        if (i == invalid_samples[idx[i]]) {
            sample.valid[i] = 0;
        }
    }

    my_kernel<<<1, N_samples>>>(reservoir, sample, locks, weight_sum_index, idx);
    cudaDeviceSynchronize();

    for (uint32_t i = 0; i < N_reservoirs; i++) {
        uint32_t wsi = weight_sum_index[i];
        reservoir.weight_sum[i] = *((float*)&wsi);

        std::cout << "----------------- Reservoir " << i << " ------------------" << std::endl;
        std::cout << "weight_sum = " << reservoir.weight_sum[i] << std::endl;
        std::cout << "pixel_idx = " << reservoir.pixel_idx[i] << std::endl;
        std::cout << "uvw = [" << std::endl;
        std::cout << "  " << reservoir.u[i] << std::endl;
        std::cout << "  " << reservoir.v[i] << std::endl;
        std::cout << "  " << reservoir.w[i] << std::endl;
        std::cout << "]" << std::endl;
        std::cout << "sample_value = [" << std::endl;
        std::cout << "  " << reservoir.sample_value_r[i] << std::endl;
        std::cout << "  " << reservoir.sample_value_g[i] << std::endl;
        std::cout << "  " << reservoir.sample_value_b[i] << std::endl;
        std::cout << "]" << std::endl;

        float expected_weight_sum = ((2 * i + 1) * N_samples_per_reservoir - 1) * N_samples_per_reservoir / 2.f;
        expected_weight_sum -= invalid_samples[i];

        if (reservoir.weight_sum[i] != expected_weight_sum) {
            throw std::runtime_error("Incorrect weight sum at reservoir " + std::to_string(i));
        }
        if (reservoir.pixel_idx[i] < i * N_samples_per_reservoir || reservoir.pixel_idx[i] >= (i + 1) * N_samples_per_reservoir) {
            throw std::runtime_error("Invalid pixel_idx at reservoir " + std::to_string(i));
        }
        if (reservoir.pixel_idx[i] != reservoir.u[i] &&
            reservoir.pixel_idx[i] + 1 != reservoir.v[i] &&
            reservoir.pixel_idx[i] + 2 != reservoir.w[i] &&
            reservoir.pixel_idx[i] + 3 != reservoir.sample_value_r[i] &&
            reservoir.pixel_idx[i] + 4 != reservoir.sample_value_g[i] &&
            reservoir.pixel_idx[i] + 5 != reservoir.sample_value_b[i]
        ) {
            throw std::runtime_error("Invalid pixel_idx/uvw/sample_value at reservoir " + std::to_string(i));
        }
    }

    std::cout << "Success!" << std::endl;

    cudaFree(reservoir.weight_sum);
    cudaFree(reservoir.pixel_idx);
    cudaFree(reservoir.u);
    cudaFree(reservoir.v);
    cudaFree(reservoir.w);
    cudaFree(reservoir.sample_value_r);
    cudaFree(reservoir.sample_value_g);
    cudaFree(reservoir.sample_value_b);
    
    cudaFree(sample.weight);
    cudaFree(sample.pixel_idx);
    cudaFree(sample.u);
    cudaFree(sample.v);
    cudaFree(sample.w);
    cudaFree(sample.sample_value_r);
    cudaFree(sample.sample_value_g);
    cudaFree(sample.sample_value_b);
    cudaFree(sample.rand_num);
    cudaFree(sample.valid);

    cudaFree(locks);
    cudaFree(idx);
    cudaFree(weight_sum_index);

    return 0;
}