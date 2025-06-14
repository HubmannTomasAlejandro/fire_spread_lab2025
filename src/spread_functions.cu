#include "spread_functions.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <cstdint>
#include <iostream>
#include <cstddef>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <array>


#include "fires.hpp"
#include "landscape.hpp"
#include "constants.hpp"

__constant__ float DEV_ANGLES[8];
__constant__ int DEV_MOVES[8][2];


CUDA_CALLABLE float spread_probability_scalar(
    const Cell& burning,
    const Cell& neighbor,
    float distance,
    float elevation_mean,
    float elevation_sd,
    float angle,
    float upper_limit,
    const SimulationParams& params
) {
    std::array<float,4> veg_pred = {0.f,0.f,0.f,0.f};
    veg_pred[static_cast<int>(VegetationType::SUBALPINE)] = params.subalpine_pred;
    veg_pred[static_cast<int>(VegetationType::WET)] = params.wet_pred;
    veg_pred[static_cast<int>(VegetationType::DRY)] = params.dry_pred;

    if (!neighbor.burnable) return 0.0f;

    float slope = (neighbor.elevation - burning.elevation) / distance;
    float slope_term = sin(atan(slope));
    float wind_term = cos(angle - burning.wind_direction);
    float elev_term = (neighbor.elevation - elevation_mean) / elevation_sd;

    float linear_pred = params.independent_pred;
    linear_pred += veg_pred[static_cast<int>(neighbor.vegetation_type)];
    linear_pred += params.fwi_pred * neighbor.fwi;
    linear_pred += params.aspect_pred * neighbor.aspect;
    linear_pred += params.wind_pred * wind_term;
    linear_pred += params.elevation_pred * elev_term;
    linear_pred += params.slope_pred * slope_term;

    return upper_limit / (1.0f + exp(-linear_pred));
}

__global__ void fire_spread_kernel(
    const Cell* landscape,
    unsigned int* burning_state,
    size_t width, size_t height,
    float distance, float elevation_mean, float elevation_sd,
    float upper_limit, const SimulationParams params,
    unsigned int current_iteration,
    bool* active_flag
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    size_t idx = y * width + x;

    // Only process cells that burned in previous iteration
    if (burning_state[idx] != current_iteration - 1) return;

    unsigned long seed =
        (blockIdx.x * 2654435761) ^
        (threadIdx.x * 2246822519) ^
        (current_iteration * 3266489917) ^
        (clock64() % 7919);
    curandStatePhilox4_32_10_t rng_state;
    curand_init(seed, idx, 0, &rng_state);

    Cell burning_cell = landscape[idx];
    bool thread_active = false;

    for (int i = 0; i < 8; i++) {
        int nx = x + DEV_MOVES[i][0];
        int ny = y + DEV_MOVES[i][1];
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

        size_t neighbor_idx = ny * width + nx;
        if (burning_state[neighbor_idx] != 0) continue; // Skip already burned

        Cell neighbor_cell = landscape[neighbor_idx];
        if (!neighbor_cell.burnable) continue;

        float prob = spread_probability_scalar(
            burning_cell, neighbor_cell, distance,
            elevation_mean, elevation_sd, DEV_ANGLES[i],
            upper_limit, params
        );

        float rand_val = curand_uniform(&rng_state);

        if (rand_val < prob) {
            burning_state[neighbor_idx] = current_iteration;
            thread_active = true;
        }
    }

    if (thread_active) {
        *active_flag = true;
    }
}

__global__ void setup_rng_kernel(curandState* state, unsigned long seed, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

void simulate_fire_cuda(
    const Landscape& landscape,
    const std::vector<IgnitionPair>& ignition_cells,
    SimulationParams params,
    float distance,
    float elevation_mean,
    float elevation_sd,
    float upper_limit,
    Fire& result
) {
    size_t width = landscape.width;
    size_t height = landscape.height;
    size_t num_cells = width * height;

    Cell* d_landscape;
    cudaMalloc(&d_landscape, num_cells * sizeof(Cell));
    cudaMemcpy(d_landscape, landscape.cells, num_cells * sizeof(Cell), cudaMemcpyHostToDevice);

    unsigned int* d_burning_state;
    cudaMalloc(&d_burning_state, num_cells * sizeof(unsigned int));
    cudaMemset(d_burning_state, 0, num_cells * sizeof(unsigned int));

    unsigned int value = 1;
    for (const auto& cell : ignition_cells) {
        size_t idx = cell.second * width + cell.first;
        cudaMemcpy(d_burning_state + idx, &value, sizeof(unsigned int), cudaMemcpyHostToDevice);
    }

    cudaMemcpyToSymbol(DEV_ANGLES, ANGLES, 8 * sizeof(float));
    cudaMemcpyToSymbol(DEV_MOVES, MOVES, 8 * 2 * sizeof(int));


    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                 (height + blockSize.y - 1) / blockSize.y);

    bool* d_active_flag;
    cudaMalloc(&d_active_flag, sizeof(bool));

    unsigned int current_iteration = 2;
    bool h_active = true;

    for (int i = 0; i < 10; i++) {
        // Copy just this cell's data
        unsigned int state;
        Cell cell;
        cudaMemcpy(&state, d_burning_state + i, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cell, d_landscape + i, sizeof(Cell), cudaMemcpyDeviceToHost);


        if (cell.burnable != landscape.cells[i].burnable ) {
            printf("Cells in device memory:\n");
            printf("Cell %d - Burnable: %d, Vegetation: %d, State: %u\n",
                i, cell.burnable, static_cast<int>(cell.vegetation_type), state);
            printf("Cells in landscape:\n");
            printf("Cell %d - Burnable: %d, Vegetation: %d\n",
                i, landscape.cells[i].burnable, static_cast<int>(landscape.cells[i].vegetation_type));
        }


    }

    while (h_active) {
        h_active = false;
        cudaMemset(d_active_flag, 0, sizeof(bool));

        fire_spread_kernel<<<gridSize, blockSize>>>(
            d_landscape, d_burning_state, width, height,
            distance, elevation_mean, elevation_sd, upper_limit,
            params, current_iteration,
            d_active_flag
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
        }

        cudaDeviceSynchronize();
        cudaMemcpy(&h_active, d_active_flag, sizeof(bool), cudaMemcpyDeviceToHost);

        current_iteration++;
    }


    std::vector<unsigned int> temp_burned_layer(num_cells);
    cudaMemcpy(temp_burned_layer.data(), d_burning_state,
            num_cells * sizeof(unsigned int), cudaMemcpyDeviceToHost);



    std::vector<IgnitionPair> burned_ids_aux;
    std::vector<size_t> burned_ids_steps_aux;

    // Convert to Matrix<bool>
    Matrix<bool> burned_layer(width, height);
    unsigned int burned_amount = 0;
    for (size_t i = 0; i < num_cells; i++) {
        burned_layer.data()[i] = (temp_burned_layer[i] > 0);
        burned_amount += int(burned_layer.data()[i]);
    }
    result.width = width;
    result.height = height;
    result.burned_layer = burned_layer;

    cudaFree(d_landscape);
    cudaFree(d_burning_state);
    cudaFree(d_active_flag);
}

Fire simulate_fire(
    const Landscape& landscape,
    const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params,
    float distance,
    float elevation_mean,
    float elevation_sd,
    float upper_limit
) {
    std::vector<IgnitionPair> ignition;
    ignition.reserve(ignition_cells.size());
    for (const auto& p : ignition_cells) {
        ignition.emplace_back(p.first, p.second);
    }

    Fire result{
        landscape.width,
        landscape.height,
        Matrix<bool>(landscape.width, landscape.height),
        std::vector<IgnitionPair>(),
        std::vector<size_t>()
    };

    simulate_fire_cuda(
        landscape, ignition, params,
        distance, elevation_mean, elevation_sd,
        upper_limit, result
    );

    unsigned int burned_amount = 0;
    for (size_t i = 0; i < landscape.width * landscape.height; i++) {
        burned_amount += int(result.burned_layer.data()[i]);
    }


    return result;
}