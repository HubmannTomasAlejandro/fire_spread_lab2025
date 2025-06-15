#include "many_simulations.hpp"
#include "spread_functions.hpp"

#include <cmath>
#include <cassert>
#include <omp.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>  // For fprintf

__global__ void setup_rng_kernel(curandStateXORWOW_t* states,
                                unsigned long seed,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}


Matrix<size_t> burned_amounts_per_cell(
    const Landscape& landscape,
    const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params,
    double distance,
    double elevation_mean,
    double elevation_sd,
    double upper_limit,
    size_t n_replicates
) {
    Matrix<size_t> burned_amounts(landscape.width, landscape.height);
    double t = omp_get_wtime();
    size_t amount_of_burned_cells = 0;
    size_t num_cells = landscape.width * landscape.height;

    // Allocate and copy landscape to device
    Cell* d_landscape;
    cudaMalloc(&d_landscape, num_cells * sizeof(Cell));
    cudaMemcpy(d_landscape, landscape.cells, num_cells * sizeof(Cell), cudaMemcpyHostToDevice);

    // Prepare ignition points
    std::vector<IgnitionPair> ignition;
    ignition.reserve(ignition_cells.size());
    for (const auto& p : ignition_cells) {
        ignition.emplace_back(p.first, p.second);
    }


    dim3 blockSize(16, 16);
    dim3 gridSize((landscape.width + blockSize.x - 1) / blockSize.x,
                 (landscape.height + blockSize.y - 1) / blockSize.y);

    // Allocate burning state on device
    unsigned int* d_burning_state;
    cudaMalloc(&d_burning_state, num_cells * sizeof(unsigned int));

    unsigned int seed=(landscape.width * 2654435761) ^
        (landscape.height * 2246822519);
    curandStateXORWOW_t* d_rng_states;
    cudaMalloc(&d_rng_states, num_cells * sizeof(curandStateXORWOW_t));

    setup_rng_kernel<<<gridSize, blockSize>>>(d_rng_states, seed, num_cells);
    // Initialize RNG states

    for (size_t i = 0; i < n_replicates; i++) {
        // Reset burning state for each replicate
        cudaMemset(d_burning_state, 0, num_cells * sizeof(unsigned int));
        // Set ignition points
        unsigned int value = 1;
        for (const auto& cell : ignition) {
            size_t idx = cell.second * landscape.width + cell.first;
            cudaMemcpy(d_burning_state + idx, &value, sizeof(unsigned int), cudaMemcpyHostToDevice);
        }

        Fire fire = simulate_fire(
            landscape,
            d_landscape,
            ignition,  // Pass the original ignition points
            d_burning_state,
            d_rng_states,
            params,
            static_cast<float>(distance),
            static_cast<float>(elevation_mean),
            static_cast<float>(elevation_sd),
            static_cast<float>(upper_limit)
        );

        // Update burned amounts
        for (size_t j = 0; j < num_cells; j++) {
            if (fire.burned_layer[j]) {
                amount_of_burned_cells++;
                size_t col = j % landscape.width;
                size_t row = j / landscape.width;
                burned_amounts[{col, row}]++;
            }
        }
    }

    // Cleanup
    cudaFree(d_landscape);
    cudaFree(d_burning_state);
    cudaFree(d_rng_states);

    // Print statistics
    fprintf(stderr, "cells_burned_per_micro_sec: %lf\n",
        static_cast<double>(amount_of_burned_cells) / ((omp_get_wtime() - t) * 1e6));

    return burned_amounts;
}