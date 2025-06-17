#include "many_simulations.hpp"
#include "spread_functions.hpp"
#include "constants.hpp"

#include <cmath>
#include <cassert>
#include <omp.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>  // For fprintf


__constant__ float DEV_ANGLES[8];
__constant__ int DEV_MOVES[8][2];

__global__ void setup_rng_kernel_auto(curandStateXORWOW_t* state, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed + idx, 0, 0, &state[idx]);
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
    size_t amount_of_burned_cells = 0;
    size_t num_cells = landscape.width * landscape.height;
    size_t width = landscape.width;

    // ─── Pre-asignación de recursos persistentes ──────────────────────────
    Cell* d_landscape = nullptr;
    unsigned int* d_burning_state = nullptr;
    curandStateXORWOW_t* d_rng_states = nullptr;
    IgnitionPair* d_ignition = nullptr;
    unsigned int* h_burned_layer = nullptr;
    bool* d_active_flag = nullptr;
    cudaStream_t stream;

    // 1. Memoria pinned para resultados (reutilizable entre réplicas)
    cudaHostAlloc(&h_burned_layer, num_cells * sizeof(unsigned int), cudaHostAllocDefault);

    // 2. Stream y flag de estado
    cudaStreamCreate(&stream);
    cudaMalloc(&d_active_flag, sizeof(bool));

    // 3. Memoria para paisaje en device
    cudaMalloc(&d_landscape, num_cells * sizeof(Cell));
    cudaMemcpyAsync(d_landscape, landscape.cells, num_cells * sizeof(Cell),
                   cudaMemcpyHostToDevice, stream);

    // 4. Estado de quemado y RNG
    cudaMalloc(&d_burning_state, num_cells * sizeof(unsigned int));
    cudaMalloc(&d_rng_states, num_cells * sizeof(curandStateXORWOW_t));

    // 5. Copiar puntos de ignición a device (una sola vez)
    std::vector<IgnitionPair> ignition;
    ignition.reserve(ignition_cells.size());
    for (const auto& p : ignition_cells) {
        ignition.emplace_back(static_cast<unsigned int>(p.first),
                             static_cast<unsigned int>(p.second));
    }

    cudaMalloc(&d_ignition, ignition.size() * sizeof(IgnitionPair));
    cudaMemcpyAsync(d_ignition, ignition.data(), ignition.size() * sizeof(IgnitionPair),
                   cudaMemcpyHostToDevice, stream);

    // 6. Copiar constantes a símbolos (una sola vez)
    cudaMemcpyToSymbol(DEV_ANGLES, ANGLES, 8 * sizeof(float));
    cudaMemcpyToSymbol(DEV_MOVES,  MOVES,  8 * 2 * sizeof(int));
    cudaStreamSynchronize(stream);

    // ─── Configuración kernel de ignición ─────────────────────────────────
    const size_t ignition_size = ignition.size();
    const dim3 ignitionBlock(256);
    const dim3 ignitionGrid((ignition_size + ignitionBlock.x - 1) / ignitionBlock.x);

    //double t = omp_get_wtime();
    /*auto setup_rng_kernel_auto = [] __device__ (curandStateXORWOW_t* state,    unsigned long seed, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            curand_init(seed + idx, 0, 0, &state[idx]);
        }
    };
*/
    double t = omp_get_wtime();
    unsigned long base_seed = static_cast<unsigned long>(t * 1000);

    // ─── Bucle principal de réplicas ──────────────────────────────────────
    for (size_t i = 0; i < n_replicates; i++) {
        // 1. Reset estado de quemado
        cudaMemsetAsync(d_burning_state, 0, num_cells * sizeof(unsigned int), stream);

	dim3 rngBlock(256);
        dim3 rngGrid((num_cells + rngBlock.x - 1) / rngBlock.x);
        setup_rng_kernel_auto<<<rngGrid, rngBlock, 0, stream>>>(d_rng_states, base_seed + i, num_cells);
        cudaStreamSynchronize(stream);

        // 2. Inicializar puntos de ignición con kernel
        /*set_ignition_kernel<<<ignitionGrid, ignitionBlock, 0, stream>>>(
            d_burning_state,
            d_ignition,
            ignition_size,
            width
        );*/

	unsigned int value = 1;
        for (const auto& cell : ignition) {
            size_t idx = cell.second * landscape.width + cell.first;
            cudaMemcpy(d_burning_state + idx, &value, sizeof(unsigned int), cudaMemcpyHostToDevice);
        }

        // 3. Simular propagación de fuego (reutiliza recursos pre-asignados)
        Fire fire = simulate_fire(
            landscape,
            d_landscape,
            ignition,
            d_burning_state,
            d_rng_states,
            params,
            static_cast<float>(distance),
            static_cast<float>(elevation_mean),
            static_cast<float>(elevation_sd),
            i,
            static_cast<float>(upper_limit),
            d_active_flag,    // Pre-asignado
            stream,           // Pre-creado
            h_burned_layer   // Buffer pinned
        );

        // 4. Procesar resultados (usa buffer pinned directamente)
        for (size_t j = 0; j < num_cells; j++) {
            if (fire.burned_layer[j]) {
                amount_of_burned_cells++;
                size_t col = j % width;
                size_t row = j / width;
                burned_amounts[{col, row}]++;
            }
        }
    }

    // ─── Liberación de recursos persistentes ──────────────────────────────
    cudaStreamSynchronize(stream);
    cudaFreeHost(h_burned_layer);
    cudaFree(d_active_flag);
    cudaStreamDestroy(stream);
    cudaFree(d_landscape);
    cudaFree(d_burning_state);
    cudaFree(d_rng_states);
    cudaFree(d_ignition);

    // ─── Estadísticas de rendimiento ─────────────────────────────────────
    double elapsed = (omp_get_wtime() - t) * 1e6;  // microsegundos
    if (elapsed > 0) {
        fprintf(stderr, "cells_burned_per_micro_sec: %.2f\n",
                static_cast<double>(amount_of_burned_cells) / elapsed);
    }

    return burned_amounts;
}