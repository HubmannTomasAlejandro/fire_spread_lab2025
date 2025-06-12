#include "spread_functions.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <vector>
#include <cstdint>
#include <iostream>
#include <bitset>
#include <cstddef>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <array>

#include "fires.hpp"
#include "landscape.hpp"
#include "constants.hpp"

// Ángulos de dirección para vecinos (constante global)
__constant__ float DEV_ANGLES[8];

// Movimientos para vecinos (constante global)
__constant__ int DEV_MOVES[8][2];

// Función de probabilidad escalar (compatible CPU/GPU)
CUDA_CALLABLE float spread_probability_scalar(
    const Cell& burning,
    const Cell& neighbor,
    const SimulationParams& params,
    float distance,
    float elevation_mean,
    float elevation_sd,
    float angle,
    float upper_limit
) {
    static std::array<float,4>  veg_pred = {0.f,0.f,0.f,0.f};
    veg_pred[static_cast<int>(VegetationType::SUBALPINE)] = params.subalpine_pred;
    veg_pred[static_cast<int>(VegetationType::WET)] = params.wet_pred;
    veg_pred[static_cast<int>(VegetationType::DRY)] = params.dry_pred;
    

    if (!neighbor.burnable) return 0.0f;

    float slope = (neighbor.elevation - burning.elevation) / distance;
    float slope_term = std::sin(std::atan(slope));
    float wind_term = std::cos(angle - burning.wind_direction);
    float elev_term = (neighbor.elevation - elevation_mean) / elevation_sd;

    float linear_pred = params.independent_pred;
    linear_pred += veg_pred[static_cast<int>(neighbor.vegetation_type)];
    linear_pred += params.fwi_pred * neighbor.fwi;
    linear_pred += params.aspect_pred * neighbor.aspect;
    linear_pred += params.wind_pred * wind_term;
    linear_pred += params.elevation_pred * elev_term;
    linear_pred += params.slope_pred * slope_term;

    float prob = upper_limit / (1.0f + std::exp(-linear_pred));
    return prob;
}

// Kernel CUDA para simular propagación
__global__ void fire_spread_kernel(
    const Cell* landscape,
    unsigned int* burned_bin,
    const IgnitionPair* burning_cells,
    size_t num_burning,
    size_t width,
    size_t height,
    const SimulationParams params,
    float distance,
    float elevation_mean,
    float elevation_sd,
    float upper_limit,
    curandState_t* rng_states,
    IgnitionPair* new_burning,
    int* new_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_burning * 8) return;

    int cell_idx = idx / 8;
    int neighbor_idx = idx % 8;

    auto cell = burning_cells[cell_idx];
    int neighbor_i = cell.first + DEV_MOVES[neighbor_idx][0];
    int neighbor_j = cell.second + DEV_MOVES[neighbor_idx][1];

    if (neighbor_i < 0 || neighbor_i >= height || neighbor_j < 0 || neighbor_j >= width) {
        return;
    }

    size_t neighbor_linear = neighbor_i * width + neighbor_j;
    Cell neighbor_cell = landscape[neighbor_linear];
    if (!neighbor_cell.burnable || burned_bin[neighbor_linear]) {
        return;
    }

    float prob = spread_probability_scalar(
        landscape[cell.first * width + cell.second],
        neighbor_cell,
        params,
        distance,
        elevation_mean,
        elevation_sd,
        DEV_ANGLES[neighbor_idx],
        upper_limit
    );

    curandState_t local_state = rng_states[idx];
    float rand_val = curand_uniform(&local_state);
    rng_states[idx] = local_state;

    if (rand_val < prob) {
        unsigned int old = atomicCAS(&burned_bin[neighbor_linear], 0, 1);
        if (!old) {
            int pos = atomicAdd(new_count, 1);
            new_burning[pos] = {neighbor_i, neighbor_j};
        }
    }
}

// Inicializar estados RNG
__global__ void setup_rng_kernel(curandState_t* state, unsigned long seed, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

// Implementación principal usando CUDA
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

    // Copiar datos al dispositivo
    Cell* d_landscape;
    cudaMalloc(&d_landscape, num_cells * sizeof(Cell));
    cudaMemcpy(d_landscape, landscape.cells.data(), num_cells * sizeof(Cell), cudaMemcpyHostToDevice);

    unsigned int* d_burned_bin;
    cudaMalloc(&d_burned_bin, num_cells * sizeof(unsigned int));
    cudaMemset(d_burned_bin, 0, num_cells * sizeof(unsigned int));

    // Configurar constantes
    cudaMemcpyToSymbol(DEV_ANGLES, ANGLES, 8 * sizeof(float));
    cudaMemcpyToSymbol(DEV_MOVES, MOVES, 8 * 2 * sizeof(int));

    // Inicializar celdas de ignición
    std::vector<IgnitionPair> current_burning = ignition_cells;
    for (const auto& cell : ignition_cells) {
        size_t idx = cell.first * width + cell.second;
        cudaMemset(d_burned_bin + idx, true, sizeof(bool));
    }

    // Configurar RNG
    curandState_t* d_rng_states;
    size_t max_threads = ignition_cells.size() * 8;
    cudaMalloc(&d_rng_states, max_threads * sizeof(curandState_t));
    setup_rng_kernel<<<(max_threads+255)/256, 256>>>(d_rng_states, 12345, max_threads);

    // Buffers para celdas quemadas
    std::vector<IgnitionPair> burned_ids = ignition_cells;
    std::vector<size_t> burned_steps;
    burned_steps.push_back(ignition_cells.size());

    // Bucle de propagación
    while (!current_burning.empty()) {
        size_t current_size = current_burning.size();
        IgnitionPair* d_current;
        cudaMalloc(&d_current, current_size * sizeof(IgnitionPair));
        cudaMemcpy(d_current, current_burning.data(), current_size * sizeof(IgnitionPair), cudaMemcpyHostToDevice);

        int* d_new_count;
        cudaMalloc(&d_new_count, sizeof(int));
        cudaMemset(d_new_count, 0, sizeof(int));

        IgnitionPair* d_new_burning;
        cudaMalloc(&d_new_burning, current_size * 8 * sizeof(IgnitionPair));

        // Lanzar kernel
        size_t total_threads = current_size * 8;
        dim3 blockSize(256);
        dim3 gridSize((total_threads + blockSize.x - 1) / blockSize.x);
        fire_spread_kernel<<<gridSize, blockSize>>>(
            d_landscape, d_burned_bin, d_current, current_size,
            width, height, params, distance, elevation_mean,
            elevation_sd, upper_limit, d_rng_states,
            d_new_burning, d_new_count
        );

        // Recuperar resultados
        int new_count;
        cudaMemcpy(&new_count, d_new_count, sizeof(int), cudaMemcpyDeviceToHost);

        std::vector<IgnitionPair> new_burning(new_count);
        if (new_count > 0) {
            cudaMemcpy(new_burning.data(), d_new_burning, new_count * sizeof(IgnitionPair), cudaMemcpyDeviceToHost);
            burned_ids.insert(burned_ids.end(), new_burning.begin(), new_burning.end());
        }

        burned_steps.push_back(burned_ids.size());
        current_burning = std::move(new_burning);

        // Liberar memoria
        cudaFree(d_current);
        cudaFree(d_new_count);
        cudaFree(d_new_burning);
    }

    // Recuperar matriz quemada
    Matrix<bool> burned_layer(width, height);
    cudaMemcpy(burned_layer.data(), d_burned_bin, num_cells * sizeof(bool), cudaMemcpyDeviceToHost);

    // Construir resultado 
    result.width = width;
    result.height = height;
    result.burned_layer = std::move(burned_layer);  // Mover para evitar copia
    result.burned_ids = std::move(burned_ids);
    result.burned_ids_steps = std::move(burned_steps);

    // Liberar memoria
    cudaFree(d_landscape);
    cudaFree(d_burned_bin);
    cudaFree(d_rng_states);
}
// Función principal
Fire simulate_fire(
    const Landscape& landscape,
    const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params,
    float distance,
    float elevation_mean,
    float elevation_sd,
    float upper_limit
) {
    // Convertir ignition_cells a IgnitionPair
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
    return result;
}
