#pragma once

#include <vector>
#include <utility>
#include <cstddef>

#include "fires.hpp"
#include "landscape.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>

struct SimulationParams {
  float independent_pred;
  float wind_pred;
  float elevation_pred;
  float slope_pred;
  float subalpine_pred;
  float wet_pred;
  float dry_pred;
  float fwi_pred;
  float aspect_pred;
};


// Declaraciones para CUDA
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

struct Cell;
struct Landscape;

// Función de probabilidad escalar para CUDA
CUDA_CALLABLE float spread_probability_scalar(
    const Cell& burning,
    const Cell& neighbor,
    const SimulationParams& params,
    float distance,
    float elevation_mean,
    float elevation_sd,
    float angle,
    float upper_limit = 1.0f
);

__global__ void set_ignition_kernel(
    unsigned int* d_burning_state,
    const IgnitionPair* d_ignition,
    size_t num_ignition,
    size_t width
);


Fire simulate_fire(
    const Landscape& landscape,
    const Cell*        d_landscape,
    const std::vector<IgnitionPair>& ignition_cells,
    unsigned int*      d_burning_state,
    curandStateXORWOW_t* d_rng_states,
    SimulationParams   params,
    float              distance,
    float              elevation_mean,
    float              elevation_sd,
    size_t             n_replicates,
    float              upper_limit,
    bool*              d_active_flag,    // Recibido desde fuera (pre-asignado)
    cudaStream_t       stream,           // Recibido desde fuera (pre-creado)
    unsigned int*      h_burned_layer    // Buffer pinned recibido desde fuera
);
