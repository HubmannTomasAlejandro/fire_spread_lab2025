#pragma once

#include <vector>
#include <utility>

#include "fires.hpp"
#include "landscape.hpp"

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

// Declarar kernel CUDA
void simulate_fire_cuda(
    const Landscape& landscape,
    const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params,
    float distance,
    float elevation_mean,
    float elevation_sd,
    float upper_limit,
    Fire& result
);

// Función principal modificada para usar CUDA
Fire simulate_fire(
    const Landscape& landscape,
    const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params,
    float distance,
    float elevation_mean,
    float elevation_sd,
    float upper_limit = 1.0f
);

