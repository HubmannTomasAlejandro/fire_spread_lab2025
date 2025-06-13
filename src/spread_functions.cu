#include "spread_functions.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <vector>
#include <iostream>
#include <bitset>

#include "fires.hpp"
#include "landscape.hpp"
#include "constants.hpp"

// Semillas para los 8 generadores (una por dirección de vecino)
const uint32_t seeds[8] = {12345, 67890, 13579, 24680, 11223, 44556, 77889, 99000};

class XorShift32 {
private:
    uint32_t state;

public:
    __host__ __device__
    explicit XorShift32(uint32_t seed = 12345) {
        state = seed ? seed : 2463534242U; // Evita estado cero
    }

    __host__ __device__
    float nextFloat() {
        // Guardar el estado actual para devolverlo
        uint32_t current = state;
        
        // Actualizar el estado para la próxima llamada
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        
        // Convertir el valor actual (no el actualizado)
        return static_cast<float>(current) / static_cast<float>(UINT32_MAX);
    }
};

void inline spread_probability(
  const Landscape& landscape,
  const Cell& burning,
  const int neighbours[2][8], 
  SimulationParams params,
  float distance,
  float elevation_mean,
  float elevation_sd,
  float* probs,
  std::bitset<8>& burnable_cell,
  float upper_limit = 1.0f
) {
  IgnitionPair neighbour;
  for (size_t i = 0; i < 8; i++) {
    neighbour.first = neighbours[0][i];
    neighbour.second = neighbours[1][i];

    float slope_term = sinf(atanf((landscape.elevations[{neighbour.first, neighbour.second}] - burning.elevation) / distance);
    float wind_term = cosf(ANGLES[i] - burning.wind_direction);
    float elev_term = (landscape.elevations[{neighbour.first, neighbour.second}] - elevation_mean) / elevation_sd;

    float linpred = params.independent_pred;

    if (landscape.vegetation_types[{neighbour.first, neighbour.second}] == SUBALPINE) {
        linpred += params.subalpine_pred;
    } else if (landscape.vegetation_types[{neighbour.first, neighbour.second}] == WET) {
        linpred += params.wet_pred;
    } else if (landscape.vegetation_types[{neighbour.first, neighbour.second}] == DRY) {
        linpred += params.dry_pred;
    }

    linpred += params.fwi_pred * landscape.fwis[{neighbour.first, neighbour.second}];
    linpred += params.aspect_pred * landscape.aspects[{neighbour.first, neighbour.second}];
    linpred += wind_term * params.wind_pred +
               elev_term * params.elevation_pred +
               slope_term * params.slope_pred;

    probs[i] = (landscape.vegetation_types[neighbour] == NONE || !burnable_cell[i])
               ? 0.0f
               : upper_limit / (1.0f + expf(-linpred));
  }
}

Fire simulate_fire(
    const Landscape& landscape, 
    const std::vector<IgnitionPair>& ignition_cells,
    SimulationParams params, 
    float distance, 
    float elevation_mean, 
    float elevation_sd,
    float upper_limit = 1.0
) {
    size_t n_row = landscape.height;
    size_t n_col = landscape.width;

    std::vector<IgnitionPair> burned_ids(ignition_cells);
    std::vector<size_t> burned_ids_steps = {ignition_cells.size()};

    Matrix<bool> burned_bin(n_col, n_row);
    for (const auto& cell : ignition_cells) {
        burned_bin[cell] = true;
    }

    // 8 generadores independientes (uno por dirección)
    XorShift32 rngs[8];
    for (int i = 0; i < 8; i++) {
        rngs[i] = XorShift32(seeds[i]);
    }

    int neighbours_coords[2][8];
    float probs[8];
    
    size_t start = 0;
    size_t end = ignition_cells.size();
    size_t burning_size = end - start;

    while (burning_size > 0) {
        size_t end_forward = end;

        for (size_t b = start; b < end; b++) {
            const IgnitionPair& burning_id = burned_ids[b];
            size_t burning_cell_0 = burning_id.first;
            size_t burning_cell_1 = burning_id.second;

            std::bitset<8> burnable_cell;
            const Cell& burning_cell = landscape[burning_id];

            // Calcular vecinos y celdas quemables
            for (int i = 0; i < 8; i++) {
                neighbours_coords[0][i] = burning_cell_0 + MOVES[i][0];
                neighbours_coords[1][i] = burning_cell_1 + MOVES[i][1];

                bool out_of_range = 
                    neighbours_coords[0][i] < 0 || 
                    neighbours_coords[0][i] >= n_col ||
                    neighbours_coords[1][i] < 0 || 
                    neighbours_coords[1][i] >= n_row;

                if (!out_of_range) {
                    IgnitionPair neighbour(neighbours_coords[0][i], neighbours_coords[1][i]);
                    burnable_cell[i] = !burned_bin[neighbour] && landscape.burnables[neighbour];
                }
            }

            if (burnable_cell.none()) continue;

            spread_probability(
                landscape, burning_cell, neighbours_coords, params, distance,
                elevation_mean, elevation_sd, probs, burnable_cell, upper_limit
            );

            // Usar generador específico para cada dirección
            for (int i = 0; i < 8; i++) {
                if (burnable_cell[i]) {
                    float rand_val = rngs[i].nextFloat();
                    if (rand_val < probs[i]) {
                        IgnitionPair new_burn(neighbours_coords[0][i], neighbours_coords[1][i]);
                        burned_ids.push_back(new_burn);
                        burned_bin[new_burn] = true;
                        end_forward++;
                    }
                }
            }
        }

        start = end;
        end = end_forward;
        burning_size = end - start;
        burned_ids_steps.push_back(end);
    }

    return {n_col, n_row, burned_bin, burned_ids, burned_ids_steps};
}
