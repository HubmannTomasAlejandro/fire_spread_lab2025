#include "spread_functions.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <vector>
#include <cstdint>
#include <iostream>

#include "fires.hpp"
#include "landscape.hpp"
#include "constants.hpp"


static Cell out_of_bounds_cell = {
  0,               // elevation
  0.0f,            // wind_direction
  false,           // burnable
  VegetationType::NONE, // vegetation_type
  0.0f,            // fwi (Fire Weather Index)
  0.0f             // aspect
};

static uint32_t seeds[8] = {12345, 67890, 13579, 24680, 11223, 44556, 77889, 99000};

class XorShift32 {
private:
    uint32_t state[8];

public:
    // Constructor initializes state array with seeds
    explicit XorShift32() {
        for (size_t i = 0; i < 8; i++) {
            state[i] = seeds[i];
            if (state[i] == 0) {
                state[i] = 2463534242U; // Avoid all-zero state
            }
        }
    }

    uint32_t operator()(size_t idx) {
        state[idx] ^= state[idx] << 13;
        state[idx] ^= state[idx] >> 17;
        state[idx] ^= state[idx] << 5;
        return state[idx];
    }

    // Method to get a single random value
    uint32_t next() {
        return (*this)(0);  // Returns the first element for simplicity
    }

    // Method to generate 8 random values and store in an array
    void nextRandomArray(float random_values[8]) {
        for (size_t i = 0; i < 8; i++) {
            random_values[i] = static_cast<float>((*this)(i)) / static_cast<float>(UINT32_MAX);
        }
    }
};

XorShift32 rng;

void spread_probability(
  const Cell& burning,
  const Cell neighbors[8],  // arreglo C-style de 8 vecinos
  SimulationParams params,
  float distance,
  float elevation_mean,
  float elevation_sd,
  float* probs,
  bool* burnable_cell,              // puntero a array de 8 floats
  float upper_limit = 1.0f   // ahora sí, último argumento con valor por defecto
) {
  for (size_t i = 0; i < 8; i++) {
      const Cell& neighbour = neighbors[i];

      float slope_term = sin(atan((neighbour.elevation - burning.elevation) / distance));
      float wind_term = cos(ANGLES[i] - burning.wind_direction);
      float elev_term = (neighbour.elevation - elevation_mean) / elevation_sd;

      float linpred = params.independent_pred;

      if (neighbour.vegetation_type == SUBALPINE) {
          linpred += params.subalpine_pred;
      } else if (neighbour.vegetation_type == WET) {
          linpred += params.wet_pred;
      } else if (neighbour.vegetation_type == DRY) {
          linpred += params.dry_pred;
      }

      linpred += params.fwi_pred * neighbour.fwi;
      linpred += params.aspect_pred * neighbour.aspect;

      linpred += wind_term * params.wind_pred +
                 elev_term * params.elevation_pred +
                 slope_term * params.slope_pred;

      probs[i] = (neighbour.vegetation_type == NONE || !burnable_cell[i])
          ? 0.0f
          : upper_limit / (1.0f + exp(-linpred));
  }
}


Fire simulate_fire(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit = 1.0
) {

  size_t n_row = landscape.height;
  size_t n_col = landscape.width;

  std::vector<std::pair<size_t, size_t>> burned_ids;

  size_t start = 0;
  size_t end = ignition_cells.size();

  for (size_t i = 0; i < end; i++) {
    burned_ids.push_back(ignition_cells[i]);
  }

  std::vector<size_t> burned_ids_steps;
  burned_ids_steps.push_back(end);

  size_t burning_size = end + 1;

  Matrix<bool> burned_bin = Matrix<bool>(n_col, n_row);

  float random_values[8];
  Cell neighbour_cells[8];
  float probs[8];
  bool out_of_range;
  bool burned_cell [8];
  bool burnable_cell [8];

  for (size_t i = 0; i < end; i++) {
    size_t cell_0 = ignition_cells[i].first;
    size_t cell_1 = ignition_cells[i].second;
    burned_bin[{ cell_0, cell_1 }] = 1;
  }

  while (burning_size > 0) {
    size_t end_forward = end;

    // Loop over burning cells in the cycle

    // b is going to keep the position in burned_ids that have to be evaluated
    // in this burn cycle
    for (size_t b = start; b < end; b++) {

      size_t burning_cell_0 = burned_ids[b].first;
      size_t burning_cell_1 = burned_ids[b].second;

      for (size_t i = 0; i < 8; i++) {
        burnable_cell[i] = false;
        burned_cell[i] = false;
    }

      const Cell& burning_cell = landscape[{ burning_cell_0, burning_cell_1 }];

      int neighbours_coords[2][8];

      for (size_t i = 0; i < 8; i++) {
        neighbours_coords[0][i] = int(burning_cell_0) + MOVES[i][0];
        neighbours_coords[1][i] = int(burning_cell_1) + MOVES[i][1];


        // Is the cell in range?
        out_of_range = 0 > neighbours_coords[0][i] || neighbours_coords[0][i] >= int(n_col) ||
                            0 > neighbours_coords[1][i] || neighbours_coords[1][i] >= int(n_row);

        neighbour_cells[i] = out_of_range ? out_of_bounds_cell : landscape[{neighbours_coords[0][i], neighbours_coords[1][i]}];

        burnable_cell[i] = out_of_range ?
                false :
                (!burned_bin[{ neighbours_coords[0][i], neighbours_coords[1][i] }] && neighbour_cells[i].burnable);
      }

      // Burn with probability prob (Bernoulli)
      spread_probability(
        burning_cell, neighbour_cells, params, distance, elevation_mean,
        elevation_sd, probs, burnable_cell, upper_limit
      );
      rng.nextRandomArray(random_values);

      bool any_burn = false;  // Flag to check if any cell should burn

      // Check each neighbor's burn condition
      for (size_t i = 0; i < 8; i++) {
        burned_cell[i] = random_values[i] < probs[i]; // Set burn condition for each cell

        // If any cell should burn, set 'any_burn' to true
        if (burned_cell[i]) {
          any_burn = true;
        }
      }

      // Proceed only if at least one cell burns
      if (!any_burn){
        continue;  // No cell burns, skip to next iteration
      }

      for (size_t i = 0; i < 8; i++) {
        if (burned_cell[i]) {  // If the cell should burn
            // Assuming `neighbour_cell_0` and `neighbour_cell_1` are the coordinates of the neighbor cell
            burned_ids.push_back({ neighbours_coords[0][i], neighbours_coords[1][i] });
            burned_bin[{ neighbours_coords[0][i], neighbours_coords[1][i] }] = true;  // Mark as burned
            end_forward++;  // Increase the count of burned cells
        }
    }
  }

    // update start and end
    start = end;
    end = end_forward;
    burning_size = end - start;

    burned_ids_steps.push_back(end);
  }

  return { n_col, n_row, burned_bin, burned_ids, burned_ids_steps };
}
