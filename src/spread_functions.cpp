#include "spread_functions.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <vector>
#include <cstdint>
#include <iostream>
#include <bitset>
#include <immintrin.h>

#include <cstddef>
#include <omp.h>

#include "fires.hpp"
#include "landscape.hpp"
#include "constants.hpp"


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
    void inline nextRandomArray(float random_values[8]) {
        for (size_t i = 0; i < 8; i++) {
            random_values[i] = static_cast<float>((*this)(i)) / static_cast<float>(UINT32_MAX);
        }
    }
};

XorShift32 rng;

void inline spread_probability(
  const Landscape& landscape,
  const Cell& burning,
  const int neighbours[2][8],
  const SimulationParams& params,
  float distance,
  float elevation_mean,
  float elevation_sd,
  float* __restrict probs,
  std::bitset<8> const& burnable_cell,
  float upper_limit = 1.0f
) {
  // Tabla de predictores por VegetationType
  static const float veg_pred[5] = {
    0.f,
    params.subalpine_pred,
    params.wet_pred,
    params.dry_pred,
    0.f
  };

  // Buffers temporales para cargar datos vecinos
  alignas(32) float  elevs[8], fwis[8], aspects[8];
  alignas(32) int    veg_types[8];
  alignas(32) float  burnable_mask[8]; // 1.0 si es quemable, 0.0 si no

  for (int i = 0; i < 8; ++i) {
    auto idx = std::pair<size_t, size_t>{
      static_cast<size_t>(neighbours[0][i]),
      static_cast<size_t>(neighbours[1][i])
    };

    elevs[i]       = static_cast<float>(landscape.elevations[idx]);
    fwis[i]        = landscape.fwis[idx];
    aspects[i]     = landscape.aspects[idx];
    veg_types[i]   = static_cast<int>(landscape.vegetation_types[idx]);
    burnable_mask[i] = (landscape.vegetation_types[idx] == NONE || !burnable_cell[i]) ? 0.f : 1.f;
  }

  // Cargar en vectores AVX datos de los vecinos
  __m256 v_elev     = _mm256_load_ps(elevs);
  __m256 v_fwi      = _mm256_load_ps(fwis);
  __m256 v_aspect   = _mm256_load_ps(aspects);
  __m256 v_angles   = _mm256_load_ps(ANGLES);
  //__m256i v_veg_idx = _mm256_load_si256((__m256i*)veg_types);
  __m256 v_mask     = _mm256_load_ps(burnable_mask);

  // hacer vecctores con los parametros de la celda quemada, repetidos
  __m256 v_burning_elev = _mm256_set1_ps(burning.elevation);
  __m256 v_burning_dir  = _mm256_set1_ps(burning.wind_direction);
  __m256 v_dist         = _mm256_set1_ps(distance);
  __m256 v_elev_mean    = _mm256_set1_ps(elevation_mean);
  __m256 v_elev_sd      = _mm256_set1_ps(elevation_sd);

  __m256 v_diff = _mm256_sub_ps(v_elev, v_burning_elev);
  __m256 v_slope = _mm256_div_ps(v_diff, v_dist);
  __m256 v_slope_term = _mm256_sin_ps(_mm256_atan_ps(v_slope));

  __m256 v_wind_term =_mm256_cos_ps(_mm256_sub_ps(v_angles, v_burning_dir));
  __m256 v_elev_term = _mm256_div_ps(_mm256_sub_ps(v_elev, v_elev_mean), v_elev_sd);

  __m256 v_linpred = _mm256_set1_ps(params.independent_pred);

  // Agregamos predictores por vegetación (lookup en tiempo de compilación)
  alignas(32) float veg_lookup[8];
  for (int i = 0; i < 8; ++i) {
    veg_lookup[i] = veg_pred[veg_types[i]];
  }
  v_linpred = _mm256_add_ps(v_linpred, _mm256_load_ps(veg_lookup));

  // Otros predictores
  v_linpred = _mm256_fmadd_ps(_mm256_set1_ps(params.fwi_pred), v_fwi, v_linpred);
  v_linpred = _mm256_fmadd_ps(_mm256_set1_ps(params.aspect_pred), v_aspect, v_linpred);
  v_linpred = _mm256_fmadd_ps(_mm256_set1_ps(params.wind_pred), v_wind_term, v_linpred);
  v_linpred = _mm256_fmadd_ps(_mm256_set1_ps(params.elevation_pred), v_elev_term, v_linpred);
  v_linpred = _mm256_fmadd_ps(_mm256_set1_ps(params.slope_pred), v_slope_term, v_linpred);

  // Aplicamos la función logística
  __m256 v_neg_pred = _mm256_mul_ps(v_linpred, _mm256_set1_ps(-1.0f));
  __m256 v_prob = _mm256_div_ps(
      _mm256_set1_ps(upper_limit),
      _mm256_add_ps(_mm256_set1_ps(1.0f),_mm256_exp_ps(v_neg_pred))
  );

  // Aplicar la máscara de celdas quemables
  v_prob = _mm256_mul_ps(v_prob, v_mask);

  // Guardar el resultado
  _mm256_store_ps(probs, v_prob);
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
  float probs[8];
  bool out_of_range;
  std::bitset<8> burnable_cell;
  int neighbours_coords[2][8];


  __m256 rand_vec;
  __m256 prob_vec;
  __m256 cmp_result;

  u_int8_t mask; // 8-bit int


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
    #pragma omp parallel for schedule(dynamic)
    for (size_t b = start; b < end; b++) {

      size_t burning_cell_0 = burned_ids[b].first;
      size_t burning_cell_1 = burned_ids[b].second;

      burnable_cell.reset();


      const Cell& burning_cell = landscape[{ burning_cell_0, burning_cell_1 }];

      for (size_t i = 0; i < 8; i++) {

        neighbours_coords[0][i] = int(burning_cell_0) + MOVES[i][0];
        neighbours_coords[1][i] = int(burning_cell_1) + MOVES[i][1];

        // Is the cell in range?
        out_of_range = 0 > neighbours_coords[0][i] || neighbours_coords[0][i] >= int(n_col) ||
                            0 > neighbours_coords[1][i] || neighbours_coords[1][i] >= int(n_row);

        neighbours_coords[0][i] = out_of_range ? burning_cell_0 : neighbours_coords[0][i];
        neighbours_coords[1][i] = out_of_range ? burning_cell_1 : neighbours_coords[1][i];

        burnable_cell[i] = out_of_range ?
                false :
                (!burned_bin[{ neighbours_coords[0][i], neighbours_coords[1][i] }] && landscape.burnables[{ neighbours_coords[0][i], neighbours_coords[1][i] }]);
      }

      if (burnable_cell.none()) continue;  // No burnable neighbours

      // Burn with probability prob (Bernoulli)
      spread_probability(
        landscape, burning_cell, neighbours_coords, params, distance, elevation_mean,
        elevation_sd, probs, burnable_cell, upper_limit
      );
      rng.nextRandomArray(random_values);

      rand_vec = _mm256_loadu_ps(random_values); // or _mm256_load_ps if aligned
      prob_vec = _mm256_loadu_ps(probs);
      // Compare: random_values < probs
      cmp_result = _mm256_cmp_ps(rand_vec, prob_vec, _CMP_LT_OQ);

      // Move mask to int (each bit in lower byte corresponds to result)
      mask = static_cast<uint8_t>(_mm256_movemask_ps(cmp_result)); // 8-bit int


      // Proceed only if at least one cell burns
      if (mask==0) continue;

      #pragma omp critical
      {
        // Add the burning cells to the burned_ids vector
        for (size_t i = 0; i < 8; i++) {
          auto coord = std::make_pair(neighbours_coords[0][i], neighbours_coords[1][i]);
          if ((mask >> i) & 1 && !burned_bin[coord]) {  // If the cell should burn
            burned_ids.push_back(coord);
            burned_bin[coord] = true;  // Mark as burned
            end_forward++;  // Increase the count of burned cells
          }
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