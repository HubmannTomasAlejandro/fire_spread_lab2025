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

/*
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
  std::bitset<8>& burnable_cell,              // puntero a array de 8 floats
  float upper_limit = 1.0f   // ahora sí, último argumento con valor por defecto
) {
  #pragma omp simd
  for (size_t i = 0; i < 8; i++) {
      const Cell& neighbour = neighbors[i];

      float slope_term = sinf(atanf((neighbour.elevation - burning.elevation) / distance));
      float wind_term = cosf(ANGLES[i] - burning.wind_direction);
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
          : upper_limit / (1.0f + expf(-linpred));
  }
}
*/
/*
// La versión vectorizada+paralela de spread_probability
void spread_probability(
    const Cell& burning,
    const Cell neighbors[8],
    const SimulationParams& params,
    float distance,
    float elevation_mean,
    float elevation_sd,
    float* __restrict probs,
    const std::bitset<8>& burnable_cell,
    float upper_limit = 1.0f
) {

   // Empaquetar campos en Structure‑of‑Arrays
   alignas(32) float elev_arr[8];
   VegetationType veg_arr[8];
   float fwi_arr[8];
   float asp_arr[8];

   for (size_t i = 0; i < 8; ++i) {
       elev_arr[i]  = float(neighbors[i].elevation);
       veg_arr[i]   = neighbors[i].vegetation_type;
       fwi_arr[i]   = neighbors[i].fwi;
       asp_arr[i]   = neighbors[i].aspect;
       // burn_arr[i] = neighbors[i].burnable;
   }

  // Parallel + SIMD en un solo pragma
    #pragma omp simd
    for (size_t i = 0; i < 8; ++i) {
      float diff       = elev_arr[i] - burning.elevation;
      float slope_term = sinf(atanf(diff / distance));
      float wind_term  = cosf(ANGLES[i] - burning.wind_direction);
      float elev_term  = (elev_arr[i] - elevation_mean) / elevation_sd;

      float linpred = params.independent_pred;
      switch (veg_arr[i]) {
          case SUBALPINE: linpred += params.subalpine_pred; break;
          case WET:       linpred += params.wet_pred;       break;
          case DRY:       linpred += params.dry_pred;       break;
          default:                                          break;
      }
      linpred += params.fwi_pred     * fwi_arr[i]
              + params.aspect_pred  * asp_arr[i]
              + params.wind_pred    * wind_term
              + params.elevation_pred * elev_term
              + params.slope_pred   * slope_term;

        // Cálculo final de la probabilidad
        bool can_burn = (veg_arr[i] == NONE) || !burnable_cell[i];
        probs[i] = can_burn
                 ? 0.0f
                 : upper_limit / (1.0f + expf(-linpred));
    }
}
*/

/*
void spread_probability(
    const Landscape& landscape,
    const Cell& burning,
    const int neighbours[2][8],     // { x_coords[8], y_coords[8] }
    const SimulationParams& params,
    float distance,
    float elevation_mean,
    float elevation_sd,
    float* probs,                   // output[8]
    std::bitset<8>& burnable_cell,
    float upper_limit = 1.0f
) {
    // 1) Construye vector de índices lineales = y * width + x
    int idx_arr[8];
    for (int i = 0; i < 8; ++i) {
        idx_arr[i] = neighbours[1][i] * landscape.width + neighbours[0][i];
    }
    __m256i v_idx = _mm256_loadu_si256((__m256i*)idx_arr);

    // 2) Gather elevations, fwis, aspects, vegetation_types
    __m256  v_elev   = _mm256_i32gather_ps(landscape.elevations,   v_idx, sizeof(float));
    __m256  v_fwi    = _mm256_i32gather_ps(landscape.fwis,         v_idx, sizeof(float));
    __m256  v_aspect = _mm256_i32gather_ps(landscape.aspects,      v_idx, sizeof(float));
    __m256i v_veg    = _mm256_i32gather_epi32(landscape.vegetation_types, v_idx, sizeof(int));

    // 3) Constantes broadcast
    __m256 v_b_elev  = _mm256_set1_ps(burning.elevation);
    __m256 v_b_wind  = _mm256_set1_ps(burning.wind_direction);
    __m256 v_dist    = _mm256_set1_ps(distance);
    __m256 v_mean    = _mm256_set1_ps(elevation_mean);
    __m256 v_sd      = _mm256_set1_ps(elevation_sd);
    __m256 v_ipred   = _mm256_set1_ps(params.independent_pred);
    __m256 v_sub    = _mm256_set1_ps(params.subalpine_pred);
    __m256 v_wet    = _mm256_set1_ps(params.wet_pred);
    __m256 v_dry    = _mm256_set1_ps(params.dry_pred);
    __m256 v_fwipe  = _mm256_set1_ps(params.fwi_pred);
    __m256 v_aspepe = _mm256_set1_ps(params.aspect_pred);
    __m256 v_windp  = _mm256_set1_ps(params.wind_pred);
    __m256 v_elevp  = _mm256_set1_ps(params.elevation_pred);
    __m256 v_slopep = _mm256_set1_ps(params.slope_pred);
    __m256 v_upper  = _mm256_set1_ps(upper_limit);

    // 4) slope_term = sin(atan((elev - belev)/dist))
    __m256 v_delta   = _mm256_sub_ps(v_elev, v_b_elev);
    __m256 v_slope   = _mm256_sin_ps(_mm256_atan_ps(_mm256_div_ps(v_delta, v_dist)));

    // 5) wind_term = cos(ANGLES - b_wind)
    __m256 v_angles  = _mm256_loadu_ps(ANGLES);
    __m256 v_wterm   = _mm256_cos_ps(_mm256_sub_ps(v_angles, v_b_wind));

    // 6) elev_term = (elev - mean)/sd
    __m256 v_eterm   = _mm256_div_ps(_mm256_sub_ps(v_elev, v_mean), v_sd);

    // 7) Predicción lineal base
    __m256 v_linpred = v_ipred;

    // 8) Predictores por tipo de vegetación
    __m256i vi_sub   = _mm256_set1_epi32(SUBALPINE),
            vi_wet   = _mm256_set1_epi32(WET),
            vi_dry   = _mm256_set1_epi32(DRY);
    __m256 m_sub     = _mm256_castsi256_ps(_mm256_cmpeq_epi32(v_veg, vi_sub));
    __m256 m_wet     = _mm256_castsi256_ps(_mm256_cmpeq_epi32(v_veg, vi_wet));
    __m256 m_dry     = _mm256_castsi256_ps(_mm256_cmpeq_epi32(v_veg, vi_dry));

    v_linpred = _mm256_add_ps(v_linpred,
                 _mm256_add_ps(_mm256_mul_ps(m_sub, v_sub),
                   _mm256_add_ps(_mm256_mul_ps(m_wet, v_wet),
                                 _mm256_mul_ps(m_dry, v_dry))));

    // 9) Añade fwi y aspect
    v_linpred = _mm256_fmadd_ps(v_fwipe, v_fwi,    v_linpred);
    v_linpred = _mm256_fmadd_ps(v_aspepe, v_aspect, v_linpred);

    // 10) Añade terms continuos
    v_linpred = _mm256_fmadd_ps(v_windp,  v_wterm,  v_linpred);
    v_linpred = _mm256_fmadd_ps(v_elevp,  v_eterm,  v_linpred);
    v_linpred = _mm256_fmadd_ps(v_slopep, v_slope,  v_linpred);

    // 11) Logistic: upper/(1+exp(-linpred))
    __m256 v_neg     = _mm256_sub_ps(_mm256_setzero_ps(), v_linpred);
    __m256 v_exp     = _mm256_exp_ps(v_neg);
    __m256 v_logis   = _mm256_div_ps(v_upper, _mm256_add_ps(_mm256_set1_ps(1.0f), v_exp));

    // 12) Máscara final: veg != NONE && burnable_cell
    __m256i vi_none  = _mm256_set1_epi32(NONE);
    __m256 m_notnone = _mm256_castsi256_ps(_mm256_cmpgt_epi32(v_veg, vi_none));

    int bm_arr[8];
    for(int i=0;i<8;i++) bm_arr[i] = burnable_cell[i] ? -1 : 0;
    __m256  m_burn   = _mm256_castsi256_ps(_mm256_loadu_si256((__m256i*)bm_arr));

    __m256 m_final   = _mm256_and_ps(m_notnone, m_burn);
    __m256 v_zero    = _mm256_setzero_ps();
    __m256 v_result  = _mm256_blendv_ps(v_zero, v_logis, m_final);

    // 13) Almacena
    _mm256_storeu_ps(probs, v_result);
}
*/

/*
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
  std::bitset<8> burnable_cell;

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
    for (size_t b = start; b < end; b++) {

      size_t burning_cell_0 = burned_ids[b].first;
      size_t burning_cell_1 = burned_ids[b].second;

      burnable_cell.reset();


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

      if (burnable_cell.none()) continue;  // No burnable neighbours

      // Burn with probability prob (Bernoulli)
      spread_probability(
        burning_cell, neighbour_cells, params, distance, elevation_mean,
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

      for (size_t i = 0; i < 8; i++) {
        if ((mask >> i) & 1) {  // If the cell should burn
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
*/

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
/*
void inline spread_probability(
  const Landscape& landscape,
  const Cell& burning,
  const int neighbours[2][8],  // arreglo C-style de 8 vecinos
  SimulationParams params,
  float distance,
  float elevation_mean,
  float elevation_sd,
  float* __restrict probs,   // agregando __restrict para mejorar la optimización
  std::bitset<8>& burnable_cell,              // puntero a array de 8 floats
  float upper_limit = 1.0f   // ahora sí, último argumento con valor por defecto
) {
  //__builtin_assume_aligned(probs, 32) ;
  std::pair<size_t, size_t> neighbour;

  // 1) Definimos una tabla de predictores por VegetationType
  //static const float veg_pred[5] = {
  //  /* MATORRAL */    //0.f,
  //  /* SUBALPINE */   params.subalpine_pred,
  //  /* WET */         params.wet_pred,
  //  /* DRY */         params.dry_pred,
  //  /* NONE */        0.f
  //};

  //#pragma omp simd
  /*for (size_t i = 0; i < 8; i++) {

      neighbour.first = neighbours[0][i];
      neighbour.second =  neighbours[1][i];

      float slope_term = sinf(atanf((landscape.elevations[{neighbour.first, neighbour.second}] - burning.elevation) / distance));
      float wind_term = cosf(ANGLES[i] - burning.wind_direction);
      float elev_term = (landscape.elevations[{ neighbour.first, neighbour.second}] - elevation_mean) / elevation_sd;


      float linpred = params.independent_pred;

      VegetationType vt = landscape.vegetation_types[{neighbour.first, neighbour.second}];
      // suma directa desde la tabla, sin if/switch
      linpred += veg_pred[ static_cast<int>(vt) ];

      linpred += params.fwi_pred * landscape.fwis[{ neighbour.first, neighbour.second }];
      linpred += params.aspect_pred * landscape.aspects[{ neighbour.first, neighbour.second }];

      linpred += wind_term * params.wind_pred +
                 elev_term * params.elevation_pred +
                 slope_term * params.slope_pred;

      probs[i] = (landscape.vegetation_types[{neighbour.first, neighbour.second}] == NONE || !burnable_cell[i])
          ? 0.0f
          : upper_limit / (1.0f + expf(-linpred));
  }
}
*/


void inline spread_probability_simd(
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
    // 1) punteros a las matrices internas
    const short* elev_data = landscape.elevations[{0,0}];
    const float* wind_dir_data = landscape.wind_directions[{0,0}];
    const VegetationType* veg_data = landscape.vegetation_types[{0,0}];
    const float* fwi_data  = landscape.fwis[{0,0}];
    const float* aspect_data = landscape.aspects[{0,0}];
    const size_t width = landscape.width;

    // 2) cargar constantes escalares en vectores
    __m256 v_burn_elev   = _mm256_set1_ps((float)burning.elevation);
    __m256 v_wind_dir    = _mm256_set1_ps(burning.wind_direction);
    __m256 v_dist_inv    = _mm256_set1_ps(1.0f / distance);
    __m256 v_elev_mean   = _mm256_set1_ps(elevation_mean);
    __m256 v_elev_sd_inv = _mm256_set1_ps(1.0f / elevation_sd);
    __m256 v_indep       = _mm256_set1_ps(params.independent_pred);
    __m256 v_fwi_pred    = _mm256_set1_ps(params.fwi_pred);
    __m256 v_aspect_pred = _mm256_set1_ps(params.aspect_pred);
    __m256 v_wind_pred   = _mm256_set1_ps(params.wind_pred);
    __m256 v_elev_pred   = _mm256_set1_ps(params.elevation_pred);
    __m256 v_slope_pred  = _mm256_set1_ps(params.slope_pred);
    __m256 v_upper       = _mm256_set1_ps(upper_limit);

    // 3) tabla de vegetación en un array alineado a 32 B
    alignas(32) static float veg_pred_tbl[5] = {
        0.f,
        0.f, // SUBALPINE
        0.f, // WET
        0.f, // DRY
        0.f  // NONE
    };
    veg_pred_tbl[1] = params.subalpine_pred;
    veg_pred_tbl[2] = params.wet_pred;
    veg_pred_tbl[3] = params.dry_pred;

    // 4) cargar ANGLES y vecinos
    __m256  v_angles = _mm256_loadu_ps(ANGLES);
    __m256i vx       = _mm256_loadu_si256((const __m256i*)neighbours[0]);
    __m256i vy       = _mm256_loadu_si256((const __m256i*)neighbours[1]);
    __m256i v_width  = _mm256_set1_epi32((int)width);
    __m256i vidx     = _mm256_add_epi32(_mm256_mullo_epi32(vy, v_width), vx);

    // 5) gather de elevaciones, fwi y aspects (float)  
    __m256 v_elev   = _mm256_i32gather_ps(elev_data, vidx,  sizeof(*elev_data));
    __m256 v_fwi    = _mm256_i32gather_ps(fwi_data, vidx,  sizeof(*fwi_data));
    __m256 v_aspect = _mm256_i32gather_ps(aspect_data, vidx,  sizeof(*aspect_data));

    // 6) gather de tipos de vegetación (byte → int → índice en veg_pred_tbl)
    //    Primero gather 32 bit (cogemos el byte + basura, pero luego lo enmascaramos)
    __m256i raw_vt = _mm256_i32gather_epi32(
        reinterpret_cast<const int*>(veg_data), 
        vidx, 
        sizeof(*veg_data)
    );
    // enmascarar para quedarnos con el LSB
    __m256i vt_idx = _mm256_and_si256(raw_vt, _mm256_set1_epi32(0xFF));
    __m256  v_veg_pred = _mm256_i32gather_ps(veg_pred_tbl, vt_idx, sizeof(veg_pred_tbl[0]));

    // 7) cálculo de slope_term, wind_term y elev_term
    __m256 v_diff     = _mm256_sub_ps(v_elev, v_burn_elev);
    __m256 v_ratio    = _mm256_mul_ps(v_diff, v_dist_inv);
    __m256 v_slope    = _mm256_sin_ps( _mm256_atan_ps(v_ratio) );
    __m256 v_wind     = _mm256_cos_ps( _mm256_sub_ps(v_angles, v_wind_dir) );
    __m256 v_elev_tm  = _mm256_mul_ps(
        _mm256_sub_ps(v_elev, v_elev_mean),
        v_elev_sd_inv
    );

    // 8) montar el predictor lineal con FMAs
    __m256 lin = v_indep;
    lin = _mm256_add_ps(lin, v_veg_pred);
    lin = _mm256_fmadd_ps(v_fwi,        v_fwi_pred,    lin);
    lin = _mm256_fmadd_ps(v_aspect,     v_aspect_pred, lin);
    lin = _mm256_fmadd_ps(v_wind,       v_wind_pred,   lin);
    lin = _mm256_fmadd_ps(v_elev_tm,    v_elev_pred,   lin);
    lin = _mm256_fmadd_ps(v_slope,      v_slope_pred,  lin);

    // 9) sigmoide vectorizada: upper/(1+exp(-lin))
    __m256 expv    = _mm256_exp_ps( _mm256_sub_ps(_mm256_setzero_ps(), lin) );
    __m256 probv   = _mm256_div_ps(v_upper, _mm256_add_ps(_mm256_set1_ps(1.f), expv));

    // 10) mascarillas: vt==NONE o burnable_cell==false → 0
    __m256i mask_none = _mm256_cmpeq_epi32(vt_idx, _mm256_set1_epi32(NONE));
    alignas(32) uint32_t bm[8];
    for(int i = 0; i < 8; ++i) bm[i] = burnable_cell[i] ? 0xFFFFFFFFu : 0;
    __m256i mask_burn = _mm256_load_si256((const __m256i*)bm);
    __m256i final_m  = _mm256_andnot_si256(
        _mm256_or_si256(mask_none, _mm256_xor_si256(mask_burn, _mm256_set1_epi32(-1))),
        _mm256_set1_epi32(-1)
    );
    __m256  m_ps     = _mm256_castsi256_ps(final_m);

    // 11) aplicar máscara y almacenar
    __m256 result = _mm256_and_ps(probv, m_ps);
    _mm256_storeu_ps(probs, result);
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

      for (size_t i = 0; i < 8; i++) {
        if ((mask >> i) & 1) {  // If the cell should burn
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