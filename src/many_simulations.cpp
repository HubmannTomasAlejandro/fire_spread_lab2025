#include "many_simulations.hpp"

#include <cmath>
#include <omp.h> // omp_get_wtime()
#include <algorithm>    // For std::transform
#include <functional>   // For std::plus
#include <vector>


#pragma omp declare reduction(vec_size_t_plus : std::vector<size_t> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<size_t>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

Matrix<size_t> burned_amounts_per_cell(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, double distance, double elevation_mean, double elevation_sd,
    double upper_limit, size_t n_replicates
) {

  std::vector<size_t> burned_amounts(landscape.width * landscape.height, 0);

  double t = omp_get_wtime();

  unsigned int amount_of_burned_cells = 0;
  #pragma omp parallel reduction(vec_size_t_plus: burned_amounts) reduction(+: amount_of_burned_cells)
  {
      // Each thread gets its own private copy of burned_amounts

      #pragma omp for
      for (size_t i = 0; i < n_replicates; ++i) {
          Fire fire = simulate_fire(
              landscape, ignition_cells, params, distance, elevation_mean, elevation_sd, upper_limit
          );

          amount_of_burned_cells += fire.burned_ids.size();

          for (size_t col = 0; col < landscape.width; ++col) {
              for (size_t row = 0; row < landscape.height; ++row) {
                  if (fire.burned_layer[{col, row}]) {
                      size_t index = col + row * landscape.width;
                      burned_amounts[index] += 1;
                  }
              }
          }
      }
  }

  fprintf(stderr,"cells_burned_per_micro_sec: %lf\n",
    amount_of_burned_cells / ((omp_get_wtime() - t) * 1e6));

  // Convert burned_amounts to Matrix<size_t>
  Matrix<size_t> burned_amounts_matrix(landscape.width, landscape.height);
  for (size_t col = 0; col < landscape.width; col++) {
    for (size_t row = 0; row < landscape.height; row++) {
      burned_amounts_matrix[{col, row}] = burned_amounts[col + row*landscape.width];
    }
  }

  return burned_amounts_matrix;
}
