#include "many_simulations.hpp"

#include <cmath>
#include <cassert>  // for assert
#include <omp.h> // omp_get_wtime()

Matrix<size_t> burned_amounts_per_cell(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, double distance, double elevation_mean, double elevation_sd,
    double upper_limit, size_t n_replicates
) {

  Matrix<size_t> burned_amounts(landscape.width, landscape.height);

  double t = omp_get_wtime();

  unsigned int amount_of_burned_cells = 0;



  for (size_t i = 0; i < n_replicates; i++) {
    Fire fire = simulate_fire(
        landscape, ignition_cells, params, distance, elevation_mean, elevation_sd, upper_limit
    );

    for (size_t j = 0; j < landscape.width * landscape.height; j++) {
        bool was_burned = fire.burned_layer.data()[j];
        size_t col = j % landscape.width;
        size_t row = j / landscape.width;
        if (was_burned) {
          amount_of_burned_cells++;
          burned_amounts[{col, row}] += 1;
        }
      }
  }


  fprintf(stderr,"burned_amounts:\n");

  fprintf(stderr,"cells_burned_per_micro_sec: %lf\n",
    amount_of_burned_cells / ((omp_get_wtime() - t) * 1e6));


  return burned_amounts;
}
