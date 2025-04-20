#pragma once

#include "csv.hpp"
#include "matrix.hpp"

// enum of vegetation type between: matorral, subalpine, wet, dry
enum VegetationType {
  MATORRAL,
  SUBALPINE,
  WET,
  DRY,
  NONE
} __attribute__((packed)); // packed so that it takes only 1 byte

static_assert( sizeof(VegetationType) == 1 );

struct Cell {
  float elevation;
  float wind_direction;
  bool burnable;
  VegetationType vegetation_type;
  float fwi;
  float aspect;
};

struct Landscape {
  size_t width;
  size_t height;
  Matrix<float> elevations;
  Matrix<float> wind_directions;
  Matrix<bool> burnables;
  Matrix<VegetationType> vegetation_types;
  Matrix<float> fwis;
  Matrix<float> aspects;

  Landscape(size_t width, size_t height);
  Landscape(std::string metadata_filename, std::string data_filename);

  Cell operator[](std::pair<size_t, size_t> index) const;

};
