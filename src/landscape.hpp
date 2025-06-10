#pragma once

#include "csv.hpp"
#include "matrix.hpp"

// enum of vegetation type between: matorral, subalpine, wet, dry
enum VegetationType {
  MATORRAL,
  SUBALPINE,
  WET,
  DRY
} __attribute__((packed)); // packed so that it takes only 1 byte

static_assert( sizeof(VegetationType) == 1 );

struct Cell {
  short elevation;
  float wind_direction;
  bool burnable;
  VegetationType vegetation_type;
  float fwi;
  float aspect;
};

struct Landscape {
  size_t width;
  size_t height;

  Landscape(size_t width, size_t height);
  Landscape(std::string metadata_filename, std::string data_filename);

  Cell operator[](std::pair<size_t, size_t> index) const;
  Cell& operator[](std::pair<size_t, size_t> index);

  Matrix<Cell> cells;
};

struct DeviceCell {
  short elevation;
  float wind_direction;
  uint8_t burnable;      // ocupa 1 byte
  int vegetation_type;
  float fwi;
  float aspect;
};