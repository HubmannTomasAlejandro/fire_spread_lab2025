#pragma once

#include "csv.hpp"
#include "matrix.hpp"
#include "ignition_cells.hpp"

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
  Cell* cells;

  // constructors & destructor
  Landscape(size_t width_, size_t height_);
  Landscape(const std::string& metadata_filename,
            const std::string& data_filename);
  ~Landscape();

  // element access
  Cell& operator[](const std::pair<size_t, size_t>& index);
  Cell  operator[](const std::pair<size_t, size_t>& index) const;
};