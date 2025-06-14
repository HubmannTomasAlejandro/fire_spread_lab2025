#include "landscape.hpp"

#include <fstream>
#include <cstddef>
#include <string>

Landscape::Landscape(size_t width_, size_t height_)
    : width(width_), height(height_) {
  cells = NULL;
  // Optionally initialize to zero:
  // std::memset(cells, 0, width * height * sizeof(Cell));
}

Landscape::Landscape(const std::string& metadata_filename,
                     const std::string& data_filename)
    : cells(nullptr) {

  std::ifstream metadata_file(metadata_filename);
  if (!metadata_file.is_open()) {
    throw std::runtime_error("Can't open metadata file");
  }

  CSVIterator metadata_csv(metadata_file);
  ++metadata_csv;
  if (metadata_csv == CSVIterator() || (*metadata_csv).size() < 2) {
    throw std::runtime_error("Invalid metadata file");
  }

  width = atoi((*metadata_csv)[0].data());
  height = atoi((*metadata_csv)[1].data());

  cells = new Cell[width * height];

  metadata_file.close();

  std::ifstream landscape_file(data_filename);

  if (!landscape_file.is_open()) {
    throw std::runtime_error("Can't open landscape file");
  }

  CSVIterator loop_csv(landscape_file);
  ++loop_csv;

  for (size_t j = 0; j < height; j++) {
    for (size_t i = 0; i < width; i++, ++loop_csv) {
      if (loop_csv == CSVIterator() || (*loop_csv).size() < 8) {
        throw std::runtime_error("Invalid landscape file");
      }
      size_t idx = j * width + i;
      if (atoi((*loop_csv)[0].data()) == 1) {
        cells[idx].vegetation_type = SUBALPINE;
      } else if (atoi((*loop_csv)[1].data()) == 1) {
        cells[idx].vegetation_type = WET;
      } else if (atoi((*loop_csv)[2].data()) == 1) {
        cells[idx].vegetation_type = DRY;
      } else {
        cells[idx].vegetation_type = MATORRAL;
      }
      cells[idx].fwi = atof((*loop_csv)[3].data());
      cells[idx].aspect = atof((*loop_csv)[4].data());
      cells[idx].wind_direction = atof((*loop_csv)[5].data());
      cells[idx].elevation = atof((*loop_csv)[6].data());
      cells[idx].burnable = atoi((*loop_csv)[7].data());
    }
  }

  landscape_file.close();
}

Landscape::~Landscape() {
  delete[] cells;
}

Cell& Landscape::operator[](const std::pair<size_t, size_t>& index) {
  size_t x = index.first;
  size_t y = index.second;
  return cells[y * width + x];
}

Cell Landscape::operator[](const std::pair<size_t, size_t>& index) const {
  // reuse non-const version
  return const_cast<Landscape*>(this)->operator[](index);
}
