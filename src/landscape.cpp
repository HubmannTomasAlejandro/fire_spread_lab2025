#include "landscape.hpp"

#include <fstream>
#include <cstddef>
#include <string>

Landscape::Landscape(size_t width, size_t height)
    : width(width),
      height(height),
      elevations(width, height),
      wind_directions(width, height),
      burnables(width, height),
      vegetation_types(width, height),
      fwis(width, height),
      aspects(width, height)
{}

Landscape::Landscape(std::string metadata_filename, std::string data_filename)
    : width(0), height(0),
      elevations(0, 0),
      wind_directions(0, 0),
      burnables(0, 0),
      vegetation_types(0, 0),
      fwis(0, 0),
      aspects(0, 0){
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


  elevations = Matrix<float>(width, height);
  wind_directions = Matrix<float>(width, height);
  burnables = Matrix<bool>(width, height);
  vegetation_types = Matrix<VegetationType>(width, height);
  fwis = Matrix<float>(width, height);
  aspects = Matrix<float>(width, height);

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
      if (atoi((*loop_csv)[0].data()) == 1) {
        vegetation_types[{i, j}] = SUBALPINE;
      } else if (atoi((*loop_csv)[1].data()) == 1) {
        vegetation_types[{i, j}] = WET;
      } else if (atoi((*loop_csv)[2].data()) == 1) {
        vegetation_types[{i, j}] = DRY;
      } else {
        vegetation_types[{i, j}] = MATORRAL;
      }
      fwis[{i, j}] = atof((*loop_csv)[3].data());
      aspects[{i, j}] = atof((*loop_csv)[4].data());
      wind_directions[{i, j}] = atof((*loop_csv)[5].data());
      elevations[{i, j}] = atof((*loop_csv)[6].data());
      burnables[{i, j}] = atoi((*loop_csv)[7].data());
    }
  }

  landscape_file.close();
}

Cell Landscape::operator[](std::pair<size_t, size_t> index) const {
  Cell cell = Cell();
  cell.elevation = elevations[index];
  cell.wind_direction = wind_directions[index];
  cell.burnable = burnables[index];
  cell.vegetation_type = vegetation_types[index];
  cell.fwi = fwis[index];
  cell.aspect = aspects[index];
  return cell;
}
