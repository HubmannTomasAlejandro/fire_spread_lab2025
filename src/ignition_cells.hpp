#pragma once

#include <string>
#include <vector>

struct IgnitionPair {
    size_t first;   
    size_t second;  

    __host__ __device__
    IgnitionCell() = default;

    __host__ __device__
    IgnitionCell(size_t f, size_t s)
      : first(f), second(s) {}
};


typedef std::vector<IgnitionPair> IgnitionCells;

IgnitionCells read_ignition_cells(std::string filename);
