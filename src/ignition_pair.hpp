#include <cstddef>
#include <cuda_runtime.h>


struct IgnitionPair {
    size_t first;   
    size_t second;  

    __host__ __device__
    IgnitionPair() = default;

    __host__ __device__
    IgnitionPair(size_t f, size_t s)
      : first(f), second(s) {};

    // Conversión implícita a std::pair
    operator std::pair<size_t, size_t>() const {
        return {first, second};
    }

    // Conversión desde std::pair
    IgnitionPair(const std::pair<size_t, size_t>& p) 
        : first(p.first), second(p.second) {};

    // Operador de comparación
    __host__ __device__
    bool operator==(const IgnitionPair& other) const {
        return first == other.first && second == other.second;
    }
};