#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>
#include <cstdint>
#include "constants.hpp"  

// --- Data structures on device/host ---
struct SimulationParams {
    float independent_pred;
    float wind_pred;
    float elevation_pred;
    float slope_pred;
    float subalpine_pred;
    float wet_pred;
    float dry_pred;
    float fwi_pred;
    float aspect_pred;
};

enum VegetationType : uint8_t {
    MATORRAL = 0,
    SUBALPINE,
    WET,
    DRY
};

struct Cell {
    short elevation;
    float wind_direction;
    bool burnable;
    VegetationType vegetation_type;
    float fwi;
    float aspect;
};

// --- Device functions ---
__device__ inline void spread_probability(
    const Cell* __restrict__ grid,
    int width, int height,
    int base_x, int base_y,
    int neighbours_x[8], int neighbours_y[8],
    const SimulationParams params,
    float distance,
    float elev_mean,
    float elev_sd,
    float* __restrict__ probs,
    unsigned char burnable_mask,
    float upper_limit = 1.0f
) {
    // predictor by vegetation type
    float veg_pred[4] = {
        0.f,
        params.subalpine_pred,
        params.wet_pred,
        params.dry_pred
    };

    float elevs[8], fwis[8], aspects[8];
    uint8_t vegs[8];
    float mask_f[8];

    // gather neighbor data
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = neighbours_y[i] * width + neighbours_x[i];
        Cell c = grid[idx];
        elevs[i]       = float(c.elevation);
        fwis[i]        = c.fwi;
        aspects[i]     = c.aspect;
        vegs[i]        = uint8_t(c.vegetation_type);
        mask_f[i]      = ((burnable_mask >> i) & 1) ? 1.f : 0.f;
    }

    // load broadcasted burning cell data
    Cell bcell = grid[ base_y * width + base_x ];
    float b_elev = float(bcell.elevation);
    float b_dir  = bcell.wind_direction;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float diff  = elevs[i] - b_elev;
        float slope = diff / distance;
        float sin_slope = sinf(atanf(slope));
        float wind_term  = cosf(ANGLES[i] - b_dir);
        float elev_term  = (elevs[i] - elev_mean) / elev_sd;

        // linear predictor
        float lin = params.independent_pred
            + veg_pred[ vegs[i] ]
            + params.fwi_pred     * fwis[i]
            + params.aspect_pred  * aspects[i]
            + params.wind_pred    * wind_term
            + params.elevation_pred * elev_term
            + params.slope_pred   * sin_slope;

        // logistic
        float prob = upper_limit / (1.0f + expf(-lin));
        probs[i] = prob * mask_f[i];
    }
}

// kernel to process one burning cell
__global__ void propagate_kernel(
    const Cell* grid,
    int width, int height,
    const SimulationParams params,
    float distance, float elev_mean, float elev_sd,
    int* d_neigh_x, int* d_neigh_y,
    bool* d_burned,
    int num_active,
    int* active_x,
    int* active_y,
    int* new_active_x,
    int* new_active_y,
    int* new_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_active) return;

    int bx = active_x[idx];
    int by = active_y[idx];
    unsigned char mask = 0;
    int nxs[8], nys[8];

    // compute neighbours
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int nx = bx + MOVES[i][0];
        int ny = by + MOVES[i][1];
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
            nxs[i] = bx; nys[i] = by;
        } else {
            nxs[i] = nx; nys[i] = ny;
            int nidx = ny * width + nx;
            if (!d_burned[nidx] && grid[nidx].burnable)
                mask |= (1 << i);
        }
    }

    if (mask == 0) return;

    float probs[8];
    spread_probability(
        grid, width, height,
        bx, by,
        nxs, nys,
        params,
        distance, elev_mean, elev_sd,
        probs,
        mask
    );

    // RNG: simple LCG per thread
    uint state = idx ^ 0xA3C59AC3;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float rnd = (state = state * 1664525u + 1013904223u) / float(0xFFFFFFFFu);
        if ( (mask & (1<<i)) && rnd < probs[i] ) {
            int out_idx = atomicAdd(new_count, 1);
            new_active_x[out_idx] = nxs[i];
            new_active_y[out_idx] = nys[i];
            d_burned[ nys[i] * width + nxs[i] ] = true;
        }
    }
}

// --- Host wrapper ---
Fire simulate_fire_cuda(
    const std::vector<Cell>& h_cells,
    size_t width, size_t height,
    const std::vector<std::pair<size_t,size_t>>& ignition,
    const SimulationParams& params,
    float distance, float elev_mean, float elev_sd
) {
    size_t N = width * height;
    Cell* d_cells;
    bool* d_burned;

    cudaMalloc(&d_cells, N * sizeof(Cell));
    cudaMemcpy(d_cells, h_cells.data(), N * sizeof(Cell), cudaMemcpyHostToDevice);

    cudaMalloc(&d_burned, N * sizeof(bool));
    cudaMemset(d_burned, 0, N * sizeof(bool));

    // allocate active lists
    int max_cells = N;
    int *d_act_x0, *d_act_y0, *d_act_x1, *d_act_y1;
    cudaMalloc(&d_act_x0, max_cells * sizeof(int));
    cudaMalloc(&d_act_y0, max_cells * sizeof(int));
    cudaMalloc(&d_act_x1, max_cells * sizeof(int));
    cudaMalloc(&d_act_y1, max_cells * sizeof(int));
    int *d_new_count;
    cudaMalloc(&d_new_count, sizeof(int));

    // copy ignition
    int h_count = ignition.size();
    std::vector<int> h_ix(h_count), h_iy(h_count);
    for (int i = 0; i < h_count; ++i) {
        h_ix[i] = ignition[i].first;
        h_iy[i] = ignition[i].second;
        int idx = h_iy[i]*width + h_ix[i];
        // mark burned
        cudaMemcpy(d_burned + idx, &(bool){true}, sizeof(bool), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_act_x0, h_ix.data(), h_count*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_act_y0, h_iy.data(), h_count*sizeof(int), cudaMemcpyHostToDevice);

    int cur_count = h_count;
    int *active_x = d_act_x0, *active_y = d_act_y0;
    int *next_x   = d_act_x1, *next_y   = d_act_y1;

    std::vector<std::pair<size_t,size_t>> burned_list = ignition;
    std::vector<size_t> steps = { (size_t)h_count };

    while (cur_count > 0) {
        cudaMemset(d_new_count, 0, sizeof(int));
        int threads = 128;
        int blocks = (cur_count + threads - 1) / threads;
        propagate_kernel<<<blocks,threads>>>(
            d_cells, width, height,
            params, distance, elev_mean, elev_sd,
            nullptr,nullptr,
            d_burned,
            cur_count,
            active_x, active_y,
            next_x, next_y,
            d_new_count
        );
        cudaMemcpy(&cur_count, d_new_count, sizeof(int), cudaMemcpyDeviceToHost);
        // copy new actives to host and append
        std::vector<int> h_nx(cur_count), h_ny(cur_count);
        cudaMemcpy(h_nx.data(), next_x, cur_count*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ny.data(), next_y, cur_count*sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < cur_count; ++i)
            burned_list.emplace_back(h_nx[i], h_ny[i]);
        steps.push_back(burned_list.size());
        // swap buffers
        std::swap(active_x, next_x);
        std::swap(active_y, next_y);
    }

    // Liberar memoria
    cudaFree(d_cells);
    cudaFree(d_burned);
    cudaFree(d_act_x0); cudaFree(d_act_y0);
    cudaFree(d_act_x1); cudaFree(d_act_y1);
    cudaFree(d_new_count);

    return { (size_t)width, (size_t)height, /*burned_bin not returned*/ {}, burned_list, steps };
}
