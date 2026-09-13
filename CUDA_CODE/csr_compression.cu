#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

// CSR structure
typedef struct {
    uint64_t* vertex_csr;
    uint64_t* edge_csr;
    uint64_t num_vertices;
    uint64_t num_edges;
} CSR;

// EF Layout structure
typedef struct {
    CSR csr;
    uint64_t num_vertices;
    uint64_t num_edges;
    uint64_t* chunk_off_exsum;
    uint64_t offset_size;
    uint64_t data_size;
    uint64_t degree_size;
    uint64_t num_lower_bits_size;
    uint64_t storage_size;
} EFLayout;

// Get unsigned type size (similar to Python implementation)
__host__ __device__ uint8_t get_unsigned_type_size(uint64_t val) {
    if (val < (1ULL << 8)) return 1;
    else if (val < (1ULL << 16)) return 2;
    else if (val < (1ULL << 32)) return 4;
    else return 8;
}

// Helper function to compute bit width
__host__ __device__ int bitWidth(uint64_t value) {
    if (value == 0) return 1;
    return (int)(log2(value) + 1);
}

// Calculate the EF encoded size for a list of neighbors
// Fixed to more accurately match Python implementation
__device__ uint64_t ef_encoded_size(uint64_t* neighbors, uint64_t n) {
    if (n == 0) return 0;

    uint64_t U = neighbors[n-1] + 1;  // max + 1
    double log_ratio = log2(fmax((double)U / n, 2.0));

    // Match Python implementation more closely
    double bits = n * (log_ratio + 2);
    uint64_t bytes = (uint64_t)ceil(bits / 8.0);

    return bytes;
}

// Sort array (simple insertion sort for on-device usage)
__device__ void sort_array(uint64_t* arr, uint64_t n) {
    for (int i = 1; i < n; i++) {
        uint64_t key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Get neighbors for a vertex
__device__ void get_nbrs(uint64_t vid, uint64_t* vertex_csr, uint64_t* edge_csr,
                          uint64_t* neighbors, uint64_t* count) {
    uint64_t start = vertex_csr[vid];
    uint64_t end = vertex_csr[vid + 1];
    *count = end - start;

    for (uint64_t i = 0; i < *count; i++) {
        neighbors[i] = edge_csr[start + i];
    }
}

// CUDA kernel to compute chunk counts for each vertex
__global__ void compute_chunks_kernel(uint64_t* vertex_csr, uint64_t* edge_csr,
                                       uint64_t num_vertices, uint64_t* num_chunks_vec,
                                       uint64_t max_degree, uint64_t chunk_align_size) {
    uint64_t vid = blockIdx.x * blockDim.x + threadIdx.x;

    if (vid < num_vertices) {
        uint64_t* neighbors = (uint64_t*)malloc(max_degree * sizeof(uint64_t));
        if (neighbors == NULL) return; // Handle out of memory

        uint64_t count;
        get_nbrs(vid, vertex_csr, edge_csr, neighbors, &count);
        sort_array(neighbors, count);

        uint64_t encoded_size = ef_encoded_size(neighbors, count);
        num_chunks_vec[vid] = (encoded_size + chunk_align_size - 1) / chunk_align_size;

        free(neighbors);
    }
}

// Calculate maximum degree in CSR
uint64_t get_max_degree(CSR csr) {
    uint64_t max_degree = 0;
    for (uint64_t i = 0; i < csr.num_vertices; i++) {
        uint64_t degree = csr.vertex_csr[i+1] - csr.vertex_csr[i];
        if (degree > max_degree) {
            max_degree = degree;
        }
    }
    return max_degree;
}

// CPU version of ef_encoded_size for verification
uint64_t cpu_ef_encoded_size(uint64_t* neighbors, uint64_t n) {
    if (n == 0) return 0;

    uint64_t U = neighbors[n-1] + 1;  // max + 1
    double log_ratio = log2(fmax((double)U / n, 2.0));
    double bits = n * (log_ratio + 2);

    return (uint64_t)ceil(bits / 8.0);
}

// Setup EF layout parameters
EFLayout set_ef_params(CSR csr) {
    EFLayout ef_layout;
    ef_layout.csr = csr;
    ef_layout.num_vertices = csr.num_vertices;
    ef_layout.num_edges = csr.num_edges;

    uint8_t val_type_size = get_unsigned_type_size(csr.num_vertices);
    uint8_t edge_type_size = get_unsigned_type_size(csr.num_edges);
    uint64_t max_degree = get_max_degree(csr);
    uint8_t degree_type_size = get_unsigned_type_size(max_degree);
    uint8_t fwd_type_size = get_unsigned_type_size(max_degree * 2);

    uint64_t chunk_align_size = fwd_type_size;

    // Allocate memory for vectors
    uint64_t* num_chunks_vec = (uint64_t*)malloc(csr.num_vertices * sizeof(uint64_t));
    if (num_chunks_vec == NULL) {
        printf("Memory allocation failed for num_chunks_vec\n");
        exit(1);
    }

    // CPU-only version for debugging/fallback
    if (max_degree > 0) {
        // Set all values to zero
        for (uint64_t i = 0; i < csr.num_vertices; i++) {
            num_chunks_vec[i] = 0;
        }

        // Process each vertex
        for (uint64_t i = 0; i < csr.num_vertices; i++) {
            uint64_t start = csr.vertex_csr[i];
            uint64_t end = csr.vertex_csr[i + 1];
            uint64_t degree = end - start;

            if (degree > 0) {
                // Extract and sort neighbors
                uint64_t* neighbors = (uint64_t*)malloc(degree * sizeof(uint64_t));
                if (neighbors == NULL) {
                    printf("Memory allocation failed for neighbors array\n");
                    exit(1);
                }

                for (uint64_t j = 0; j < degree; j++) {
                    neighbors[j] = csr.edge_csr[start + j];
                }

                // Simple insertion sort
                for (uint64_t j = 1; j < degree; j++) {
                    uint64_t key = neighbors[j];
                    int64_t k = j - 1;
                    while (k >= 0 && neighbors[k] > key) {
                        neighbors[k + 1] = neighbors[k];
                        k--;
                    }
                    neighbors[k + 1] = key;
                }

                // Calculate encoding size
                uint64_t encoded_size = cpu_ef_encoded_size(neighbors, degree);
                num_chunks_vec[i] = (encoded_size + chunk_align_size - 1) / chunk_align_size;

                free(neighbors);
            }
        }
    }

    // Calculate exclusive sum for chunk offsets
    ef_layout.chunk_off_exsum = (uint64_t*)malloc((csr.num_vertices + 1) * sizeof(uint64_t));
    if (ef_layout.chunk_off_exsum == NULL) {
        printf("Memory allocation failed for chunk_off_exsum\n");
        exit(1);
    }

    ef_layout.chunk_off_exsum[0] = 0;
    for (uint64_t i = 0; i < csr.num_vertices; i++) {
        ef_layout.chunk_off_exsum[i+1] = ef_layout.chunk_off_exsum[i] + num_chunks_vec[i];
    }

    uint64_t total_chunks = ef_layout.chunk_off_exsum[csr.num_vertices];
    uint8_t chunk_offset_size = get_unsigned_type_size(total_chunks);

    // Calculate sizes more carefully matching Python implementation
    ef_layout.offset_size = (csr.num_vertices + 1) * chunk_offset_size;
    ef_layout.data_size = total_chunks * chunk_align_size;
    ef_layout.degree_size = degree_type_size * csr.num_vertices;
    ef_layout.num_lower_bits_size = 1 * csr.num_vertices;  // 1 byte per vertex
    ef_layout.storage_size = ef_layout.offset_size + ef_layout.data_size +
                             ef_layout.degree_size + ef_layout.num_lower_bits_size;

    free(num_chunks_vec);
    return ef_layout;
}

// Get optimal CSR size
uint64_t get_optimal_csr_size(EFLayout ef_layout) {
    uint8_t row_offset_size = get_unsigned_type_size(ef_layout.num_edges + 1);
    uint8_t col_ind_size = get_unsigned_type_size(ef_layout.num_vertices);
    return (ef_layout.num_vertices + 1) * row_offset_size + (ef_layout.num_edges * col_ind_size);
}

// Read binary CSR files
CSR read_binary_csr(const char* vertex_file, const char* edge_file) {
    CSR csr;
    FILE* f_vertex = fopen(vertex_file, "rb");
    FILE* f_edge = fopen(edge_file, "rb");

    if (!f_vertex || !f_edge) {
        printf("Error opening CSR files\n");
        exit(1);
    }

    // Get file sizes
    fseek(f_vertex, 0, SEEK_END);
    uint64_t vertex_size = ftell(f_vertex) / sizeof(uint64_t);
    rewind(f_vertex);

    fseek(f_edge, 0, SEEK_END);
    uint64_t edge_size = ftell(f_edge) / sizeof(uint64_t);
    rewind(f_edge);

    // Allocate memory
    csr.vertex_csr = (uint64_t*)malloc(vertex_size * sizeof(uint64_t));
    csr.edge_csr = (uint64_t*)malloc(edge_size * sizeof(uint64_t));

    if (csr.vertex_csr == NULL || csr.edge_csr == NULL) {
        printf("Memory allocation failed for CSR arrays\n");
        exit(1);
    }

    // Read data
    fread(csr.vertex_csr, sizeof(uint64_t), vertex_size, f_vertex);
    fread(csr.edge_csr, sizeof(uint64_t), edge_size, f_edge);

    csr.num_vertices = vertex_size - 1;
    csr.num_edges = edge_size;

    fclose(f_vertex);
    fclose(f_edge);

    return csr;
}

// Verify that binary files exist
bool verify_files(const char* vertex_file, const char* edge_file) {
    FILE* f_vertex = fopen(vertex_file, "rb");
    FILE* f_edge = fopen(edge_file, "rb");

    bool result = (f_vertex != NULL && f_edge != NULL);

    if (f_vertex) fclose(f_vertex);
    if (f_edge) fclose(f_edge);

    return result;
}

// Main function
int main() {
    // Verify files exist
    if (!verify_files("vertex.CSR", "edge.CSR")) {
        printf("Error: CSR files not found. Please ensure vertex.CSR and edge.CSR exist in current directory.\n");
        return 1;
    }

    // Read CSR files
    CSR csr = read_binary_csr("vertex.CSR", "edge.CSR");

    // Calculate EF Layout parameters
    EFLayout ef_layout = set_ef_params(csr);

    // Calculate sizes and compression ratio
    uint64_t optimal_csr_size = get_optimal_csr_size(ef_layout);
    uint64_t compressed_size = ef_layout.storage_size;
    double compression_ratio = (double)optimal_csr_size / compressed_size;

    // Print results
    printf("Vertices: %lu, Edges: %lu\n", csr.num_vertices, csr.num_edges);
    printf("Optimal CSR Size: %lu bytes\n", optimal_csr_size);
    printf("Compressed (Elias–Fano) Size: %lu bytes\n", compressed_size);
    printf("Compression Ratio (CSR / EF): %.4f\n", compression_ratio);

    // Free memory
    free(csr.vertex_csr);
    free(csr.edge_csr);
    free(ef_layout.chunk_off_exsum);

    return 0;
}
