
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

// CSR structure for graph representation
typedef struct {
    uint64_t* vertex_csr;
    uint64_t* edge_csr;
    uint64_t num_vertices;
    uint64_t num_edges;
    uint64_t max_degree;
} CSR;

// EF Layout structure for compression
typedef struct {
    CSR* csr;
    uint64_t num_vertices;
    uint64_t num_edges;
    uint64_t* chunk_off_exsum;
    uint64_t offset_size;
    uint64_t data_size;
    uint64_t degree_size;
    uint64_t num_lower_bits_size;
    uint64_t storage_size;
} EFLayout;

// Helper functions
uint64_t get_unsigned_type_size(uint64_t val) {
    if (val < (1ULL << 8))
        return 1;
    else if (val < (1ULL << 16))
        return 2;
    else if (val < (1ULL << 32))
        return 4;
    else
        return 8;
}

uint64_t ef_encoded_size(uint64_t* neighbors, uint64_t n) {
    if (n == 0)
        return 0;

    uint64_t U = neighbors[n-1] + 1;
    double log_ratio = log2(fmax(U / (double)n, 2.0));
    double bits = n * (log_ratio + 2);
    return (uint64_t)ceil(bits / 8);
}

uint64_t get_max_degree(CSR* csr) {
    uint64_t max_degree = 0;
    for (uint64_t i = 0; i < csr->num_vertices; i++) {
        uint64_t start = csr->vertex_csr[i];
        uint64_t end = csr->vertex_csr[i + 1];
        uint64_t degree = end - start;
        if (degree > max_degree)
            max_degree = degree;
    }
    return max_degree;
}

// Initialize CSR structure
CSR* initialize_csr(uint64_t* vertex_csr, uint64_t* edge_csr, uint64_t num_vertices, uint64_t num_edges) {
    CSR* csr = (CSR*)malloc(sizeof(CSR));
    csr->vertex_csr = vertex_csr;
    csr->edge_csr = edge_csr;
    csr->num_vertices = num_vertices;
    csr->num_edges = num_edges;
    csr->max_degree = get_max_degree(csr);
    return csr;
}

// Get neighbors for a vertex
uint64_t* get_nbrs(CSR* csr, uint64_t vid, uint64_t* count) {
    if (vid >= csr->num_vertices) {
        *count = 0;
        return NULL;
    }

    uint64_t start = csr->vertex_csr[vid];
    uint64_t end = csr->vertex_csr[vid + 1];
    *count = end - start;

    if (*count == 0)
        return NULL;

    return &(csr->edge_csr[start]);
}

// Initialize EF Layout
EFLayout* initialize_ef_layout(CSR* csr) {
    EFLayout* ef = (EFLayout*)malloc(sizeof(EFLayout));
    ef->csr = csr;
    ef->num_vertices = csr->num_vertices;
    ef->num_edges = csr->num_edges;

    uint64_t val_type_size = get_unsigned_type_size(ef->num_vertices);
    uint64_t edge_type_size = get_unsigned_type_size(ef->num_edges);
    uint64_t max_degree = csr->max_degree;
    uint64_t degree_type_size = get_unsigned_type_size(max_degree);
    uint64_t fwd_type_size = get_unsigned_type_size(max_degree * 2);
    uint64_t chunk_align_size = fwd_type_size;

    // Temporary array to store sorted neighbors
    uint64_t* sorted_neighbors = (uint64_t*)malloc(max_degree * sizeof(uint64_t));
    uint64_t* num_chunks_vec = (uint64_t*)malloc(ef->num_vertices * sizeof(uint64_t));

    for (uint64_t i = 0; i < ef->num_vertices; i++) {
        uint64_t count;
        uint64_t* neighbors = get_nbrs(csr, i, &count);

        // Copy and sort neighbors
        if (count > 0) {
            memcpy(sorted_neighbors, neighbors, count * sizeof(uint64_t));
            // Simple insertion sort (for small neighborhoods)
            for (uint64_t j = 1; j < count; j++) {
                uint64_t key = sorted_neighbors[j];
                int64_t k = j - 1;
                while (k >= 0 && sorted_neighbors[k] > key) {
                    sorted_neighbors[k + 1] = sorted_neighbors[k];
                    k--;
                }
                sorted_neighbors[k + 1] = key;
            }
        }

        uint64_t encoded_size = ef_encoded_size(sorted_neighbors, count);
        uint64_t num_chunks = (encoded_size + chunk_align_size - 1) / chunk_align_size;
        num_chunks_vec[i] = num_chunks;
    }

    // Calculate exclusive sum
    ef->chunk_off_exsum = (uint64_t*)malloc((ef->num_vertices + 1) * sizeof(uint64_t));
    ef->chunk_off_exsum[0] = 0;

    for (uint64_t i = 0; i < ef->num_vertices; i++) {
        ef->chunk_off_exsum[i + 1] = ef->chunk_off_exsum[i] + num_chunks_vec[i];
    }

    uint64_t total_chunks = ef->chunk_off_exsum[ef->num_vertices];
    uint64_t chunk_offset_size = get_unsigned_type_size(total_chunks);

    ef->offset_size = (ef->num_vertices + 1) * chunk_offset_size;
    ef->data_size = total_chunks * chunk_align_size;
    ef->degree_size = degree_type_size * ef->num_vertices;
    ef->num_lower_bits_size = 1 * ef->num_vertices;
    ef->storage_size = ef->offset_size + ef->data_size + ef->degree_size + ef->num_lower_bits_size;

    free(sorted_neighbors);
    free(num_chunks_vec);

    return ef;
}

// Get optimal CSR size
uint64_t get_optimal_csr_size(EFLayout* ef) {
    uint64_t row_offset_size = get_unsigned_type_size(ef->num_edges + 1);
    uint64_t col_ind_size = get_unsigned_type_size(ef->num_vertices);
    return (ef->num_vertices + 1) * row_offset_size + (ef->num_edges * col_ind_size);
}

// Read binary CSR files
void read_binary_csr(const char* vertex_file, const char* edge_file, uint64_t** vertex_csr, uint64_t** edge_csr, uint64_t* num_vertices, uint64_t* num_edges) {
    FILE* vf = fopen(vertex_file, "rb");
    if (!vf) {
        fprintf(stderr, "Error opening vertex file: %s\n", vertex_file);
        exit(1);
    }

    // Get file size and calculate number of vertices
    fseek(vf, 0, SEEK_END);
    long vf_size = ftell(vf);
    rewind(vf);
    *num_vertices = vf_size / sizeof(uint64_t) - 1;

    // Allocate and read vertex CSR
    *vertex_csr = (uint64_t*)malloc(vf_size);
    if (fread(*vertex_csr, 1, vf_size, vf) != vf_size) {
        fprintf(stderr, "Error reading vertex file\n");
        fclose(vf);
        exit(1);
    }
    fclose(vf);

    FILE* ef = fopen(edge_file, "rb");
    if (!ef) {
        fprintf(stderr, "Error opening edge file: %s\n", edge_file);
        free(*vertex_csr);
        exit(1);
    }

    // Get file size and calculate number of edges
    fseek(ef, 0, SEEK_END);
    long ef_size = ftell(ef);
    rewind(ef);
    *num_edges = ef_size / sizeof(uint64_t);

    // Allocate and read edge CSR
    *edge_csr = (uint64_t*)malloc(ef_size);
    if (fread(*edge_csr, 1, ef_size, ef) != ef_size) {
        fprintf(stderr, "Error reading edge file\n");
        fclose(ef);
        free(*vertex_csr);
        exit(1);
    }
    fclose(ef);
}

// CUDA kernel for BFS
__global__ void bfs_kernel(uint64_t* vertex_csr, uint64_t* edge_csr, int* distance, bool* visited,
                           bool* frontier, bool* next_frontier, uint64_t num_vertices, unsigned long long* edge_count) {
    uint64_t v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v < num_vertices && frontier[v]) {
        frontier[v] = false;

        uint64_t start = vertex_csr[v];
        uint64_t end = vertex_csr[v + 1];
        uint64_t edge_traversed = end - start;

        // Count traversed edges once for this vertex
        if (edge_traversed > 0) {
            atomicAdd(edge_count, edge_traversed);
        }

        for (uint64_t i = start; i < end; i++) {
            uint64_t neighbor = edge_csr[i];

            if (!visited[neighbor]) {
                visited[neighbor] = true;
                distance[neighbor] = distance[v] + 1;
                next_frontier[neighbor] = true;
            }
        }
    }
}

// BFS on GPU
uint64_t sssp_bfs_gpu(CSR* csr, uint64_t source) {
    uint64_t num_vertices = csr->num_vertices;

    // Check if source vertex is valid and has neighbors
    if (source >= num_vertices) {
        printf("Invalid source vertex %lu (max: %lu)\n", source, num_vertices - 1);
        return 0;
    }

    uint64_t start = csr->vertex_csr[source];
    uint64_t end = csr->vertex_csr[source + 1];
    if (start == end) {
        printf("Source vertex %lu has no outgoing edges\n", source);
        return 0;
    }

    // Host arrays
    int* h_distance = (int*)malloc(num_vertices * sizeof(int));
    bool* h_visited = (bool*)malloc(num_vertices * sizeof(bool));

    memset(h_distance, -1, num_vertices * sizeof(int));
    memset(h_visited, 0, num_vertices * sizeof(bool));

    h_distance[source] = 0;
    h_visited[source] = true;

    // Device arrays
    uint64_t* d_vertex_csr;
    uint64_t* d_edge_csr;
    int* d_distance;
    bool* d_visited;
    bool* d_frontier;
    bool* d_next_frontier;
    unsigned long long* d_edge_count;

    // Allocate memory on GPU
    cudaMalloc(&d_vertex_csr, (num_vertices + 1) * sizeof(uint64_t));
    cudaMalloc(&d_edge_csr, csr->num_edges * sizeof(uint64_t));
    cudaMalloc(&d_distance, num_vertices * sizeof(int));
    cudaMalloc(&d_visited, num_vertices * sizeof(bool));
    cudaMalloc(&d_frontier, num_vertices * sizeof(bool));
    cudaMalloc(&d_next_frontier, num_vertices * sizeof(bool));
    cudaMalloc(&d_edge_count, sizeof(unsigned long long));

    // Copy data to GPU
    cudaMemcpy(d_vertex_csr, csr->vertex_csr, (num_vertices + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_csr, csr->edge_csr, csr->num_edges * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance, h_distance, num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited, num_vertices * sizeof(bool), cudaMemcpyHostToDevice);

    // Initialize frontiers
    bool* h_frontier = (bool*)calloc(num_vertices, sizeof(bool));
    h_frontier[source] = true;

    cudaMemcpy(d_frontier, h_frontier, num_vertices * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemset(d_next_frontier, 0, num_vertices * sizeof(bool));

    // Initialize edge count to zero
    unsigned long long h_edge_count = 0;
    cudaMemcpy(d_edge_count, &h_edge_count, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    // BFS parameters
    int blockSize = 256;
    int numBlocks = (num_vertices + blockSize - 1) / blockSize;

    // Debug source vertex neighbors
    printf("Source %lu has %lu neighbors\n", source, end - start);

    // BFS iterations
    bool has_frontier = true;
    int iterations = 0;
    while (has_frontier && iterations < 100) {  // Add iteration limit to avoid infinite loops
        iterations++;

        bfs_kernel<<<numBlocks, blockSize>>>(d_vertex_csr, d_edge_csr, d_distance, d_visited,
                                            d_frontier, d_next_frontier, num_vertices, d_edge_count);
        cudaDeviceSynchronize();

        // Check CUDA errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            break;
        }

        // Check if next frontier is empty
        cudaMemcpy(h_frontier, d_next_frontier, num_vertices * sizeof(bool), cudaMemcpyDeviceToHost);

        has_frontier = false;
        for (uint64_t i = 0; i < num_vertices; i++) {
            if (h_frontier[i]) {
                has_frontier = true;
                break;
            }
        }

        // Swap frontiers
        bool* temp = d_frontier;
        d_frontier = d_next_frontier;
        d_next_frontier = temp;

        cudaMemset(d_next_frontier, 0, num_vertices * sizeof(bool));
    }

    // Get final edge count
    cudaMemcpy(&h_edge_count, d_edge_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    printf("BFS completed in %d iterations\n", iterations);
    printf("Edge count from BFS: %llu\n", h_edge_count);

    // Validate results (optional)
    int* final_distance = (int*)malloc(num_vertices * sizeof(int));
    cudaMemcpy(final_distance, d_distance, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);

    int reachable = 0;
    for (uint64_t i = 0; i < num_vertices; i++) {
        if (final_distance[i] >= 0) {
            reachable++;
        }
    }
    printf("Reachable nodes: %d of %lu\n", reachable, num_vertices);

    free(final_distance);
    free(h_distance);
    free(h_visited);
    free(h_frontier);

    cudaFree(d_vertex_csr);
    cudaFree(d_edge_csr);
    cudaFree(d_distance);
    cudaFree(d_visited);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_edge_count);

    return (uint64_t)h_edge_count;
}

// Generate random indices of connected vertices
void get_connected_vertices(CSR* csr, uint64_t* connected, uint64_t* count) {
    *count = 0;

    for (uint64_t i = 0; i < csr->num_vertices; i++) {
        uint64_t start = csr->vertex_csr[i];
        uint64_t end = csr->vertex_csr[i + 1];

        if (end > start) {
            connected[(*count)++] = i;

            // Debug: print some connected vertices and their degrees
            if (*count <= 5) {
                printf("Debug: Vertex %lu has %lu neighbors\n", i, end - start);
            }
        }
    }

    printf("Found %lu connected vertices out of %lu total vertices\n", *count, csr->num_vertices);
}

// Average GTEPS calculation
void average_gteps(CSR* csr, int trials) {
    uint64_t num_vertices = csr->num_vertices;

    // Find connected vertices
    uint64_t* connected_vertices = (uint64_t*)malloc(num_vertices * sizeof(uint64_t));
    uint64_t connected_count;

    get_connected_vertices(csr, connected_vertices, &connected_count);

    if (connected_count == 0) {
        printf("Error: No connected vertices found in the graph!\n");
        free(connected_vertices);
        return;
    }

    if (connected_count < trials) {
        printf("Only %lu connected vertices found, reducing trials to %lu.\n", connected_count, connected_count);
        trials = connected_count;
    }

    // Print some diagnostics about the graph
    uint64_t total_edges = 0;
    uint64_t max_degree = 0;
    uint64_t vertex_with_max_degree = 0;

    for (uint64_t i = 0; i < num_vertices; i++) {
        uint64_t degree = csr->vertex_csr[i+1] - csr->vertex_csr[i];
        total_edges += degree;

        if (degree > max_degree) {
            max_degree = degree;
            vertex_with_max_degree = i;
        }
    }

    printf("Graph stats: %lu vertices, %lu edges, average degree: %.2f\n",
           num_vertices, total_edges, (double)total_edges / num_vertices);
    printf("Max degree: %lu at vertex %lu\n", max_degree, vertex_with_max_degree);

    // Random sampling of source vertices - use vertices with significant degree
    uint64_t* random_sources = (uint64_t*)malloc(trials * sizeof(uint64_t));
    srand(time(NULL));

    // Find high-degree vertices for testing
    uint64_t min_degree_threshold = 10;  // Only use vertices with at least this many neighbors
    uint64_t high_degree_count = 0;
    uint64_t* high_degree_vertices = (uint64_t*)malloc(connected_count * sizeof(uint64_t));

    for (uint64_t i = 0; i < connected_count; i++) {
        uint64_t v = connected_vertices[i];
        uint64_t degree = csr->vertex_csr[v+1] - csr->vertex_csr[v];

        if (degree >= min_degree_threshold) {
            high_degree_vertices[high_degree_count++] = v;
        }
    }

    if (high_degree_count > 0) {
        printf("Found %lu vertices with degree >= %lu\n", high_degree_count, min_degree_threshold);

        int actual_trials = trials < high_degree_count ? trials : high_degree_count;

        // Choose random high-degree vertices
        for (int i = 0; i < actual_trials; i++) {
            uint64_t idx = rand() % high_degree_count;
            random_sources[i] = high_degree_vertices[idx];
        }
    } else {
        printf("No vertices with degree >= %lu found, using random connected vertices\n", min_degree_threshold);

        // Choose random connected vertices
        for (int i = 0; i < trials; i++) {
            uint64_t random_index = rand() % connected_count;
            random_sources[i] = connected_vertices[random_index];
        }
    }

    double total_gteps = 0.0;
    int successful_runs = 0;

    for (int i = 0; i < trials; i++) {
        uint64_t src = random_sources[i];

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        uint64_t edges = sssp_bfs_gpu(csr, src);

        clock_gettime(CLOCK_MONOTONIC, &end);

        // Skip vertices that result in zero edges traversed
        if (edges == 0) {
            printf("Run %d from vertex %lu: Skipped (0 edges traversed)\n", i+1, src);
            continue;
        }

        double duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        // Avoid division by zero
        if (duration < 1e-9) {
            printf("Run %d from vertex %lu: Skipped (duration too short)\n", i+1, src);
            continue;
        }

        double gteps = edges / (duration * 1e9);
        total_gteps += gteps;
        successful_runs++;

        printf("Run %d from vertex %lu: %.6f GTEPS (%lu edges in %.6f seconds)\n",
               i+1, src, gteps, edges, duration);
    }

    if (successful_runs > 0) {
        double avg_gteps = total_gteps / successful_runs;
        printf("Average GTEPS over %d successful runs: %.6f\n", successful_runs, avg_gteps);
    } else {
        printf("No successful runs to average.\n");
    }

    free(connected_vertices);
    free(random_sources);
    free(high_degree_vertices);
}

int main() {
    uint64_t *vertex_csr, *edge_csr;
    uint64_t num_vertices, num_edges;

    // Read CSR files
    read_binary_csr("vertex.CSR", "edge.CSR", &vertex_csr, &edge_csr, &num_vertices, &num_edges);

    // Initialize CSR structure
    CSR* csr = initialize_csr(vertex_csr, edge_csr, num_vertices, num_edges);

    // Initialize EF layout for compression analysis
    EFLayout* ef_layout = initialize_ef_layout(csr);

    // Print statistics
    printf("Vertices: %lu, Edges: %lu\n", csr->num_vertices, csr->num_edges);
    printf("Optimal CSR Size: %lu bytes\n", get_optimal_csr_size(ef_layout));
    printf("Compressed (Elias-Fano) Size: %lu bytes\n", ef_layout->storage_size);
    printf("Compression Ratio (CSR / EF): %.4f\n",
           (double)get_optimal_csr_size(ef_layout) / ef_layout->storage_size);

    printf("\n--- SSSP Average GTEPS Benchmark ---\n");
    // Reduce the number of trials to avoid excessive computation
    average_gteps(csr, 100);

    // Free memory
    free(vertex_csr);
    free(edge_csr);
    free(ef_layout->chunk_off_exsum);
    free(ef_layout);
    free(csr);

    return 0;
}
