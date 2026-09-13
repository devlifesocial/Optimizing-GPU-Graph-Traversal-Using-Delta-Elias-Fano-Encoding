
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>
#include <curand_kernel.h>

// CSR Graph Structure
typedef struct {
    unsigned long long* vertex_csr;
    unsigned long long* edge_csr;
    unsigned int num_vertices;
    unsigned int num_edges;
    unsigned int* graph_degrees;
    unsigned long long** graph_neighbors;
} CSRGraph;

// Elias-Fano Layout Structure
typedef struct {
    CSRGraph* csr;
    unsigned int num_vertices;
    unsigned int num_edges;
    unsigned long long* chunk_off_exsum;
    size_t offset_size;
    size_t data_size;
    size_t degree_size;
    size_t num_lower_bits_size;
    size_t storage_size;
} EFLayout;

// Helper function to get the appropriate unsigned type size
unsigned int get_unsigned_type_size(unsigned long long val) {
    if (val < (1ULL << 8)) return 1;
    else if (val < (1ULL << 16)) return 2;
    else if (val < (1ULL << 32)) return 4;
    else return 8;
}

// Read binary CSR files
void read_binary_csr(const char* vertex_file, const char* edge_file,
                     unsigned long long** vertex_csr, unsigned long long** edge_csr,
                     unsigned int* num_vertices, unsigned int* num_edges) {
    FILE* vf = fopen(vertex_file, "rb");
    if (!vf) {
        printf("Failed to open vertex file: %s\n", vertex_file);
        exit(1);
    }

    // Get file size for vertex_csr
    fseek(vf, 0, SEEK_END);
    size_t vertex_file_size = ftell(vf);
    rewind(vf);

    size_t vertex_count = vertex_file_size / sizeof(unsigned long long);
    *num_vertices = vertex_count - 1;

    // Allocate memory for vertex_csr
    *vertex_csr = (unsigned long long*)malloc(vertex_file_size);
    if (!(*vertex_csr)) {
        printf("Failed to allocate memory for vertex_csr\n");
        fclose(vf);
        exit(1);
    }

    // Read vertex_csr
    if (fread(*vertex_csr, sizeof(unsigned long long), vertex_count, vf) != vertex_count) {
        printf("Failed to read vertex file\n");
        free(*vertex_csr);
        fclose(vf);
        exit(1);
    }
    fclose(vf);

    // Now handle edge file
    FILE* ef = fopen(edge_file, "rb");
    if (!ef) {
        printf("Failed to open edge file: %s\n", edge_file);
        free(*vertex_csr);
        exit(1);
    }

    // Get file size for edge_csr
    fseek(ef, 0, SEEK_END);
    size_t edge_file_size = ftell(ef);
    rewind(ef);

    size_t edge_count = edge_file_size / sizeof(unsigned long long);
    *num_edges = edge_count;

    // Allocate memory for edge_csr
    *edge_csr = (unsigned long long*)malloc(edge_file_size);
    if (!(*edge_csr)) {
        printf("Failed to allocate memory for edge_csr\n");
        free(*vertex_csr);
        fclose(ef);
        exit(1);
    }

    // Read edge_csr
    if (fread(*edge_csr, sizeof(unsigned long long), edge_count, ef) != edge_count) {
        printf("Failed to read edge file\n");
        free(*vertex_csr);
        free(*edge_csr);
        fclose(ef);
        exit(1);
    }
    fclose(ef);
}

// Build graph from CSR
void build_graph(CSRGraph* graph) {
    graph->graph_degrees = (unsigned int*)malloc(graph->num_vertices * sizeof(unsigned int));
    graph->graph_neighbors = (unsigned long long**)malloc(graph->num_vertices * sizeof(unsigned long long*));

    for (unsigned int i = 0; i < graph->num_vertices; i++) {
        unsigned long long start = graph->vertex_csr[i];
        unsigned long long end = graph->vertex_csr[i + 1];
        unsigned int degree = end - start;

        graph->graph_degrees[i] = degree;

        if (degree > 0) {
            graph->graph_neighbors[i] = (unsigned long long*)malloc(degree * sizeof(unsigned long long));
            for (unsigned int j = 0; j < degree; j++) {
                graph->graph_neighbors[i][j] = graph->edge_csr[start + j];
            }
        } else {
            graph->graph_neighbors[i] = NULL;
        }
    }
}

// Calculate EF encoded size for a set of neighbors
unsigned int ef_encoded_size(unsigned long long* neighbors, unsigned int n) {
    if (n == 0) return 0;

    unsigned long long U = neighbors[n-1] + 1;
    double log_ratio = log2(fmax(U / (double)n, 2.0));
    double bits = n * (log_ratio + 2);

    return (unsigned int)ceil(bits / 8);
}

// Set Elias-Fano parameters
void set_ef_params(EFLayout* ef_layout) {
    CSRGraph* csr = ef_layout->csr;

    // Find max degree
    unsigned int max_degree = 0;
    for (unsigned int i = 0; i < csr->num_vertices; i++) {
        if (csr->graph_degrees[i] > max_degree) {
            max_degree = csr->graph_degrees[i];
        }
    }

    unsigned int val_type_size = get_unsigned_type_size(csr->num_vertices);
    unsigned int edge_type_size = get_unsigned_type_size(csr->num_edges);
    unsigned int degree_type_size = get_unsigned_type_size(max_degree);
    unsigned int fwd_type_size = get_unsigned_type_size(max_degree * 2);
    unsigned int chunk_align_size = fwd_type_size;

    // Allocate and calculate chunk offsets
    unsigned int* num_chunks_vec = (unsigned int*)malloc(csr->num_vertices * sizeof(unsigned int));

    for (unsigned int i = 0; i < csr->num_vertices; i++) {
        unsigned int encoded_size = 0;
        if (csr->graph_degrees[i] > 0) {
            // Sort neighbors
            qsort(csr->graph_neighbors[i], csr->graph_degrees[i], sizeof(unsigned long long),
                  [](const void* a, const void* b) {
                      return (*(unsigned long long*)a > *(unsigned long long*)b) ? 1 : -1;
                  });

            encoded_size = ef_encoded_size(csr->graph_neighbors[i], csr->graph_degrees[i]);
        }

        num_chunks_vec[i] = (encoded_size + chunk_align_size - 1) / chunk_align_size;
    }

    // Calculate exclusive sum for chunk offsets
    ef_layout->chunk_off_exsum = (unsigned long long*)malloc((csr->num_vertices + 1) * sizeof(unsigned long long));
    ef_layout->chunk_off_exsum[0] = 0;

    for (unsigned int i = 0; i < csr->num_vertices; i++) {
        ef_layout->chunk_off_exsum[i + 1] = ef_layout->chunk_off_exsum[i] + num_chunks_vec[i];
    }

    unsigned long long total_chunks = ef_layout->chunk_off_exsum[csr->num_vertices];
    unsigned int chunk_offset_size = get_unsigned_type_size(total_chunks);

    // Calculate storage sizes
    ef_layout->offset_size = (csr->num_vertices + 1) * chunk_offset_size;
    ef_layout->data_size = total_chunks * chunk_align_size;
    ef_layout->degree_size = degree_type_size * csr->num_vertices;
    ef_layout->num_lower_bits_size = 1 * csr->num_vertices;

    ef_layout->storage_size = ef_layout->offset_size + ef_layout->data_size +
                             ef_layout->degree_size + ef_layout->num_lower_bits_size;

    free(num_chunks_vec);
}

// Get optimal CSR size
size_t get_optimal_csr_size(EFLayout* ef_layout) {
    unsigned int row_offset_size = get_unsigned_type_size(ef_layout->num_edges + 1);
    unsigned int col_ind_size = get_unsigned_type_size(ef_layout->num_vertices);

    return (ef_layout->num_vertices + 1) * row_offset_size +
           (ef_layout->num_edges * col_ind_size);
}

// CUDA kernel for initializing random states
__global__ void init_random_states(curandState *states, unsigned long long seed, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// CUDA BFS kernel
__global__ void sssp_bfs_kernel(unsigned int* d_graph_degrees,
                               unsigned long long** d_graph_neighbors,
                               bool* d_visited, int* d_queue, int* d_queue_size,
                               int* d_next_queue, int* d_next_queue_size,
                               unsigned long long* d_edge_traversals) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int queue_size = *d_queue_size;

    if (tid < queue_size) {
        int u = d_queue[tid];
        unsigned int degree = d_graph_degrees[u];

        // Count edges traversed
        atomicAdd(d_edge_traversals, degree);

        // Process neighbors
        for (unsigned int i = 0; i < degree; i++) {
            unsigned long long v = d_graph_neighbors[u][i];
            if (!d_visited[v]) {
                d_visited[v] = true;
                int next_idx = atomicAdd(d_next_queue_size, 1);
                d_next_queue[next_idx] = v;
            }
        }
    }
}

// Single-source shortest path BFS
unsigned long long sssp_bfs_ef(EFLayout* ef_layout, unsigned int src) {
    CSRGraph* csr = ef_layout->csr;
    unsigned int num_vertices = csr->num_vertices;

    // CPU arrays
    bool* visited = (bool*)calloc(num_vertices, sizeof(bool));
    int* queue = (int*)malloc(num_vertices * sizeof(int));
    int* next_queue = (int*)malloc(num_vertices * sizeof(int));
    int queue_size = 1;
    int next_queue_size = 0;

    // Initialize
    visited[src] = true;
    queue[0] = src;

    unsigned long long edge_traversals = 0;

    // BFS
    while (queue_size > 0) {
        for (int i = 0; i < queue_size; i++) {
            int u = queue[i];
            edge_traversals += csr->graph_degrees[u];

            for (unsigned int j = 0; j < csr->graph_degrees[u]; j++) {
                unsigned long long v = csr->graph_neighbors[u][j];
                if (!visited[v]) {
                    visited[v] = true;
                    next_queue[next_queue_size++] = v;
                }
            }
        }

        // Swap queues
        int* temp = queue;
        queue = next_queue;
        next_queue = temp;
        queue_size = next_queue_size;
        next_queue_size = 0;
    }

    free(visited);
    free(queue);
    free(next_queue);

    return edge_traversals;
}

// Find connected vertices for sampling
void find_connected_vertices(CSRGraph* csr, unsigned int** connected_vertices, unsigned int* num_connected) {
    unsigned int count = 0;
    for (unsigned int i = 0; i < csr->num_vertices; i++) {
        if (csr->graph_degrees[i] > 0) {
            count++;
        }
    }

    *num_connected = count;
    *connected_vertices = (unsigned int*)malloc(count * sizeof(unsigned int));

    count = 0;
    for (unsigned int i = 0; i < csr->num_vertices; i++) {
        if (csr->graph_degrees[i] > 0) {
            (*connected_vertices)[count++] = i;
        }
    }
}

// Average GTEPS calculation
double average_gteps_ef(EFLayout* ef_layout, unsigned int trials) {
    CSRGraph* csr = ef_layout->csr;

    // Find connected vertices
    unsigned int* connected_vertices;
    unsigned int num_connected;
    find_connected_vertices(csr, &connected_vertices, &num_connected);

    if (num_connected < trials) {
        printf("Only %u connected vertices found, reducing trials.\n", num_connected);
        trials = num_connected;
    }

    // Sample random sources
    unsigned int* random_sources = (unsigned int*)malloc(trials * sizeof(unsigned int));

    // Simple random sampling without replacement
    if (trials == num_connected) {
        // Use all connected vertices
        for (unsigned int i = 0; i < trials; i++) {
            random_sources[i] = connected_vertices[i];
        }
    } else {
        // Fisher-Yates shuffle for sampling
        for (unsigned int i = 0; i < trials; i++) {
            unsigned int j = i + rand() % (num_connected - i);
            // Swap
            unsigned int temp = connected_vertices[i];
            connected_vertices[i] = connected_vertices[j];
            connected_vertices[j] = temp;
            random_sources[i] = connected_vertices[i];
        }
    }

    double total_gteps = 0.0;

    for (unsigned int i = 0; i < trials; i++) {
        unsigned int src = random_sources[i];

        clock_t start = clock();
        unsigned long long edges = sssp_bfs_ef(ef_layout, src);
        clock_t end = clock();

        double duration = ((double)(end - start)) / CLOCKS_PER_SEC;
        double gteps = edges / (duration * 1e9);
        total_gteps += gteps;
    }

    free(connected_vertices);
    free(random_sources);

    return total_gteps / trials;
}

// Free CSR Graph resources
void free_csr_graph(CSRGraph* graph) {
    for (unsigned int i = 0; i < graph->num_vertices; i++) {
        if (graph->graph_neighbors[i]) {
            free(graph->graph_neighbors[i]);
        }
    }

    free(graph->graph_neighbors);
    free(graph->graph_degrees);
    free(graph->vertex_csr);
    free(graph->edge_csr);
}

// Free EF Layout resources
void free_ef_layout(EFLayout* ef_layout) {
    free(ef_layout->chunk_off_exsum);
}

int main() {
    // Start overall runtime measurement
    clock_t total_start_time = clock();

    // Read CSR files
    unsigned long long* vertex_csr;
    unsigned long long* edge_csr;
    unsigned int num_vertices, num_edges;

    printf("Starting to read CSR files...\n");
    clock_t read_start = clock();
    read_binary_csr("vertex.CSR", "edge.CSR", &vertex_csr, &edge_csr, &num_vertices, &num_edges);
    clock_t read_end = clock();
    double read_time = ((double)(read_end - read_start)) / CLOCKS_PER_SEC;
    printf("CSR files read in %.3f seconds\n", read_time);

    // Create CSR Graph
    CSRGraph csr_graph = {
        .vertex_csr = vertex_csr,
        .edge_csr = edge_csr,
        .num_vertices = num_vertices,
        .num_edges = num_edges
    };

    // Build graph
    printf("Building graph...\n");
    clock_t build_start = clock();
    build_graph(&csr_graph);
    clock_t build_end = clock();
    double build_time = ((double)(build_end - build_start)) / CLOCKS_PER_SEC;
    printf("Graph built in %.3f seconds\n", build_time);

    // Create EF Layout
    EFLayout ef_layout = {
        .csr = &csr_graph,
        .num_vertices = num_vertices,
        .num_edges = num_edges
    };

    // Set EF parameters
    printf("Setting EF parameters...\n");
    clock_t ef_start = clock();
    set_ef_params(&ef_layout);
    clock_t ef_end = clock();
    double ef_time = ((double)(ef_end - ef_start)) / CLOCKS_PER_SEC;
    printf("EF parameters set in %.3f seconds\n", ef_time);

    // Print stats
    printf("\n--- Graph Statistics ---\n");
    printf("Vertices: %u, Edges: %u\n", csr_graph.num_vertices, csr_graph.num_edges);
    size_t optimal_csr_size = get_optimal_csr_size(&ef_layout);
    printf("Optimal CSR Size: %zu bytes\n", optimal_csr_size);
    printf("Compressed (Elias–Fano) Size: %zu bytes\n", ef_layout.storage_size);
    printf("Compression Ratio (CSR / EF): %.4f\n",
           (double)optimal_csr_size / ef_layout.storage_size);

    // Run GTEPS benchmark
    unsigned int trials = 100;
    printf("\nRunning SSSP benchmark with %u trials...\n", trials);
    clock_t bench_start = clock();
    double avg_gteps = average_gteps_ef(&ef_layout, trials);
    clock_t bench_end = clock();
    double bench_time = ((double)(bench_end - bench_start)) / CLOCKS_PER_SEC;

    printf("\n--- SSSP on EFG Benchmark Results ---\n");
    printf("Average GTEPS over %u connected runs: %.6f\n", trials, avg_gteps);
    printf("Benchmark completed in %.3f seconds\n", bench_time);

    // Clean up
    printf("\nCleaning up resources...\n");
    free_ef_layout(&ef_layout);
    free_csr_graph(&csr_graph);

    // Calculate and print total runtime
    clock_t total_end_time = clock();
    double total_runtime = ((double)(total_end_time - total_start_time)) / CLOCKS_PER_SEC;
    printf("\n--- Program Summary ---\n");
    printf("Total runtime: %.3f seconds\n", total_runtime);
    printf("File reading: %.3f seconds (%.1f%%)\n", read_time, (read_time/total_runtime)*100);
    printf("Graph building: %.3f seconds (%.1f%%)\n", build_time, (build_time/total_runtime)*100);
    printf("EF parameter setup: %.3f seconds (%.1f%%)\n", ef_time, (ef_time/total_runtime)*100);
    printf("SSSP benchmark: %.3f seconds (%.1f%%)\n", bench_time, (bench_time/total_runtime)*100);

    return 0;
}
