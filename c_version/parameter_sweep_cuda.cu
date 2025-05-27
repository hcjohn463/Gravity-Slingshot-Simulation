#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Physical constants
#define G 6.67430e-11
#define DT 200.0
#define MOON_MASS (7.342e22 * 50)
#define MOON_RADIUS 1737000.0
#define SPACECRAFT_MASS 1000.0
#define MAX_STEPS 100

// CUDA kernel parameters
#define BLOCK_SIZE 256
#define MAX_RESULTS_PER_BLOCK 1000

// Position and velocity structures
typedef struct {
    double x, y;
} Vector2D;

// Simulation parameters
typedef struct {
    Vector2D spacecraft_pos;
    Vector2D target_center;
    double target_radius;
} SimParams;

// Result structure for successful parameters
typedef struct {
    int vx, vy;
    int hit_time;
    double speed;
} SuccessResult;

// Device functions
__device__ Vector2D vector_add(Vector2D a, Vector2D b) {
    Vector2D result = {a.x + b.x, a.y + b.y};
    return result;
}

__device__ Vector2D vector_subtract(Vector2D a, Vector2D b) {
    Vector2D result = {a.x - b.x, a.y - b.y};
    return result;
}

__device__ Vector2D vector_multiply(Vector2D v, double scalar) {
    Vector2D result = {v.x * scalar, v.y * scalar};
    return result;
}

__device__ double vector_magnitude(Vector2D v) {
    return sqrt(v.x * v.x + v.y * v.y);
}

__device__ Vector2D vector_normalize(Vector2D v) {
    double mag = vector_magnitude(v);
    if (mag > 0) {
        Vector2D result = {v.x / mag, v.y / mag};
        return result;
    }
    Vector2D zero = {0, 0};
    return zero;
}

__device__ int check_target_hit(Vector2D spacecraft_pos, SimParams params) {
    Vector2D diff = vector_subtract(spacecraft_pos, params.target_center);
    double distance = vector_magnitude(diff);
    return distance <= params.target_radius;
}

__device__ Vector2D gravitational_force(Vector2D pos1, Vector2D pos2, double mass1, double mass2) {
    Vector2D r_vector = vector_subtract(pos2, pos1);
    double r_magnitude = vector_magnitude(r_vector);
    
    // Prevent collision
    double min_distance = MOON_RADIUS * 1.2;
    if (r_magnitude < min_distance) {
        r_magnitude = min_distance;
    }
    
    double force_magnitude = G * mass1 * mass2 / (r_magnitude * r_magnitude);
    Vector2D force_direction = vector_normalize(r_vector);
    
    return vector_multiply(force_direction, force_magnitude);
}

__device__ int run_simulation(int vx, int vy, SimParams params) {
    // Initialize positions and velocities
    Vector2D moon_pos = {0, 0};
    Vector2D moon_vel = {1500, 0};  // 1.5 km/s rightward
    Vector2D spacecraft_pos = params.spacecraft_pos;
    Vector2D spacecraft_vel = {(double)vx, (double)vy};
    
    // Run simulation
    for (int step = 0; step < MAX_STEPS; step++) {
        // Calculate gravitational force
        Vector2D force = gravitational_force(spacecraft_pos, moon_pos, 
                                           SPACECRAFT_MASS, MOON_MASS);
        
        // Update spacecraft
        Vector2D acceleration = vector_multiply(force, 1.0 / SPACECRAFT_MASS);
        Vector2D vel_change = vector_multiply(acceleration, DT);
        spacecraft_vel = vector_add(spacecraft_vel, vel_change);
        
        Vector2D pos_change = vector_multiply(spacecraft_vel, DT);
        spacecraft_pos = vector_add(spacecraft_pos, pos_change);
        
        // Update moon (constant velocity)
        Vector2D moon_pos_change = vector_multiply(moon_vel, DT);
        moon_pos = vector_add(moon_pos, moon_pos_change);
        
        // Check target hit
        if (check_target_hit(spacecraft_pos, params)) {
            return step;  // Return hit time
        }
    }
    
    return -1;  // No hit
}

__global__ void parameter_sweep_kernel(SimParams params, int vx_min, int vx_max, 
                                      int vy_min, int vy_max, 
                                      SuccessResult* results, int* result_count) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_vx = vx_max - vx_min + 1;
    int total_vy = vy_max - vy_min + 1;
    long long total_tests = (long long)total_vx * total_vy;
    
    if (idx >= total_tests) return;
    
    // Convert linear index to vx, vy coordinates
    int vx_idx = idx / total_vy;
    int vy_idx = idx % total_vy;
    int vx = vx_min + vx_idx;
    int vy = vy_min + vy_idx;
    
    // Run simulation
    int hit_time = run_simulation(vx, vy, params);
    
    // If successful, add to results
    if (hit_time >= 0) {
        int result_idx = atomicAdd(result_count, 1);
        if (result_idx < 100000) {  // Prevent overflow
            results[result_idx].vx = vx;
            results[result_idx].vy = vy;
            results[result_idx].hit_time = hit_time;
            results[result_idx].speed = sqrt((double)vx * vx + (double)vy * vy);
        }
    }
}

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

int main() {
    printf("🚀 Slingshot Parameter Sweep (CUDA GPU Version)\n");
    printf("===============================================\n");
    
    // Check CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("❌ No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    
    // Initialize simulation parameters
    SimParams params;
    params.spacecraft_pos.x = 3e7;   // 30M km right
    params.spacecraft_pos.y = -2e7;  // 20M km down
    params.target_center.x = 4e7;    // 40M km right
    params.target_center.y = 3e7;    // 30M km up
    params.target_radius = 1e5;      // 100k km radius
    
    printf("Testing velocity ranges (FINE GRID):\n");
    printf("X: -4000 to -1000 m/s (step: 1)\n");
    printf("Y: 1000 to 4000 m/s (step: 1)\n");
    printf("Target: [%.0f, %.0f] km, Radius: %.0f km\n\n", 
           params.target_center.x/1000, params.target_center.y/1000, 
           params.target_radius/1000);
    
    // Parameter ranges
    int vx_min = -4000, vx_max = -1000;
    int vy_min = 1000, vy_max = 4000;
    
    long long total_tests = ((long long)(vx_max - vx_min + 1)) * 
                           ((long long)(vy_max - vy_min + 1));
    
    printf("Total parameter combinations: %lld\n", total_tests);
    printf("Using GPU parallel computation\n");
    
    // Allocate host memory for results
    const int MAX_RESULTS = 100000;
    SuccessResult* h_results = (SuccessResult*)malloc(MAX_RESULTS * sizeof(SuccessResult));
    int h_result_count = 0;
    
    // Allocate device memory
    SuccessResult* d_results;
    int* d_result_count;
    
    CUDA_CHECK(cudaMalloc(&d_results, MAX_RESULTS * sizeof(SuccessResult)));
    CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_result_count, &h_result_count, sizeof(int), cudaMemcpyHostToDevice));
    
    // Calculate grid and block dimensions
    int threads_per_block = BLOCK_SIZE;
    int blocks = (total_tests + threads_per_block - 1) / threads_per_block;
    
    printf("Launching %d blocks with %d threads each\n", blocks, threads_per_block);
    printf("Progress: Computing...\n");
    
    // Record start time
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    // Launch kernel
    parameter_sweep_kernel<<<blocks, threads_per_block>>>(
        params, vx_min, vx_max, vy_min, vy_max, d_results, d_result_count);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Record end time
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float execution_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&execution_time_ms, start, stop));
    double execution_time = execution_time_ms / 1000.0;
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(&h_result_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_results, d_results, h_result_count * sizeof(SuccessResult), cudaMemcpyDeviceToHost));
    
    printf("\n✅ CUDA parameter sweep completed!\n");
    printf("Total tests: %lld\n", total_tests);
    printf("Successful parameters: %d\n", h_result_count);
    printf("Success rate: %.4f%%\n", (double)h_result_count / total_tests * 100);
    printf("Execution time: %.3f seconds\n", execution_time);
    printf("Performance: %.0f tests/second\n", total_tests / execution_time);
    
    // Write results to file
    printf("💾 Writing results to file...\n");
    FILE *output_file = fopen("successful_parameters_cuda.txt", "w");
    if (output_file) {
        fprintf(output_file, "# Successful Slingshot Parameters (CUDA GPU Version)\n");
        fprintf(output_file, "# Format: Vx(m/s), Vy(m/s), Speed(m/s), HitTime(steps)\n");
        fprintf(output_file, "# Target: [4e7, 3e7] km, Radius: 100,000 km\n");
        fprintf(output_file, "# Spacecraft initial position: [3e7, -2e7] km\n");
        fprintf(output_file, "# Grid resolution: 1 m/s step size\n");
        fprintf(output_file, "# GPU: %s\n", prop.name);
        fprintf(output_file, "# Execution time: %.3f seconds\n\n", execution_time);
        
        for (int i = 0; i < h_result_count; i++) {
            fprintf(output_file, "%6d, %6d, %7.0f, %3d\n",
                   h_results[i].vx, h_results[i].vy,
                   h_results[i].speed, h_results[i].hit_time);
        }
        fclose(output_file);
        printf("💾 Results saved to 'successful_parameters_cuda.txt'\n");
    }
    
    // Show first few successful parameters if any
    if (h_result_count > 0) {
        printf("\n📊 First few successful parameters:\n");
        printf("Vx (m/s) | Vy (m/s) | Speed (m/s) | Hit Time\n");
        printf("---------|----------|-------------|----------\n");
        
        int show_count = (h_result_count < 5) ? h_result_count : 5;
        for (int i = 0; i < show_count; i++) {
            printf("%8d | %8d | %11.0f | %8d\n",
                   h_results[i].vx, h_results[i].vy,
                   h_results[i].speed, h_results[i].hit_time);
        }
        
        if (h_result_count > 5) {
            printf("... and %d more\n", h_result_count - 5);
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_result_count));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_results);
    
    printf("\n🎯 CUDA analysis complete!\n");
    printf("GPU acceleration achieved!\n");
    
    return 0;
} 