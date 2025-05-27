#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>  // OpenMP header

// Physical constants
#define G 6.67430e-11
#define DT 200.0
#define MOON_MASS (7.342e22 * 50)
#define MOON_RADIUS 1737000.0
#define SPACECRAFT_MASS 1000.0
#define MAX_STEPS 100  // Back to original

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

// Result structure
typedef struct {
    int target_hit;
    int target_hit_time;
    Vector2D final_pos;
} SimResult;

// Thread-safe result collection structure
typedef struct {
    int vx, vy;
    int hit_time;
    double speed;
} SuccessResult;

// Vector operations
Vector2D vector_add(Vector2D a, Vector2D b) {
    Vector2D result = {a.x + b.x, a.y + b.y};
    return result;
}

Vector2D vector_subtract(Vector2D a, Vector2D b) {
    Vector2D result = {a.x - b.x, a.y - b.y};
    return result;
}

Vector2D vector_multiply(Vector2D v, double scalar) {
    Vector2D result = {v.x * scalar, v.y * scalar};
    return result;
}

double vector_magnitude(Vector2D v) {
    return sqrt(v.x * v.x + v.y * v.y);
}

Vector2D vector_normalize(Vector2D v) {
    double mag = vector_magnitude(v);
    if (mag > 0) {
        Vector2D result = {v.x / mag, v.y / mag};
        return result;
    }
    Vector2D zero = {0, 0};
    return zero;
}

// Check if spacecraft hits target
int check_target_hit(Vector2D spacecraft_pos, SimParams params) {
    Vector2D diff = vector_subtract(spacecraft_pos, params.target_center);
    double distance = vector_magnitude(diff);
    return distance <= params.target_radius;
}

// Calculate gravitational force
Vector2D gravitational_force(Vector2D pos1, Vector2D pos2, double mass1, double mass2) {
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

// Run single simulation
SimResult run_simulation(Vector2D spacecraft_vel, SimParams params) {
    // Initialize positions and velocities
    Vector2D moon_pos = {0, 0};
    Vector2D moon_vel = {1500, 0};  // 1.5 km/s rightward
    Vector2D spacecraft_pos = params.spacecraft_pos;
    
    SimResult result = {0, -1, {0, 0}};
    
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
            result.target_hit = 1;
            result.target_hit_time = step;
            result.final_pos = spacecraft_pos;
            break;
        }
    }
    
    result.final_pos = spacecraft_pos;
    return result;
}

int main() {
    printf("🚀 Slingshot Parameter Sweep (OpenMP Parallel Version)\n");
    printf("======================================================\n");
    
    // Get number of threads
    int num_threads = omp_get_max_threads();
    printf("OpenMP available threads: %d\n", num_threads);
    
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
    int vx_min = -4000, vx_max = -1000, vx_step = 1;
    int vy_min = 1000, vy_max = 4000, vy_step = 1;
    
    long long total_tests = ((long long)(vx_max - vx_min) / vx_step + 1) * 
                           ((long long)(vy_max - vy_min) / vy_step + 1);
    
    printf("Total parameter combinations: %lld\n", total_tests);
    printf("Using %d threads for parallel computation\n", num_threads);
    printf("Progress: ");
    fflush(stdout);
    
    // Allocate array to store successful results
    const int MAX_RESULTS = 100000;  // Generous estimate
    SuccessResult* successful_results = malloc(MAX_RESULTS * sizeof(SuccessResult));
    if (!successful_results) {
        printf("Error: Cannot allocate memory for results!\n");
        return 1;
    }
    
    int successful_count = 0;
    long long test_count = 0;
    long long progress_step = total_tests / 100;
    
    double start_time = omp_get_wtime();
    
    // OpenMP parallel parameter sweep
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int local_successful = 0;
        SuccessResult local_results[1000];  // Local buffer per thread
        
        #pragma omp for schedule(dynamic, 100) nowait
        for (int vx = vx_min; vx <= vx_max; vx += vx_step) {
            for (int vy = vy_min; vy <= vy_max; vy += vy_step) {
                long long current_test;
                
                #pragma omp atomic capture
                current_test = ++test_count;
                
                // Run simulation
                Vector2D spacecraft_vel = {(double)vx, (double)vy};
                SimResult result = run_simulation(spacecraft_vel, params);
                
                // Check if successful
                if (result.target_hit) {
                    if (local_successful < 1000) {  // Prevent buffer overflow
                        local_results[local_successful].vx = vx;
                        local_results[local_successful].vy = vy;
                        local_results[local_successful].hit_time = result.target_hit_time;
                        local_results[local_successful].speed = sqrt(vx * vx + vy * vy);
                        local_successful++;
                    }
                }
                
                // Progress reporting (thread 0 only)
                if (thread_id == 0 && current_test % progress_step == 0) {
                    printf(".");
                    fflush(stdout);
                }
            }
        }
        
        // Combine results from all threads (critical section)
        #pragma omp critical
        {
            if (successful_count + local_successful <= MAX_RESULTS) {
                for (int i = 0; i < local_successful; i++) {
                    successful_results[successful_count + i] = local_results[i];
                }
                successful_count += local_successful;
            }
        }
    }
    
    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;
    
    printf("\n\n✅ Parallel parameter sweep completed!\n");
    printf("Total tests: %lld\n", total_tests);
    printf("Successful parameters: %d\n", successful_count);
    printf("Success rate: %.4f%%\n", (double)successful_count / total_tests * 100);
    printf("Execution time: %.2f seconds\n", execution_time);
    printf("Performance: %.0f tests/second\n", total_tests / execution_time);
    printf("Threads used: %d\n", num_threads);
    
    // Write results to file
    printf("💾 Writing results to file...\n");
    FILE *output_file = fopen("successful_parameters_openmp.txt", "w");
    if (output_file) {
        fprintf(output_file, "# Successful Slingshot Parameters (OpenMP Parallel Version)\n");
        fprintf(output_file, "# Format: Vx(m/s), Vy(m/s), Speed(m/s), HitTime(steps)\n");
        fprintf(output_file, "# Target: [4e7, 3e7] km, Radius: 100,000 km\n");
        fprintf(output_file, "# Spacecraft initial position: [3e7, -2e7] km\n");
        fprintf(output_file, "# Grid resolution: 1 m/s step size\n");
        fprintf(output_file, "# Computed using %d OpenMP threads\n", num_threads);
        fprintf(output_file, "# Execution time: %.2f seconds\n\n", execution_time);
        
        for (int i = 0; i < successful_count; i++) {
            fprintf(output_file, "%6d, %6d, %7.0f, %3d\n",
                   successful_results[i].vx, successful_results[i].vy,
                   successful_results[i].speed, successful_results[i].hit_time);
        }
        fclose(output_file);
        printf("💾 Results saved to 'successful_parameters_openmp.txt'\n");
    }
    
    // Show first few successful parameters if any
    if (successful_count > 0) {
        printf("\n📊 First few successful parameters:\n");
        printf("Vx (m/s) | Vy (m/s) | Speed (m/s) | Hit Time\n");
        printf("---------|----------|-------------|----------\n");
        
        int show_count = (successful_count < 5) ? successful_count : 5;
        for (int i = 0; i < show_count; i++) {
            printf("%8d | %8d | %11.0f | %8d\n",
                   successful_results[i].vx, successful_results[i].vy,
                   successful_results[i].speed, successful_results[i].hit_time);
        }
        
        if (successful_count > 5) {
            printf("... and %d more\n", successful_count - 5);
        }
    }
    
    free(successful_results);
    
    printf("\n🎯 Parallel analysis complete!\n");
    printf("Speedup potential vs serial: ~%dx (with %d threads)\n", num_threads, num_threads);
    return 0;
} 