#!/bin/bash

echo "🚀 Slingshot Parameter Sweep - Performance Benchmark"
echo "===================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if OpenMP is available
echo "🔧 Checking system configuration..."
echo -n "OpenMP support: "
if gcc -fopenmp -xc /dev/null -o /dev/null 2>/dev/null <<< '#include <omp.h>'; then
    echo -e "${GREEN}✅ Available${NC}"
else
    echo -e "${RED}❌ Not available${NC}"
    echo "Please install OpenMP support for GCC"
    exit 1
fi

# Get CPU info
if command -v nproc &> /dev/null; then
    CPU_CORES=$(nproc)
    echo "CPU cores: ${CPU_CORES}"
else
    CPU_CORES=4
    echo "CPU cores: Unknown (assuming ${CPU_CORES})"
fi

echo ""

# Build all versions
echo "🔨 Building all versions..."
make clean > /dev/null 2>&1
if make all > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Build successful${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

echo ""

# Function to run timing test
run_test() {
    local name="$1"
    local command="$2"
    local description="$3"
    
    echo -e "${BLUE}$name${NC} - $description"
    echo "Command: $command"
    
    # Run 3 times and take average
    local total_time=0
    local iterations=3
    
    for i in $(seq 1 $iterations); do
        echo -n "  Run $i/3: "
        # Use time command and capture only real time
        local time_output=$(bash -c "time $command > /dev/null 2>&1" 2>&1 | grep real | awk '{print $2}')
        
        # Convert time to seconds (handle mm:ss.sss format)
        local seconds=$(echo $time_output | awk -F: '{
            if (NF == 2) {
                print $1 * 60 + $2
            } else {
                print $1
            }
        }')
        
        total_time=$(echo "$total_time + $seconds" | bc -l 2>/dev/null || echo "$total_time")
        echo "${seconds}s"
    done
    
    local avg_time=$(echo "scale=2; $total_time / $iterations" | bc -l 2>/dev/null || echo "N/A")
    echo -e "  ${GREEN}Average: ${avg_time}s${NC}"
    echo ""
    
    echo "$avg_time"
}

# Store results
declare -A results

echo "📊 Running performance tests..."
echo "==============================="
echo ""

# Test 1: Basic version (90K tests)
results["basic"]=$(run_test "1️⃣  Basic Version" "./parameter_sweep" "90K tests, step=10")

# Test 2: Fine grid serial (9M tests)
results["fine"]=$(run_test "2️⃣  Fine Grid (Serial)" "./parameter_sweep_fine" "9M tests, step=1, single thread")

# Test 3: OpenMP with different thread counts
echo "3️⃣  OpenMP Parallel Tests (9M tests, step=1)"
echo "============================================"

# Test with 1, 2, 4, 8 threads (up to CPU_CORES)
thread_counts=(1 2 4 8)
for threads in "${thread_counts[@]}"; do
    if [ $threads -le $CPU_CORES ]; then
        export OMP_NUM_THREADS=$threads
        results["openmp_$threads"]=$(run_test "   ${threads} threads" "./parameter_sweep_openmp" "OpenMP with $threads threads")
    fi
done

# Calculate speedups
echo ""
echo "📈 Performance Analysis"
echo "======================"
echo ""

# Get baseline (fine grid serial)
baseline=${results["fine"]}
echo "Baseline (Fine Grid Serial): ${baseline}s"
echo ""

if [ "$baseline" != "N/A" ] && [ "$baseline" != "0" ]; then
    echo "Speedup Analysis:"
    echo "-----------------"
    
    for threads in "${thread_counts[@]}"; do
        if [ $threads -le $CPU_CORES ] && [ -n "${results["openmp_$threads"]}" ]; then
            openmp_time=${results["openmp_$threads"]}
            if [ "$openmp_time" != "N/A" ] && [ "$openmp_time" != "0" ]; then
                speedup=$(echo "scale=2; $baseline / $openmp_time" | bc -l 2>/dev/null || echo "N/A")
                efficiency=$(echo "scale=1; $speedup / $threads * 100" | bc -l 2>/dev/null || echo "N/A")
                
                if [ "$speedup" != "N/A" ]; then
                    echo "  ${threads} threads: ${speedup}x speedup (${efficiency}% efficiency)"
                fi
            fi
        fi
    done
fi

echo ""
echo "📋 Summary Table"
echo "================"
printf "%-20s %-12s %-15s %-10s\n" "Version" "Time (s)" "Tests" "Rate (tests/s)"
echo "------------------------------------------------------------"

# Basic version
if [ -n "${results["basic"]}" ] && [ "${results["basic"]}" != "N/A" ]; then
    rate=$(echo "scale=0; 90000 / ${results["basic"]}" | bc -l 2>/dev/null || echo "N/A")
    printf "%-20s %-12s %-15s %-10s\n" "Basic" "${results["basic"]}" "90K" "$rate"
fi

# Fine grid serial
if [ -n "${results["fine"]}" ] && [ "${results["fine"]}" != "N/A" ]; then
    rate=$(echo "scale=0; 9000000 / ${results["fine"]}" | bc -l 2>/dev/null || echo "N/A")
    printf "%-20s %-12s %-15s %-10s\n" "Fine (Serial)" "${results["fine"]}" "9M" "$rate"
fi

# OpenMP versions
for threads in "${thread_counts[@]}"; do
    if [ $threads -le $CPU_CORES ] && [ -n "${results["openmp_$threads"]}" ]; then
        openmp_time=${results["openmp_$threads"]}
        if [ "$openmp_time" != "N/A" ]; then
            rate=$(echo "scale=0; 9000000 / $openmp_time" | bc -l 2>/dev/null || echo "N/A")
            printf "%-20s %-12s %-15s %-10s\n" "OpenMP ($threads threads)" "$openmp_time" "9M" "$rate"
        fi
    fi
done

echo ""
echo "💡 Recommendations:"
echo "==================="

if [ -n "${results["openmp_4"]}" ] && [ -n "${results["fine"]}" ]; then
    if [ "${results["openmp_4"]}" != "N/A" ] && [ "${results["fine"]}" != "N/A" ]; then
        speedup_4=$(echo "scale=1; ${results["fine"]} / ${results["openmp_4"]}" | bc -l 2>/dev/null || echo "0")
        if (( $(echo "$speedup_4 > 2" | bc -l) )); then
            echo "✅ OpenMP shows good scalability - recommended for large parameter sweeps"
        else
            echo "⚠️  OpenMP speedup is limited - check for overhead or memory bottlenecks"
        fi
    fi
fi

echo "• Use basic version for quick tests (90K combinations)"
echo "• Use fine version for detailed single-threaded analysis (9M combinations)"
echo "• Use OpenMP version for maximum performance on multi-core systems"
echo ""
echo "🎯 Benchmark complete!"

# Reset environment
unset OMP_NUM_THREADS 