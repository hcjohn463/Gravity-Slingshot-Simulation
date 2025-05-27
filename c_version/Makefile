# Makefile for Slingshot Parameter Sweep (C Versions)

CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -std=c99
NVCCFLAGS = -O3 -arch=sm_50
LDFLAGS = -lm
OPENMP_FLAGS = -fopenmp

# Targets
TARGET_BASIC = parameter_sweep
TARGET_FINE = parameter_sweep_fine
TARGET_OPENMP = parameter_sweep_openmp
TARGET_CUDA = parameter_sweep_cuda

# Source files
SOURCE_BASIC = parameter_sweep.c
SOURCE_FINE = parameter_sweep_fine.c
SOURCE_OPENMP = parameter_sweep_openmp.c
SOURCE_CUDA = parameter_sweep_cuda.cu

# Default target - build all versions
all: $(TARGET_BASIC) $(TARGET_FINE) $(TARGET_OPENMP) $(TARGET_CUDA)

# Basic version (original)
$(TARGET_BASIC): $(SOURCE_BASIC)
	$(CC) $(CFLAGS) -o $(TARGET_BASIC) $(SOURCE_BASIC) $(LDFLAGS)

# Fine grid version
$(TARGET_FINE): $(SOURCE_FINE)
	$(CC) $(CFLAGS) -o $(TARGET_FINE) $(SOURCE_FINE) $(LDFLAGS)

# OpenMP parallel version
$(TARGET_OPENMP): $(SOURCE_OPENMP)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -o $(TARGET_OPENMP) $(SOURCE_OPENMP) $(LDFLAGS)

# CUDA GPU version
$(TARGET_CUDA): $(SOURCE_CUDA)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET_CUDA) $(SOURCE_CUDA)

# Run targets
run-basic: $(TARGET_BASIC)
	@echo "🚀 Running basic version (90K tests, step=10)..."
	./$(TARGET_BASIC)

run-fine: $(TARGET_FINE)
	@echo "🚀 Running fine grid version (9M tests, step=1)..."
	./$(TARGET_FINE)

run-openmp: $(TARGET_OPENMP)
	@echo "🚀 Running OpenMP parallel version (9M tests, step=1)..."
	./$(TARGET_OPENMP)

run-cuda: $(TARGET_CUDA)
	@echo "🚀 Running CUDA GPU version (9M tests, step=1)..."
	./$(TARGET_CUDA)

# Benchmark - run all versions and compare performance
benchmark: all
	@echo "📊 Performance Benchmark - Slingshot Parameter Sweep"
	@echo "===================================================="
	@echo ""
	@echo "1️⃣  Basic Version (90K tests):"
	@time ./$(TARGET_BASIC) > /dev/null
	@echo ""
	@echo "2️⃣  Fine Grid Serial (9M tests):"
	@time ./$(TARGET_FINE) > /dev/null
	@echo ""
	@echo "3️⃣  OpenMP Parallel (9M tests):"
	@time ./$(TARGET_OPENMP) > /dev/null
	@echo ""
	@echo "4️⃣  CUDA GPU (9M tests):"
	@time ./$(TARGET_CUDA) > /dev/null
	@echo ""
	@echo "✅ Benchmark complete! Check execution times above."

# Set number of OpenMP threads
set-threads:
	@echo "Current OMP_NUM_THREADS: $$OMP_NUM_THREADS"
	@echo "Available CPU cores: $$(nproc 2>/dev/null || echo 'unknown')"
	@echo "To set threads: export OMP_NUM_THREADS=<number>"

# Test CUDA installation
test-cuda:
	@echo "Testing CUDA support..."
	@which nvcc > /dev/null 2>&1 && echo "✅ NVCC compiler found" || echo "❌ NVCC not found"
	@nvidia-smi > /dev/null 2>&1 && echo "✅ NVIDIA GPU detected" || echo "❌ No NVIDIA GPU or driver"

# Clean compiled files and outputs
clean:
	rm -f $(TARGET_BASIC) $(TARGET_FINE) $(TARGET_OPENMP) $(TARGET_CUDA)
	rm -f successful_parameters.txt successful_parameters_fine.txt successful_parameters_openmp.txt successful_parameters_cuda.txt

# Clean and rebuild all
rebuild: clean all

# Test OpenMP installation
test-openmp:
	@echo "Testing OpenMP support..."
	@echo '#include <omp.h>' > /tmp/test_openmp.c && echo 'int main() { return 0; }' >> /tmp/test_openmp.c && $(CC) $(OPENMP_FLAGS) /tmp/test_openmp.c -o /tmp/test_openmp $(LDFLAGS) 2>/dev/null && echo "✅ OpenMP is available" || echo "❌ OpenMP not found"
	@rm -f /tmp/test_openmp /tmp/test_openmp.c

# Show help
help:
	@echo "🚀 Slingshot Parameter Sweep - Makefile Help"
	@echo "============================================="
	@echo ""
	@echo "Build targets:"
	@echo "  all           - Build all versions (basic, fine, openmp, cuda)"
	@echo "  parameter_sweep       - Build basic version (90K tests)"
	@echo "  parameter_sweep_fine  - Build fine grid version (9M tests)"
	@echo "  parameter_sweep_openmp - Build OpenMP version (9M tests)"
	@echo "  parameter_sweep_cuda   - Build CUDA GPU version (9M tests)"
	@echo ""
	@echo "Run targets:"
	@echo "  run-basic     - Run basic version"
	@echo "  run-fine      - Run fine grid version"
	@echo "  run-openmp    - Run OpenMP parallel version"
	@echo "  run-cuda      - Run CUDA GPU version"
	@echo ""
	@echo "Performance:"
	@echo "  benchmark     - Run all versions and compare performance"
	@echo "  set-threads   - Show current thread settings"
	@echo "  test-openmp   - Test if OpenMP is available"
	@echo "  test-cuda     - Test if CUDA is available"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean         - Remove compiled files and outputs"
	@echo "  rebuild       - Clean and rebuild all"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "💡 Tips:"
	@echo "  - Set threads: export OMP_NUM_THREADS=4"
	@echo "  - Fine grid version tests 9 million combinations!"
	@echo "  - OpenMP version should show significant speedup"
	@echo "  - CUDA version should be fastest (if GPU available)"

.PHONY: all run-basic run-fine run-openmp run-cuda benchmark set-threads clean rebuild test-openmp test-cuda help 