# Slingshot Parameter Sweep - 技術規格文件

## 🏗️ 系統架構

### 文件結構
```
c_version/
├── parameter_sweep.c              # 基礎版本 (90K測試)
├── parameter_sweep_fine.c         # 精細網格版本 (9M測試)
├── parameter_sweep_openmp.c       # OpenMP並行版本 (9M測試)
├── parameter_sweep_cuda.cu        # CUDA GPU版本 (9M測試)
├── Makefile                       # 編譯和測試腳本
├── performance_analysis.md        # 性能分析報告
├── technical_specifications.md    # 本技術規格文件
└── successful_parameters_*.txt    # 結果輸出文件
```

## 📊 測試參數規格

### 物理參數設定
```c
// 天體參數
#define EARTH_MASS 5.972e24          // 地球質量 (kg)
#define MOON_MASS (7.342e22 * 50)    // 月球質量增強50倍
#define MOON_RADIUS 1.737e6          // 月球半徑 (m)
#define G 6.67430e-11                // 重力常數

// 初始位置 (km)
#define SPACECRAFT_X 3e7             // 太空船X位置
#define SPACECRAFT_Y -2e7            // 太空船Y位置
#define MOON_X 0                     // 月球X位置
#define MOON_Y 0                     // 月球Y位置
#define TARGET_X 4e7                 // 目標X位置
#define TARGET_Y 3e7                 // 目標Y位置
#define TARGET_RADIUS 1e5            // 目標半徑 (100km)

// 模擬參數
#define TIME_STEP 200                // 時間步長 (秒)
#define MAX_STEPS 100               // 最大模擬步數
```

### 測試網格規格

#### 基礎版本 (90K測試)
- **X速度範圍**: -4000 到 -1000 m/s (步長: 10 m/s)
- **Y速度範圍**: 1000 到 4000 m/s (步長: 10 m/s)
- **總組合數**: 301 × 301 = 90,601

#### 精細網格版本 (9M測試)
- **X速度範圍**: -4000 到 -1000 m/s (步長: 1 m/s)
- **Y速度範圍**: 1000 到 4000 m/s (步長: 1 m/s)
- **總組合數**: 3001 × 3001 = 9,006,001

## 💻 編譯規格

### 編譯器設定
```makefile
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -std=c99
NVCCFLAGS = -O3 -arch=sm_50
LDFLAGS = -lm
OPENMP_FLAGS = -fopenmp
```

### 編譯命令
```bash
# 基礎版本
gcc -Wall -Wextra -O3 -std=c99 -o parameter_sweep parameter_sweep.c -lm

# OpenMP版本
gcc -Wall -Wextra -O3 -std=c99 -fopenmp -o parameter_sweep_openmp parameter_sweep_openmp.c -lm

# CUDA版本
nvcc -O3 -arch=sm_50 -o parameter_sweep_cuda parameter_sweep_cuda.cu
```

## 🔧 實現細節

### 核心算法流程
1. **初始化**: 設定物理參數和測試範圍
2. **參數掃描**: 遍歷所有速度組合
3. **軌道計算**: 數值積分求解運動方程
4. **碰撞檢測**: 檢查月球碰撞和目標命中
5. **結果收集**: 記錄成功參數
6. **輸出報告**: 生成統計信息和結果文件

### 物理計算核心
```c
// 重力計算
double dx = moon_x - spacecraft_x;
double dy = moon_y - spacecraft_y;
double r_sq = dx*dx + dy*dy;
double r = sqrt(r_sq);
double force = G * MOON_MASS / r_sq;

// 加速度計算
double ax = force * dx / r;
double ay = force * dy / r;

// 位置更新 (Verlet積分)
spacecraft_x += spacecraft_vx * TIME_STEP + 0.5 * ax * TIME_STEP * TIME_STEP;
spacecraft_y += spacecraft_vy * TIME_STEP + 0.5 * ay * TIME_STEP * TIME_STEP;
```

## 🚀 並行化實現

### OpenMP並行化策略
```c
#pragma omp parallel for private(vx, vy) schedule(dynamic)
for (int i = 0; i < x_steps; i++) {
    for (int j = 0; j < y_steps; j++) {
        // 每個線程處理不同的參數組合
        vx = vx_min + i * step;
        vy = vy_min + j * step;
        
        // 進行軌道模擬...
        if (simulate_trajectory(vx, vy)) {
            #pragma omp critical
            {
                // 線程安全的結果記錄
                save_successful_parameter(vx, vy);
            }
        }
    }
}
```

### CUDA並行化策略
```cuda
__global__ void parameter_sweep_kernel(
    int vx_min, int vy_min, int step,
    int x_steps, int y_steps,
    int* success_count, 
    ParameterResult* results
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tests = x_steps * y_steps;
    
    if (idx < total_tests) {
        int i = idx / y_steps;
        int j = idx % y_steps;
        
        int vx = vx_min + i * step;
        int vy = vy_min + j * step;
        
        if (simulate_trajectory_gpu(vx, vy)) {
            int pos = atomicAdd(success_count, 1);
            results[pos].vx = vx;
            results[pos].vy = vy;
        }
    }
}
```

## 📊 性能優化技術

### CUDA優化策略
1. **記憶體管理**: 使用統一記憶體 (Unified Memory)
2. **執行配置**: 35,180個block × 256個thread
3. **原子操作**: atomicAdd確保結果一致性
4. **記憶體合併**: 優化記憶體存取模式

### OpenMP優化策略
1. **動態調度**: `schedule(dynamic)` 平衡負載
2. **私有變數**: `private(vx, vy)` 避免競爭
3. **臨界區**: `#pragma omp critical` 保護共享資源
4. **線程數調節**: 支援 `OMP_NUM_THREADS` 環境變數

## 🔍 測試和驗證

### 精度驗證
- 所有版本使用相同的物理模型
- 數值結果一致性檢查
- 浮點運算精度控制

### 性能測試
- 使用 `time` 命令測量執行時間
- 多次運行取平均值
- 記錄詳細的性能指標

### 結果驗證
- 成功參數數量一致性
- 物理合理性檢查
- 邊界條件測試

## 🛠️ 開發工具和環境

### 必需軟體
- **GCC 13.3+**: C編譯器
- **NVIDIA CUDA Toolkit 12.0+**: GPU編程
- **OpenMP**: 多線程支援
- **Make**: 建置工具

### 推薦環境
- **WSL2**: Windows下的Linux環境
- **NVIDIA GPU**: 支援CUDA的顯卡
- **多核CPU**: 4核心以上

## 📋 使用說明

### 快速開始
```bash
# 編譯所有版本
make all

# 測試環境
make test-openmp
make test-cuda

# 運行基準測試
make benchmark

# 查看幫助
make help
```

### 環境設定
```bash
# 設定OpenMP線程數
export OMP_NUM_THREADS=4

# 檢查GPU狀態
nvidia-smi

# 檢查CUDA版本
nvcc --version
```

## 🎯 擴展建議

### 功能擴展
1. **更多物理效果**: 地球重力、太陽風
2. **3D模擬**: 增加Z軸計算
3. **變步長積分**: 自適應時間步長
4. **GUI界面**: 視覺化軌道顯示

### 性能擴展
1. **多GPU支援**: 跨GPU並行計算
2. **MPI分散式**: 集群計算支援
3. **記憶體優化**: 減少記憶體使用
4. **向量化**: SIMD指令優化

---
*技術規格版本: 1.0*
*更新日期: 2025年5月27日* 