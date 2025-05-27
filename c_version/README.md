# 🚀 Slingshot Parameter Sweep - C版本並行計算

## 📋 項目簡介

這是一個高性能的重力助推軌道參數掃描程序，實現了四種不同的並行計算版本：
- **基礎版本** (90K測試) - 快速驗證
- **精細單線程版本** (9M測試) - 基準性能  
- **OpenMP並行版本** (9M測試) - CPU多線程
- **CUDA GPU版本** (9M測試) - GPU大規模並行

## 🏆 性能成果

### ⚡ 驚人的加速比
- **CUDA GPU**: **16.7倍加速** (16.82秒 → 1.01秒)
- **OpenMP**: **3.7倍加速** (16.82秒 → 4.50秒)
- **GPU吞吐量**: **890萬測試/秒**

### 🎯 計算精度
- 發現 **1,943-1,944個** 成功軌道參數
- 成功率: **0.0216%**
- 所有版本結果高度一致

## 🛠️ 快速開始

### 環境需求
- **操作系統**: Linux/WSL2
- **編譯器**: GCC 13.3+, NVIDIA CUDA 12.0+
- **硬體**: 多核CPU, NVIDIA GPU (推薦)

### 安裝與編譯
```bash
# 檢查環境
make test-openmp
make test-cuda

# 編譯所有版本
make all

# 運行性能測試
make benchmark
```

### 基本使用
```bash
# 運行基礎版本 (快速)
make run-basic

# 運行OpenMP版本 (多線程)
export OMP_NUM_THREADS=4
make run-openmp

# 運行CUDA版本 (最快)
make run-cuda
```

## 📊 性能基準測試結果

| 版本 | 測試數量 | 執行時間 | 性能 (測試/秒) | 成功參數 | 加速比 |
|------|----------|----------|----------------|----------|--------|
| 基礎版本 | 90,601 | 0.18秒 | 503,339 | 19 | N/A |
| 精細單線程 | 9,006,001 | 16.82秒 | 535,344 | 1,944 | 1.0x |
| OpenMP (4線程) | 9,006,001 | 4.50秒 | 2,000,361 | 1,944 | **3.7x** |
| CUDA GPU | 9,006,001 | 1.01秒 | 8,912,355 | 1,943 | **🚀 16.7x** |

## 🗂️ 文件結構

```
c_version/
├── 📁 源代碼
│   ├── parameter_sweep.c              # 基礎版本
│   ├── parameter_sweep_fine.c         # 精細網格版本
│   ├── parameter_sweep_openmp.c       # OpenMP並行版本
│   └── parameter_sweep_cuda.cu        # CUDA GPU版本
├── 🔧 建置工具
│   └── Makefile                       # 完整的編譯和測試腳本
├── 📊 分析文檔
│   ├── performance_analysis.md        # 性能分析報告
│   ├── technical_specifications.md    # 技術規格文件
│   ├── results_analysis.md           # 結果分析報告
│   └── README.md                      # 本文件
└── 📄 結果輸出
    ├── successful_parameters.txt      # 基礎版本結果
    ├── successful_parameters_fine.txt # 精細版本結果
    ├── successful_parameters_openmp.txt # OpenMP結果
    └── successful_parameters_cuda.txt  # CUDA結果
```

## 🧪 測試環境規格

### 硬體配置
- **CPU**: 4核心處理器
- **GPU**: NVIDIA GeForce RTX 3060 Ti
  - Compute Capability: 8.6
  - Multiprocessors: 38
  - Max threads per block: 1024
- **記憶體**: 16GB+ 推薦

### 軟體環境
- **OS**: Windows 11 + WSL2 (Ubuntu 24.04)
- **編譯器**: GCC 13.3.0, NVIDIA CUDA 12.0
- **函式庫**: OpenMP, CUDA Runtime

## 🚀 物理模型

### 模擬設定
- **太空船初始位置**: [30,000, -20,000] km
- **月球位置**: [0, 0] km (質量增強50倍)
- **目標位置**: [40,000, 30,000] km
- **目標半徑**: 100 km
- **時間步長**: 200秒
- **最大步數**: 100步

### 參數範圍
- **X速度**: -4000 到 -1000 m/s
- **Y速度**: 1000 到 4000 m/s
- **精細網格**: 1 m/s步長 (9M組合)
- **粗網格**: 10 m/s步長 (90K組合)

## 📈 成功參數特徵

### 發現的軌道特性
- **成功軌道數**: 約1944個
- **速度範圍**: 4000-5500 m/s
- **飛行時間**: 3-5.4小時
- **成功率**: 0.0216% (極高精確度要求)

### 物理洞察
- **X速度**: 全為負值 (向左飛行接近月球)
- **Y速度**: 全為正值 (向上飛行獲得助推)
- **敏感性**: 1 m/s差異可決定成敗

## 🎯 應用價值

### 工程應用
1. **軌道設計**: 實際太空任務參數參考
2. **風險評估**: 量化軌道設計敏感性
3. **任務規劃**: 最佳發射窗口選擇

### 教育價值
1. **並行計算教學**: 完整的性能階梯展示
2. **物理模擬**: 重力助推原理驗證
3. **高性能計算**: GPU計算實踐案例

## 🛠️ 技術特色

### 並行化策略
- **OpenMP**: 動態調度，私有變數，臨界區保護
- **CUDA**: 大規模線程，統一記憶體，原子操作
- **負載平衡**: 自動任務分配，最大化硬體利用

### 優化技術
- **編譯器優化**: -O3最高等級優化
- **記憶體管理**: 高效的GPU記憶體使用
- **數值穩定**: Verlet積分法確保精度

## 📋 可用命令

### 建置命令
```bash
make all              # 編譯所有版本
make clean            # 清理編譯文件
make rebuild          # 清理並重新編譯
```

### 測試命令
```bash
make test-openmp      # 測試OpenMP支援
make test-cuda        # 測試CUDA支援
make set-threads      # 顯示線程設定
```

### 運行命令
```bash
make run-basic        # 運行基礎版本
make run-fine         # 運行精細網格版本
make run-openmp       # 運行OpenMP版本
make run-cuda         # 運行CUDA版本
make benchmark        # 完整性能測試
```

### 幫助命令
```bash
make help             # 顯示完整幫助
```

## 🔧 環境設定

### OpenMP設定
```bash
export OMP_NUM_THREADS=4    # 設定線程數
nproc                       # 查看可用核心數
```

### CUDA檢查
```bash
nvidia-smi              # 查看GPU狀態
nvcc --version          # 查看CUDA版本
```

## 📖 詳細文檔

- **[性能分析報告](performance_analysis.md)** - 詳細的性能測試結果和分析
- **[技術規格文件](technical_specifications.md)** - 完整的技術實現細節
- **[結果分析報告](results_analysis.md)** - 成功參數的統計分析

## 🎓 學習價值

這個項目展示了：
1. **並行計算完整階梯**: 從單線程到GPU的性能提升
2. **實際應用場景**: 科學計算中的並行化實踐
3. **性能優化技術**: 達到16.7倍實際加速比
4. **跨平台開發**: Windows + WSL + Linux + GPU環境

## 🤝 貢獻與反饋

歡迎提出改進建議或報告問題：
- 性能優化建議
- 物理模型改進
- 文檔完善
- Bug報告

## 📄 授權

本項目用於教育和研究目的。

---

🚀 **準備好體驗並行計算的威力了嗎？運行 `make benchmark` 開始測試！**

*最後更新: 2025年5月27日*
*測試環境: WSL2 + NVIDIA RTX 3060 Ti*
