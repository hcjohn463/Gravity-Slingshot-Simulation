#!/usr/bin/env python3
"""
CUDA Results Visualization
==========================

讀取 successful_parameters_cuda.txt 並生成參數空間視覺化圖表
展示精細網格（1 m/s步長）的成功參數分布

Author: Generated for CUDA results analysis
Date: 2025-05-27
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_cuda_results(filename='successful_parameters_cuda.txt'):
    """
    讀取CUDA結果文件
    
    Returns:
        pandas.DataFrame: 包含 Vx, Vy, Speed, HitTime 的數據框
    """
    print(f"📖 Reading CUDA results from {filename}...")
    
    # 讀取文件並解析數據
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳過註釋行和空行
            if line.startswith('#') or not line:
                continue
            
            # 解析數據行：格式為 " vx,   vy,    speed,  hit_time"
            try:
                parts = [x.strip() for x in line.split(',')]
                if len(parts) >= 4:
                    vx = int(parts[0])
                    vy = int(parts[1]) 
                    speed = int(parts[2])
                    hit_time = int(parts[3])
                    data.append([vx, vy, speed, hit_time])
            except ValueError:
                continue
    
    # 轉換為DataFrame
    df = pd.DataFrame(data, columns=['Vx', 'Vy', 'Speed', 'HitTime'])
    print(f"✅ Successfully loaded {len(df)} successful parameters")
    return df

def create_parameter_space_plot(df):
    """
    創建參數空間視覺化圖表（僅成功/失敗分布）
    
    Args:
        df: 成功參數的DataFrame
    """
    print("🎨 Creating parameter space visualization...")
    
    # 設置圖表
    plt.figure(figsize=(12, 8))
    
    # 創建完整的參數網格
    vx_range = np.arange(-4000, -999, 1)  # 1 m/s步長
    vy_range = np.arange(1000, 4001, 1)   # 1 m/s步長
    
    # 創建網格
    VX, VY = np.meshgrid(vx_range, vy_range)
    
    # 創建結果矩陣（0=失敗，1=成功）
    results = np.zeros_like(VX)
    
    # 標記成功的參數
    for _, row in df.iterrows():
        vx_idx = int(row['Vx']) - (-4000)  # 轉換為索引
        vy_idx = int(row['Vy']) - 1000     # 轉換為索引
        if 0 <= vx_idx < len(vx_range) and 0 <= vy_idx < len(vy_range):
            results[vy_idx, vx_idx] = 1
    
    # 繪製圖表
    plt.imshow(results, extent=[-4000, -1000, 1000, 4000], 
               cmap='RdYlGn', alpha=0.8, origin='lower', aspect='auto')
    
    # 添加成功點的散點圖（突出顯示）
    plt.scatter(df['Vx'], df['Vy'], c='blue', s=1, alpha=0.8, label=f'Success ({len(df)} points)')
    
    plt.xlabel('Initial X Velocity (m/s)', fontsize=12)
    plt.ylabel('Initial Y Velocity (m/s)', fontsize=12)
    plt.title('Slingshot Parameter Space - Success/Failure Distribution\n(CUDA Results: Fine Grid 1 m/s resolution)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存圖片
    output_files = [
        'cuda_parameter_sweep_results.png',
        'cuda_parameter_sweep_results.pdf',
        'cuda_parameter_sweep_results_white.png'
    ]
    
    # 黑色背景版本
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 8))
    plt.imshow(results, extent=[-4000, -1000, 1000, 4000], 
               cmap='RdYlGn', alpha=0.8, origin='lower', aspect='auto')
    plt.scatter(df['Vx'], df['Vy'], c='cyan', s=1, alpha=0.9, label=f'Success ({len(df)} points)')
    plt.xlabel('Initial X Velocity (m/s)', fontsize=12, color='white')
    plt.ylabel('Initial Y Velocity (m/s)', fontsize=12, color='white')
    plt.title('Slingshot Parameter Space - Success/Failure Distribution\n(CUDA Results: Fine Grid 1 m/s resolution)', 
              fontsize=14, color='white')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('cuda_parameter_sweep_results_dark.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.savefig('cuda_parameter_sweep_results_dark.pdf', dpi=300, bbox_inches='tight', facecolor='black')
    print(f"💾 Saved: cuda_parameter_sweep_results_dark.png")
    print(f"💾 Saved: cuda_parameter_sweep_results_dark.pdf")
    
    # 白色背景版本
    plt.style.use('default')
    plt.figure(figsize=(12, 8))
    plt.imshow(results, extent=[-4000, -1000, 1000, 4000], 
               cmap='RdYlGn', alpha=0.8, origin='lower', aspect='auto')
    plt.scatter(df['Vx'], df['Vy'], c='blue', s=1, alpha=0.8, label=f'Success ({len(df)} points)')
    plt.xlabel('Initial X Velocity (m/s)', fontsize=12)
    plt.ylabel('Initial Y Velocity (m/s)', fontsize=12)
    plt.title('Slingshot Parameter Space - Success/Failure Distribution\n(CUDA Results: Fine Grid 1 m/s resolution)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('cuda_parameter_sweep_results.png', dpi=300, bbox_inches='tight')
    print(f"💾 Saved: cuda_parameter_sweep_results.png")
    
    plt.show()

def main():
    """主程序"""
    print("🚀 CUDA Slingshot Results Visualization")
    print("=" * 50)
    print("Reading CUDA GPU computation results and generating parameter sweep visualization...")
    print("Fine grid resolution: 1 m/s step size")
    print("Total tested combinations: 9,006,001")
    print()
    
    try:
        # 讀取CUDA結果
        df = read_cuda_results()
        
        if df.empty:
            print("❌ No data found in results file!")
            return
        
        # 基本統計
        print(f"🎯 Total successful parameters: {len(df)}")
        print(f"📐 Success rate: {len(df)/9006001*100:.4f}%")
        
        # 創建視覺化圖表
        create_parameter_space_plot(df)
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Generated parameter sweep visualization with {len(df)} successful parameters")
        
    except FileNotFoundError:
        print("❌ Error: successful_parameters_cuda.txt not found!")
        print("   Please make sure you're in the c_version directory")
        print("   and that CUDA results have been generated.")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    main() 