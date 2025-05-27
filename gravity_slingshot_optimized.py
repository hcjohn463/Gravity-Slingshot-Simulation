import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import math

class OptimizedGravitySlingshot:
    def __init__(self):
        # 物理常數（調整用於更好的可視化效果）
        self.G = 6.67430e-11 * 1e6  # 放大重力常數用於演示
        self.dt = 50  # 時間步長（秒）
        
        # 月球參數
        self.moon_mass = 7.342e22  # 月球質量 (kg)
        self.moon_radius = 1737000  # 月球半徑 (m)
        self.moon_pos = np.array([0, 0], dtype=float)  # 月球初始位置 (m)
        self.moon_vel = np.array([800, 0], dtype=float)  # 月球速度 (m/s) - 往右移動
        
        # 太空船參數 - 調整為從左下方接近
        self.spacecraft_mass = 1000  # 太空船質量 (kg)
        self.spacecraft_pos = np.array([-8e7, -6e7], dtype=float)  # 太空船初始位置
        self.spacecraft_vel = np.array([1200, 800], dtype=float)  # 太空船初始速度
        
        # 記錄軌跡
        self.moon_trajectory = [self.moon_pos.copy()]
        self.spacecraft_trajectory = [self.spacecraft_pos.copy()]
        self.time_steps = [0]
        self.spacecraft_speeds = [np.linalg.norm(self.spacecraft_vel)]
        
        # 縮放因子 (用於顯示)
        self.scale = 1e7
        
    def gravitational_force(self, pos1, pos2, mass1, mass2):
        """計算兩個物體之間的重力"""
        r_vector = pos2 - pos1
        r_magnitude = np.linalg.norm(r_vector)
        
        # 避免除零錯誤
        min_distance = self.moon_radius * 2
        if r_magnitude < min_distance:
            r_magnitude = min_distance
            
        # 重力大小
        force_magnitude = self.G * mass1 * mass2 / (r_magnitude ** 2)
        
        # 重力方向
        force_direction = r_vector / r_magnitude
        
        return force_magnitude * force_direction
    
    def update_physics(self):
        """更新物理狀態"""
        # 計算太空船受到月球的重力
        force_on_spacecraft = self.gravitational_force(
            self.spacecraft_pos, self.moon_pos, 
            self.spacecraft_mass, self.moon_mass
        )
        
        # 月球質量遠大於太空船，所以月球幾乎不受影響
        # 但為了物理準確性，我們仍然計算
        force_on_moon = -force_on_spacecraft
        
        # 更新加速度
        spacecraft_acc = force_on_spacecraft / self.spacecraft_mass
        moon_acc = force_on_moon / self.moon_mass
        
        # 更新速度
        self.spacecraft_vel += spacecraft_acc * self.dt
        self.moon_vel += moon_acc * self.dt
        
        # 更新位置
        self.spacecraft_pos += self.spacecraft_vel * self.dt
        self.moon_pos += self.moon_vel * self.dt
        
        # 記錄軌跡
        self.moon_trajectory.append(self.moon_pos.copy())
        self.spacecraft_trajectory.append(self.spacecraft_pos.copy())
        self.time_steps.append(len(self.time_steps) * self.dt)
        self.spacecraft_speeds.append(np.linalg.norm(self.spacecraft_vel))
    
    def run_simulation(self, duration=5000):
        """運行模擬"""
        steps = int(duration / self.dt)
        
        for step in range(steps):
            self.update_physics()
            
            # 如果太空船距離太遠就停止
            distance = np.linalg.norm(self.spacecraft_pos - self.moon_pos)
            if distance > 3e8:  # 3億米
                break
                
            # 進度顯示
            if step % 10 == 0:
                print(f"模擬進度: {step}/{steps} ({step/steps*100:.1f}%)")
    
    def analyze_trajectory(self):
        """分析軌跡數據"""
        speeds = np.array(self.spacecraft_speeds)
        initial_speed = speeds[0]
        max_speed = np.max(speeds)
        final_speed = speeds[-1]
        
        speed_gain = final_speed - initial_speed
        
        print(f"\n=== 軌跡分析 ===")
        print(f"初始速度: {initial_speed:.2f} m/s")
        print(f"最大速度: {max_speed:.2f} m/s")
        print(f"最終速度: {final_speed:.2f} m/s")
        print(f"速度增益: {speed_gain:.2f} m/s ({speed_gain/initial_speed*100:.1f}%)")
        
        return {
            'initial_speed': initial_speed,
            'max_speed': max_speed,
            'final_speed': final_speed,
            'speed_gain': speed_gain
        }
    
    def create_animation(self):
        """創建動畫"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 主動畫面板
        ax1.set_facecolor('black')
        
        # 設置坐標軸
        moon_traj = np.array(self.moon_trajectory)
        spacecraft_traj = np.array(self.spacecraft_trajectory)
        
        # 找到合適的顯示範圍
        all_x = np.concatenate([moon_traj[:, 0], spacecraft_traj[:, 0]]) / self.scale
        all_y = np.concatenate([moon_traj[:, 1], spacecraft_traj[:, 1]]) / self.scale
        
        margin = 2
        x_center = (all_x.min() + all_x.max()) / 2
        y_center = (all_y.min() + all_y.max()) / 2
        x_range = (all_x.max() - all_x.min()) + margin * 2
        y_range = (all_y.max() - all_y.min()) + margin * 2
        
        max_range = max(x_range, y_range)
        
        ax1.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax1.set_ylim(y_center - max_range/2, y_center + max_range/2)
        ax1.set_aspect('equal')
        ax1.set_title('重力彈弓模擬 (Gravity Slingshot)', color='white', fontsize=14)
        ax1.set_xlabel('距離 (10 Mm)', color='white')
        ax1.set_ylabel('距離 (10 Mm)', color='white')
        ax1.tick_params(colors='white')
        
        # 速度圖面板
        ax2.set_facecolor('black')
        ax2.set_title('太空船速度變化', color='white', fontsize=14)
        ax2.set_xlabel('時間 (秒)', color='white')
        ax2.set_ylabel('速度 (m/s)', color='white')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.3)
        
        # 創建圖形元素
        moon_circle = Circle((0, 0), 0.3, color='lightgray', label='月球', alpha=0.8)
        spacecraft_point, = ax1.plot([], [], 'wo', markersize=6, label='太空船')
        moon_trail, = ax1.plot([], [], 'gray', alpha=0.6, linewidth=1.5, label='月球軌跡')
        spacecraft_trail, = ax1.plot([], [], 'cyan', alpha=0.8, linewidth=2, label='太空船軌跡')
        
        # 速度向量
        moon_vel_arrow = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                                    arrowprops=dict(arrowstyle='->', 
                                                  color='white', lw=2))
        spacecraft_vel_arrow = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                                          arrowprops=dict(arrowstyle='->', 
                                                        color='yellow', lw=2))
        
        # 速度曲線
        speed_line, = ax2.plot([], [], 'cyan', linewidth=2)
        speed_point, = ax2.plot([], [], 'ro', markersize=5)
        
        ax1.add_patch(moon_circle)
        ax1.legend(loc='upper left')
        
        # 添加信息文本
        info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                           color='white', verticalalignment='top',
                           fontfamily='monospace', fontsize=10)
        
        def animate(frame):
            if frame >= len(self.moon_trajectory):
                return
                
            # 當前位置
            moon_pos = moon_traj[frame] / self.scale
            spacecraft_pos = spacecraft_traj[frame] / self.scale
            
            # 更新位置
            moon_circle.center = (moon_pos[0], moon_pos[1])
            spacecraft_point.set_data([spacecraft_pos[0]], [spacecraft_pos[1]])
            
            # 更新軌跡
            if frame > 0:
                moon_trail.set_data(moon_traj[:frame, 0] / self.scale,
                                  moon_traj[:frame, 1] / self.scale)
                spacecraft_trail.set_data(spacecraft_traj[:frame, 0] / self.scale,
                                        spacecraft_traj[:frame, 1] / self.scale)
            
            # 計算並顯示速度向量
            if frame < len(self.moon_trajectory) - 1:
                moon_vel = (moon_traj[frame+1] - moon_traj[frame]) / (self.dt * self.scale)
                spacecraft_vel = (spacecraft_traj[frame+1] - spacecraft_traj[frame]) / (self.dt * self.scale)
                
                # 縮放速度向量以便顯示
                vel_scale = 0.3
                moon_vel_end = moon_pos + moon_vel * vel_scale
                spacecraft_vel_end = spacecraft_pos + spacecraft_vel * vel_scale
                
                moon_vel_arrow.set_position(moon_pos)
                moon_vel_arrow.xy = moon_vel_end
                
                spacecraft_vel_arrow.set_position(spacecraft_pos)
                spacecraft_vel_arrow.xy = spacecraft_vel_end
            
            # 更新速度圖
            times = np.array(self.time_steps[:frame+1])
            speeds = np.array(self.spacecraft_speeds[:frame+1])
            
            speed_line.set_data(times, speeds)
            if frame < len(speeds):
                speed_point.set_data([times[frame]], [speeds[frame]])
            
            ax2.relim()
            ax2.autoscale_view()
            
            # 更新信息
            distance = np.linalg.norm(spacecraft_traj[frame] - moon_traj[frame]) / 1000000
            time = self.time_steps[frame] / 3600  # 轉換為小時
            current_speed = self.spacecraft_speeds[frame]
            speed_change = current_speed - self.spacecraft_speeds[0]
            
            info_text.set_text(f'時間: {time:.2f} 小時\n'
                             f'距離: {distance:.1f} 千公里\n'
                             f'當前速度: {current_speed:.0f} m/s\n'
                             f'速度變化: {speed_change:+.0f} m/s\n'
                             f'幀數: {frame}/{len(self.moon_trajectory)-1}')
            
            return [moon_circle, spacecraft_point, moon_trail, spacecraft_trail, 
                   moon_vel_arrow, spacecraft_vel_arrow, speed_line, speed_point, info_text]
        
        # 創建動畫
        frames = min(len(self.moon_trajectory), 500)  # 限制幀數
        anim = animation.FuncAnimation(fig, animate, frames=frames,
                                     interval=100, blit=True, repeat=True)
        
        plt.tight_layout()
        return fig, anim

def main():
    print("🚀 開始重力彈弓模擬...")
    
    # 創建模擬實例
    sim = OptimizedGravitySlingshot()
    
    # 運行物理模擬
    print("⚡ 運行物理計算...")
    sim.run_simulation(duration=4000)
    
    print(f"✅ 模擬完成！總共 {len(sim.moon_trajectory)} 個時間步")
    
    # 分析結果
    analysis = sim.analyze_trajectory()
    
    # 創建動畫
    print("🎬 創建動畫...")
    fig, anim = sim.create_animation()
    
    # 顯示動畫
    print("🎥 顯示動畫...")
    plt.show()
    
    # 可選：保存為GIF
    save_choice = input("\n💾 是否要保存動畫為GIF？(y/n): ")
    if save_choice.lower() == 'y':
        print("📁 保存動畫為 gravity_slingshot_optimized.gif...")
        try:
            anim.save('gravity_slingshot_optimized.gif', writer='pillow', fps=10)
            print("✅ 動畫已保存！")
        except Exception as e:
            print(f"❌ 保存失敗: {e}")

if __name__ == "__main__":
    main() 