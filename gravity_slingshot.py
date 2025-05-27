import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import math

class GravitySlingshot:
    def __init__(self):
        # 物理常數
        self.G = 6.67430e-11  # 重力常數
        self.dt = 0.1  # 時間步長
        
        # 月球參數
        self.moon_mass = 7.342e22  # 月球質量 (kg)
        self.moon_radius = 1737000  # 月球半徑 (m)
        self.moon_pos = np.array([-2e8, 0])  # 月球初始位置 (m)
        self.moon_vel = np.array([1000, 0])  # 月球速度 (m/s)
        
        # 太空船參數
        self.spacecraft_mass = 1000  # 太空船質量 (kg)
        self.spacecraft_pos = np.array([-5e8, -3e8])  # 太空船初始位置
        self.spacecraft_vel = np.array([2000, 1500])  # 太空船初始速度
        
        # 記錄軌跡
        self.moon_trajectory = [self.moon_pos.copy()]
        self.spacecraft_trajectory = [self.spacecraft_pos.copy()]
        self.time_steps = [0]
        
        # 縮放因子 (用於顯示)
        self.scale = 1e8
        
    def gravitational_force(self, pos1, pos2, mass1, mass2):
        """計算兩個物體之間的重力"""
        r_vector = pos2 - pos1
        r_magnitude = np.linalg.norm(r_vector)
        
        # 避免除零錯誤
        if r_magnitude < self.moon_radius:
            r_magnitude = self.moon_radius
            
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
        
        # 計算月球受到太空船的重力 (牛頓第三定律)
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
    
    def run_simulation(self, duration=1000):
        """運行模擬"""
        steps = int(duration / self.dt)
        
        for _ in range(steps):
            self.update_physics()
            
            # 如果太空船距離太遠就停止
            distance = np.linalg.norm(self.spacecraft_pos - self.moon_pos)
            if distance > 1e9:  # 10億米
                break
    
    def create_animation(self):
        """創建動畫"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor('black')
        
        # 設置坐標軸
        moon_traj = np.array(self.moon_trajectory)
        spacecraft_traj = np.array(self.spacecraft_trajectory)
        
        # 找到合適的顯示範圍
        all_x = np.concatenate([moon_traj[:, 0], spacecraft_traj[:, 0]]) / self.scale
        all_y = np.concatenate([moon_traj[:, 1], spacecraft_traj[:, 1]]) / self.scale
        
        margin = 0.5
        x_min, x_max = all_x.min() - margin, all_x.max() + margin
        y_min, y_max = all_y.min() - margin, all_y.max() + margin
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title('重力彈弓模擬 (Gravity Slingshot Simulation)', 
                     color='white', fontsize=16)
        ax.set_xlabel('距離 (100 Mm)', color='white')
        ax.set_ylabel('距離 (100 Mm)', color='white')
        
        # 創建圖形元素
        moon_circle = Circle((0, 0), 0.1, color='lightgray', label='月球')
        spacecraft_point, = ax.plot([], [], 'wo', markersize=8, label='太空船')
        moon_trail, = ax.plot([], [], 'gray', alpha=0.5, linewidth=2)
        spacecraft_trail, = ax.plot([], [], 'cyan', alpha=0.7, linewidth=2)
        
        # 速度向量
        moon_vel_arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                   arrowprops=dict(arrowstyle='->', 
                                                 color='white', lw=2))
        spacecraft_vel_arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                         arrowprops=dict(arrowstyle='->', 
                                                       color='yellow', lw=2))
        
        ax.add_patch(moon_circle)
        ax.legend(loc='upper right')
        
        # 添加信息文本
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           color='white', verticalalignment='top',
                           fontfamily='monospace')
        
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
                vel_scale = 0.1
                moon_vel_end = moon_pos + moon_vel * vel_scale
                spacecraft_vel_end = spacecraft_pos + spacecraft_vel * vel_scale
                
                moon_vel_arrow.set_position(moon_pos)
                moon_vel_arrow.xy = moon_vel_end
                
                spacecraft_vel_arrow.set_position(spacecraft_pos)
                spacecraft_vel_arrow.xy = spacecraft_vel_end
            
            # 更新信息
            distance = np.linalg.norm(spacecraft_traj[frame] - moon_traj[frame]) / 1000000  # 轉換為千公里
            time = self.time_steps[frame]
            
            info_text.set_text(f'時間: {time:.1f} 秒\n'
                             f'距離: {distance:.1f} 千公里\n'
                             f'幀數: {frame}/{len(self.moon_trajectory)-1}')
            
            return [moon_circle, spacecraft_point, moon_trail, spacecraft_trail, 
                   moon_vel_arrow, spacecraft_vel_arrow, info_text]
        
        # 創建動畫
        frames = min(len(self.moon_trajectory), 800)  # 限制幀數
        anim = animation.FuncAnimation(fig, animate, frames=frames,
                                     interval=50, blit=True, repeat=True)
        
        plt.tight_layout()
        return fig, anim

def main():
    print("開始重力彈弓模擬...")
    
    # 創建模擬實例
    sim = GravitySlingshot()
    
    # 運行物理模擬
    print("運行物理計算...")
    sim.run_simulation(duration=800)
    
    print(f"模擬完成！總共 {len(sim.moon_trajectory)} 個時間步")
    
    # 創建動畫
    print("創建動畫...")
    fig, anim = sim.create_animation()
    
    # 顯示動畫
    plt.show()
    
    # 可選：保存為GIF或MP4
    save_choice = input("是否要保存動畫？(y/n): ")
    if save_choice.lower() == 'y':
        print("保存動畫為 gravity_slingshot.gif...")
        anim.save('gravity_slingshot.gif', writer='pillow', fps=20)
        print("動畫已保存！")

if __name__ == "__main__":
    main() 