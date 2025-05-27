# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import math

class EnhancedSlingshot:
    def __init__(self):
        # Enhanced physics constants for dramatic effect
        self.G = 6.67430e-11
        self.dt = 200  # Time step (seconds)
        
        # Enhanced mode - dramatic effect with clear U-turn
        self.moon_mass = 7.342e22 * 50  # 50x real moon mass for dramatic effect
        self.moon_radius = 1737000  # Real moon radius
        self.moon_pos = np.array([0, 0], dtype=float)
        self.moon_vel = np.array([1500, 0], dtype=float)  # 1.5 km/s rightward
        
        # Spacecraft parameters - optimized for dramatic slingshot
        self.spacecraft_mass = 1000
        self.spacecraft_pos = np.array([-3e7, -2e7], dtype=float)  # 30M km behind, 20M km below
        self.spacecraft_vel = np.array([2200, 1400], dtype=float)  # Optimized initial velocity
        
        # Data recording
        self.moon_trajectory = [self.moon_pos.copy()]
        self.spacecraft_trajectory = [self.spacecraft_pos.copy()]
        self.time_steps = [0]
        self.spacecraft_speeds = [np.linalg.norm(self.spacecraft_vel)]
        self.spacecraft_directions = [np.arctan2(self.spacecraft_vel[1], self.spacecraft_vel[0])]
        
        # Scale for visualization (10 million km = 1 unit)
        self.scale = 1e7
        
        # Calculate escape velocity at initial position
        initial_distance = np.linalg.norm(self.spacecraft_pos - self.moon_pos)
        self.escape_velocity = self.calculate_escape_velocity(initial_distance)
        print(f"🌙 Enhanced Gravity Slingshot (Moon mass x50)")
        print(f"Escape velocity at initial distance ({initial_distance/1000:.0f} km): {self.escape_velocity/1000:.2f} km/s")
        print(f"Initial speed: {np.linalg.norm(self.spacecraft_vel)/1000:.2f} km/s")
        print(f"Sufficient for escape: {'Yes' if np.linalg.norm(self.spacecraft_vel) > self.escape_velocity else 'No'}")
    
    def calculate_escape_velocity(self, distance):
        """Calculate escape velocity at given distance"""
        return np.sqrt(2 * self.G * self.moon_mass / distance)
    
    def gravitational_force(self, pos1, pos2, mass1, mass2):
        """Calculate gravitational force"""
        r_vector = pos2 - pos1
        r_magnitude = np.linalg.norm(r_vector)
        
        # Prevent collision
        min_distance = self.moon_radius * 1.2
        if r_magnitude < min_distance:
            r_magnitude = min_distance
            
        force_magnitude = self.G * mass1 * mass2 / (r_magnitude ** 2)
        force_direction = r_vector / r_magnitude
        
        return force_magnitude * force_direction
    
    def update_physics(self):
        """Update simulation state"""
        # Calculate gravitational force
        force = self.gravitational_force(
            self.spacecraft_pos, self.moon_pos,
            self.spacecraft_mass, self.moon_mass
        )
        
        # Update spacecraft
        acceleration = force / self.spacecraft_mass
        self.spacecraft_vel += acceleration * self.dt
        self.spacecraft_pos += self.spacecraft_vel * self.dt
        
        # Update moon (constant velocity)
        self.moon_pos += self.moon_vel * self.dt
        
        # Record data
        self.moon_trajectory.append(self.moon_pos.copy())
        self.spacecraft_trajectory.append(self.spacecraft_pos.copy())
        self.time_steps.append(len(self.time_steps) * self.dt)
        self.spacecraft_speeds.append(np.linalg.norm(self.spacecraft_vel))
        self.spacecraft_directions.append(np.arctan2(self.spacecraft_vel[1], self.spacecraft_vel[0]))
    
    def run_simulation(self, max_steps=300):
        """Run simulation with step limit"""
        print("Running enhanced slingshot simulation...")
        
        for step in range(max_steps):
            self.update_physics()
            
            # Check if we should stop
            distance = np.linalg.norm(self.spacecraft_pos - self.moon_pos)
            
            # Stop if spacecraft is getting very far
            if distance > 2e8:  # 200 million km
                print(f"Spacecraft distance too large. Stopping at step {step}")
                break
                
            # Progress reporting
            if step % 20 == 0:
                dist_km = distance / 1000
                speed_kms = self.spacecraft_speeds[-1] / 1000
                print(f"Step {step}: Distance = {dist_km:.0f} km, Speed = {speed_kms:.2f} km/s")
        
        print(f"Simulation finished! Total steps: {len(self.spacecraft_trajectory)}")
    
    def analyze_slingshot(self):
        """Analyze the slingshot results"""
        speeds = np.array(self.spacecraft_speeds)
        directions = np.array(self.spacecraft_directions)
        
        initial_speed = speeds[0]
        max_speed = np.max(speeds)
        final_speed = speeds[-1]
        
        initial_dir = directions[0] * 180 / np.pi
        final_dir = directions[-1] * 180 / np.pi
        direction_change = final_dir - initial_dir
        
        # Normalize direction change
        while direction_change > 180:
            direction_change -= 360
        while direction_change < -180:
            direction_change += 360
        
        speed_gain = final_speed - initial_speed
        speed_percentage = (speed_gain / initial_speed) * 100
        
        # Find closest approach
        min_distance = float('inf')
        for i in range(len(self.spacecraft_trajectory)):
            dist = np.linalg.norm(
                np.array(self.spacecraft_trajectory[i]) - 
                np.array(self.moon_trajectory[i])
            )
            min_distance = min(min_distance, dist)
        
        print(f"\n=== Enhanced Slingshot Analysis ===")
        print(f"Initial speed: {initial_speed/1000:.2f} km/s")
        print(f"Maximum speed: {max_speed/1000:.2f} km/s")
        print(f"Final speed: {final_speed/1000:.2f} km/s")
        print(f"Speed gain: {speed_gain:.0f} m/s ({speed_percentage:.1f}%)")
        print(f"Direction change: {direction_change:.1f}°")
        print(f"Closest approach: {min_distance/1000:.0f} km")
        
        return {
            'speed_gain': speed_gain,
            'direction_change': direction_change,
            'closest_approach': min_distance/1000
        }
    
    def create_animation(self):
        """Create visualization"""
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.patch.set_facecolor('black')
        
        # Convert to display coordinates
        moon_traj = np.array(self.moon_trajectory) / self.scale
        spacecraft_traj = np.array(self.spacecraft_trajectory) / self.scale
        
        # Setup main plot
        ax1.set_facecolor('black')
        
        # Calculate bounds
        all_x = np.concatenate([moon_traj[:, 0], spacecraft_traj[:, 0]])
        all_y = np.concatenate([moon_traj[:, 1], spacecraft_traj[:, 1]])
        
        margin = 1.0
        x_center = np.mean(all_x)
        y_center = np.mean(all_y)
        plot_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) + 2*margin
        
        ax1.set_xlim(x_center - plot_range/2, x_center + plot_range/2)
        ax1.set_ylim(y_center - plot_range/2, y_center + plot_range/2)
        ax1.set_aspect('equal')
        ax1.set_title('Enhanced Gravity Slingshot (50x Moon Mass)', fontsize=20, color='white', pad=20)
        ax1.set_xlabel('Distance (10 Million km)', fontsize=14, color='white')
        ax1.set_ylabel('Distance (10 Million km)', fontsize=14, color='white')
        ax1.grid(True, alpha=0.3, color='gray')
        
        # Setup speed plot
        ax2.set_facecolor('black')
        ax2.set_title('Speed During Maneuver', fontsize=20, color='white', pad=20)
        ax2.set_xlabel('Time (hours)', fontsize=14, color='white')
        ax2.set_ylabel('Speed (km/s)', fontsize=14, color='white')
        ax2.grid(True, alpha=0.3, color='gray')
        
        # Create visual elements
        moon_size = 0.25
        moon_circle = Circle((0, 0), moon_size, color='lightgray', alpha=0.9, 
                           label='Moon', zorder=10)
        spacecraft_dot, = ax1.plot([], [], 'o', color='white', markersize=12, 
                                 label='Spacecraft', zorder=11)
        
        # Path trails
        moon_path, = ax1.plot([], [], color='gray', alpha=0.7, linewidth=3, 
                            label='Moon Path', zorder=5)
        spacecraft_path, = ax1.plot([], [], color='cyan', alpha=1.0, linewidth=4, 
                                  label='Spacecraft Path', zorder=6)
        
        # Velocity indicators
        arrow_scale = 1.2
        moon_velocity = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                                   arrowprops=dict(arrowstyle='->', color='yellow', 
                                                 lw=4, alpha=0.9), zorder=8)
        spacecraft_velocity = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                                         arrowprops=dict(arrowstyle='->', color='lime', 
                                                       lw=4, alpha=1.0), zorder=9)
        
        # Speed chart elements
        speed_chart, = ax2.plot([], [], color='cyan', linewidth=4, alpha=0.9)
        current_speed_marker, = ax2.plot([], [], 'o', color='red', markersize=10, zorder=10)
        
        ax1.add_patch(moon_circle)
        ax1.legend(loc='upper right', fontsize=14, framealpha=0.9)
        
        # Status display
        status_display = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                                color='lime', fontsize=14, weight='bold',
                                verticalalignment='top',
                                bbox=dict(boxstyle="round,pad=0.6", facecolor='black', 
                                        alpha=0.8, edgecolor='lime', linewidth=2))
        
        def animate(frame):
            if frame >= len(moon_traj):
                return []
            
            # Current positions
            moon_x, moon_y = moon_traj[frame]
            spacecraft_x, spacecraft_y = spacecraft_traj[frame]
            
            # Update positions
            moon_circle.center = (moon_x, moon_y)
            spacecraft_dot.set_data([spacecraft_x], [spacecraft_y])
            
            # Update paths
            moon_path.set_data(moon_traj[:frame+1, 0], moon_traj[:frame+1, 1])
            spacecraft_path.set_data(spacecraft_traj[:frame+1, 0], spacecraft_traj[:frame+1, 1])
            
            # Update velocity arrows
            if frame < len(moon_traj) - 1:
                # Moon velocity (constant)
                moon_vel_arrow = np.array([1, 0]) * arrow_scale
                moon_velocity.set_position((moon_x, moon_y))
                moon_velocity.xy = (moon_x + moon_vel_arrow[0], moon_y + moon_vel_arrow[1])
                
                # Spacecraft velocity
                if frame < len(spacecraft_traj) - 1:
                    craft_vel = spacecraft_traj[frame+1] - spacecraft_traj[frame]
                    vel_mag = np.linalg.norm(craft_vel)
                    if vel_mag > 0:
                        craft_vel_norm = craft_vel / vel_mag * arrow_scale
                        spacecraft_velocity.set_position((spacecraft_x, spacecraft_y))
                        spacecraft_velocity.xy = (spacecraft_x + craft_vel_norm[0], 
                                                spacecraft_y + craft_vel_norm[1])
            
            # Update speed chart
            times_hours = np.array(self.time_steps[:frame+1]) / 3600
            speeds_kms = np.array(self.spacecraft_speeds[:frame+1]) / 1000
            
            speed_chart.set_data(times_hours, speeds_kms)
            if frame < len(speeds_kms):
                current_speed_marker.set_data([times_hours[frame]], [speeds_kms[frame]])
            
            ax2.relim()
            ax2.autoscale_view()
            
            # Update status
            current_time = self.time_steps[frame] / 3600
            distance_km = np.linalg.norm((np.array([spacecraft_x, spacecraft_y]) - 
                                        np.array([moon_x, moon_y])) * self.scale) / 1000
            current_speed_kms = self.spacecraft_speeds[frame] / 1000
            speed_delta = (self.spacecraft_speeds[frame] - self.spacecraft_speeds[0]) / 1000
            
            status_display.set_text(
                f'Time: {current_time:.1f} hrs\n'
                f'Distance: {distance_km:.0f} km\n'
                f'Speed: {current_speed_kms:.2f} km/s\n'
                f'Δ Speed: {speed_delta:+.2f} km/s\n'
                f'Frame: {frame+1}/{len(moon_traj)}'
            )
            
            return [moon_circle, spacecraft_dot, moon_path, spacecraft_path,
                   moon_velocity, spacecraft_velocity, speed_chart, 
                   current_speed_marker, status_display]
        
        # Create animation
        total_frames = min(len(moon_traj), 400)
        frame_interval = max(50, len(moon_traj) // 8)
        
        anim = animation.FuncAnimation(
            fig, animate, frames=total_frames,
            interval=frame_interval, blit=True, repeat=True
        )
        
        plt.tight_layout()
        return fig, anim

def main():
    print("🚀 Enhanced Gravity Slingshot Simulation")
    print("=" * 50)
    print("Dramatic U-turn effect with 50x Moon mass")
    print("Shows the classic slingshot trajectory!")
    
    # Initialize simulation
    sim = EnhancedSlingshot()
    
    print(f"\n{'='*60}")
    print("Physics Explanation:")
    print("• Exaggerated gravity (50x Moon mass) for dramatic visual effect")
    print("• Shows the classic 'U-turn' trajectory that makes slingshots famous")
    print("• Demonstrates: speed increase + direction change = gravity assist")
    print("• Perfect for understanding the basic physics concept")
    print(f"{'='*60}")
    
    # Run simulation
    sim.run_simulation(max_steps=250)
    
    # Analyze results
    results = sim.analyze_slingshot()
    
    # Analyze escape success
    final_distance = np.linalg.norm(
        np.array(sim.spacecraft_trajectory[-1]) - 
        np.array(sim.moon_trajectory[-1])
    )
    final_speed = sim.spacecraft_speeds[-1]
    escape_vel_at_final = sim.calculate_escape_velocity(final_distance)
    
    print(f"\n=== Escape Analysis ===")
    print(f"Final distance: {final_distance/1000:.0f} km")
    print(f"Final speed: {final_speed/1000:.2f} km/s") 
    print(f"Escape velocity at final distance: {escape_vel_at_final/1000:.2f} km/s")
    print(f"Successfully escaped: {'Yes' if final_speed > escape_vel_at_final else 'No'}")
    
    if final_speed <= escape_vel_at_final:
        print("❌ Spacecraft failed to escape, will be captured in orbit")
        print("💡 This happens when initial velocity is insufficient for escape")
    else:
        print("✅ Spacecraft successfully escaped! Classic gravity slingshot effect!")
        print("💡 The spacecraft gained speed and changed direction - perfect gravity assist!")
    
    # Create visualization
    print("\n🎬 Generating animation...")
    fig, anim = sim.create_animation()
    
    print("🎥 Displaying enhanced slingshot maneuver...")
    plt.show()
    
    # Save option
    save_option = input("\n💾 Save animation as GIF? (y/n): ")
    if save_option.lower() == 'y':
        print("💾 Saving animation...")
        try:
            anim.save('enhanced_slingshot.gif', writer='pillow', fps=12,
                     savefig_kwargs={'facecolor': 'black'})
            print("✅ Animation saved as 'enhanced_slingshot.gif'")
        except Exception as e:
            print(f"❌ Save error: {e}")
    
    print("\n🎯 Enhanced simulation complete!")
    return sim, results

if __name__ == "__main__":
    main() 