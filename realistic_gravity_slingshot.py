import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import math

class RealisticGravitySlingshot:
    def __init__(self):
        # Physical constants - more realistic values
        self.G = 6.67430e-11
        self.dt = 300  # Time step (seconds) - 5 minutes
        
        # Moon parameters
        self.moon_mass = 7.342e22  # Real moon mass
        self.moon_radius = 1737000  # Real moon radius (m)
        self.moon_pos = np.array([0, 0], dtype=float)
        self.moon_vel = np.array([1000, 0], dtype=float)  # Moon moving right at 1 km/s
        
        # Spacecraft parameters - positioned to create proper slingshot
        self.spacecraft_mass = 1000
        # Position spacecraft to approach from behind and below at an angle
        self.spacecraft_pos = np.array([-1.5e8, -1.2e8], dtype=float)  # 150M km behind, 120M km below
        # Initial velocity aimed slightly ahead of moon's current position
        self.spacecraft_vel = np.array([2500, 1800], dtype=float)  # Moderate initial speed
        
        # Trajectory recording
        self.moon_trajectory = [self.moon_pos.copy()]
        self.spacecraft_trajectory = [self.spacecraft_pos.copy()]
        self.time_steps = [0]
        self.spacecraft_speeds = [np.linalg.norm(self.spacecraft_vel)]
        self.spacecraft_directions = [np.arctan2(self.spacecraft_vel[1], self.spacecraft_vel[0])]
        
        # Scale factor for display (100 million km = 1 unit)
        self.scale = 1e8
        
    def gravitational_force(self, pos1, pos2, mass1, mass2):
        """Calculate gravitational force between two objects"""
        r_vector = pos2 - pos1
        r_magnitude = np.linalg.norm(r_vector)
        
        # Prevent collision with moon surface
        min_distance = self.moon_radius * 1.1
        if r_magnitude < min_distance:
            r_magnitude = min_distance
            
        # Calculate gravitational force
        force_magnitude = self.G * mass1 * mass2 / (r_magnitude ** 2)
        force_direction = r_vector / r_magnitude
        
        return force_magnitude * force_direction
    
    def update_physics(self):
        """Update physics simulation"""
        # Calculate gravitational force on spacecraft
        force_on_spacecraft = self.gravitational_force(
            self.spacecraft_pos, self.moon_pos, 
            self.spacecraft_mass, self.moon_mass
        )
        
        # Update spacecraft motion
        spacecraft_acc = force_on_spacecraft / self.spacecraft_mass
        self.spacecraft_vel += spacecraft_acc * self.dt
        self.spacecraft_pos += self.spacecraft_vel * self.dt
        
        # Moon moves at constant velocity (simplified)
        self.moon_pos += self.moon_vel * self.dt
        
        # Record trajectory data
        self.moon_trajectory.append(self.moon_pos.copy())
        self.spacecraft_trajectory.append(self.spacecraft_pos.copy())
        self.time_steps.append(len(self.time_steps) * self.dt)
        self.spacecraft_speeds.append(np.linalg.norm(self.spacecraft_vel))
        self.spacecraft_directions.append(np.arctan2(self.spacecraft_vel[1], self.spacecraft_vel[0]))
    
    def run_simulation(self, max_duration=50000):
        """Run the physics simulation"""
        steps = int(max_duration / self.dt)
        
        print("Running gravity slingshot simulation...")
        
        for step in range(steps):
            self.update_physics()
            
            # Check stopping conditions
            distance = np.linalg.norm(self.spacecraft_pos - self.moon_pos)
            
            # Stop if spacecraft gets very far away
            if distance > 8e8:  # 800 million km
                print(f"Spacecraft far from moon. Stopping at step {step}")
                break
                
            # Progress update
            if step % 20 == 0:
                distance_km = distance / 1000
                speed_kms = self.spacecraft_speeds[-1] / 1000
                print(f"Step {step}/{steps}: Distance = {distance_km:.0f} km, Speed = {speed_kms:.1f} km/s")
                
        print(f"Simulation complete! Total steps: {len(self.spacecraft_trajectory)}")
    
    def analyze_results(self):
        """Analyze the slingshot maneuver results"""
        speeds = np.array(self.spacecraft_speeds)
        directions = np.array(self.spacecraft_directions)
        
        initial_speed = speeds[0]
        max_speed = np.max(speeds)
        final_speed = speeds[-1]
        
        initial_direction = directions[0] * 180 / np.pi
        final_direction = directions[-1] * 180 / np.pi
        direction_change = final_direction - initial_direction
        
        # Normalize direction change to [-180, 180]
        while direction_change > 180:
            direction_change -= 360
        while direction_change < -180:
            direction_change += 360
            
        speed_gain = final_speed - initial_speed
        speed_gain_percent = (speed_gain / initial_speed) * 100
        
        # Find closest approach
        distances = []
        for i in range(len(self.spacecraft_trajectory)):
            dist = np.linalg.norm(np.array(self.spacecraft_trajectory[i]) - np.array(self.moon_trajectory[i]))
            distances.append(dist)
        min_distance = min(distances) / 1000  # Convert to km
        
        print(f"\n=== Gravity Slingshot Analysis ===")
        print(f"Initial speed: {initial_speed/1000:.2f} km/s")
        print(f"Maximum speed: {max_speed/1000:.2f} km/s")
        print(f"Final speed: {final_speed/1000:.2f} km/s")
        print(f"Speed gain: {speed_gain:.0f} m/s ({speed_gain_percent:.1f}%)")
        print(f"Direction change: {direction_change:.1f} degrees")
        print(f"Closest approach: {min_distance:.0f} km")
        
        return {
            'initial_speed': initial_speed,
            'final_speed': final_speed,
            'speed_gain': speed_gain,
            'direction_change': direction_change,
            'closest_approach': min_distance
        }
    
    def create_animation(self):
        """Create the slingshot animation"""
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.patch.set_facecolor('black')
        
        # Convert trajectories to numpy arrays
        moon_traj = np.array(self.moon_trajectory) / self.scale
        spacecraft_traj = np.array(self.spacecraft_trajectory) / self.scale
        
        # Main trajectory plot
        ax1.set_facecolor('black')
        
        # Calculate plot boundaries
        all_x = np.concatenate([moon_traj[:, 0], spacecraft_traj[:, 0]])
        all_y = np.concatenate([moon_traj[:, 1], spacecraft_traj[:, 1]])
        
        margin = 0.5
        x_center = np.mean(all_x)
        y_center = np.mean(all_y)
        x_range = all_x.max() - all_x.min() + 2 * margin
        y_range = all_y.max() - all_y.min() + 2 * margin
        max_range = max(x_range, y_range)
        
        ax1.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax1.set_ylim(y_center - max_range/2, y_center + max_range/2)
        ax1.set_aspect('equal')
        ax1.set_title('Gravity Slingshot Maneuver', fontsize=18, color='white', pad=20)
        ax1.set_xlabel('Distance (100 Mm)', fontsize=14, color='white')
        ax1.set_ylabel('Distance (100 Mm)', fontsize=14, color='white')
        ax1.grid(True, alpha=0.2)
        
        # Speed analysis plot
        ax2.set_facecolor('black')
        ax2.set_title('Spacecraft Velocity Analysis', fontsize=18, color='white', pad=20)
        ax2.set_xlabel('Time (hours)', fontsize=14, color='white')
        ax2.set_ylabel('Speed (km/s)', fontsize=14, color='white')
        ax2.grid(True, alpha=0.3)
        
        # Create animated elements
        moon_size = 0.15
        moon_circle = Circle((0, 0), moon_size, color='lightgray', alpha=0.9, 
                           label='Moon', zorder=5)
        spacecraft_dot, = ax1.plot([], [], 'o', color='white', markersize=10, 
                                 label='Spacecraft', zorder=6)
        
        # Trajectory trails
        moon_trail, = ax1.plot([], [], color='gray', alpha=0.6, linewidth=2, 
                             label='Moon path', zorder=2)
        spacecraft_trail, = ax1.plot([], [], color='cyan', alpha=0.9, linewidth=3, 
                                   label='Spacecraft path', zorder=3)
        
        # Velocity vectors
        velocity_scale = 0.8
        moon_arrow = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle='->', color='yellow', 
                                              lw=3, alpha=0.8), zorder=4)
        spacecraft_arrow = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                                      arrowprops=dict(arrowstyle='->', color='lime', 
                                                    lw=3, alpha=0.9), zorder=4)
        
        # Speed plot elements
        speed_line, = ax2.plot([], [], color='cyan', linewidth=3, alpha=0.9)
        current_point, = ax2.plot([], [], 'o', color='red', markersize=8, zorder=5)
        
        ax1.add_patch(moon_circle)
        ax1.legend(loc='upper left', fontsize=12, framealpha=0.8)
        
        # Information panel
        info_text = ax1.text(0.02, 0.02, '', transform=ax1.transAxes, 
                           color='lime', fontsize=12, verticalalignment='bottom',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='black', 
                                   alpha=0.8, edgecolor='lime'))
        
        def animate(frame):
            if frame >= len(moon_traj):
                return []
                
            # Current positions
            moon_pos = moon_traj[frame]
            spacecraft_pos = spacecraft_traj[frame]
            
            # Update object positions
            moon_circle.center = moon_pos
            spacecraft_dot.set_data([spacecraft_pos[0]], [spacecraft_pos[1]])
            
            # Update trajectory trails
            trail_length = min(frame + 1, len(moon_traj))
            moon_trail.set_data(moon_traj[:trail_length, 0], 
                              moon_traj[:trail_length, 1])
            spacecraft_trail.set_data(spacecraft_traj[:trail_length, 0], 
                                    spacecraft_traj[:trail_length, 1])
            
            # Update velocity arrows
            if frame < len(moon_traj) - 1:
                # Moon velocity arrow (constant)
                moon_vel = np.array([1, 0]) * velocity_scale
                moon_arrow.set_position(moon_pos)
                moon_arrow.xy = moon_pos + moon_vel
                
                # Spacecraft velocity arrow
                if frame < len(spacecraft_traj) - 1:
                    dt_hours = self.dt / 3600
                    spacecraft_vel = (spacecraft_traj[frame+1] - spacecraft_traj[frame]) / dt_hours
                    vel_mag = np.linalg.norm(spacecraft_vel)
                    if vel_mag > 0:
                        vel_normalized = spacecraft_vel / vel_mag * velocity_scale
                        spacecraft_arrow.set_position(spacecraft_pos)
                        spacecraft_arrow.xy = spacecraft_pos + vel_normalized
            
            # Update speed plot
            times_hours = np.array(self.time_steps[:frame+1]) / 3600
            speeds_kms = np.array(self.spacecraft_speeds[:frame+1]) / 1000
            
            speed_line.set_data(times_hours, speeds_kms)
            if frame < len(speeds_kms):
                current_point.set_data([times_hours[frame]], [speeds_kms[frame]])
            
            ax2.relim()
            ax2.autoscale_view()
            
            # Update information display
            current_time = self.time_steps[frame] / 3600
            distance = np.linalg.norm((spacecraft_pos - moon_pos) * self.scale) / 1000
            current_speed = self.spacecraft_speeds[frame] / 1000
            speed_change = (self.spacecraft_speeds[frame] - self.spacecraft_speeds[0]) / 1000
            
            info_text.set_text(
                f'Time: {current_time:.1f} hours\n'
                f'Distance to Moon: {distance:.0f} km\n'
                f'Spacecraft Speed: {current_speed:.2f} km/s\n'
                f'Speed Change: {speed_change:+.2f} km/s\n'
                f'Frame: {frame+1}/{len(moon_traj)}'
            )
            
            return [moon_circle, spacecraft_dot, moon_trail, spacecraft_trail,
                   moon_arrow, spacecraft_arrow, speed_line, current_point, info_text]
        
        # Create animation with appropriate frame count
        total_frames = min(len(moon_traj), 600)
        interval = max(50, len(moon_traj) // 10)  # Adjust speed based on trajectory length
        
        anim = animation.FuncAnimation(
            fig, animate, frames=total_frames,
            interval=interval, blit=True, repeat=True
        )
        
        plt.tight_layout()
        return fig, anim

def main():
    print("🚀 Starting Realistic Gravity Slingshot Simulation...")
    
    # Create simulation
    sim = RealisticGravitySlingshot()
    
    # Run simulation
    sim.run_simulation(max_duration=40000)  # About 11 hours
    
    # Analyze results
    results = sim.analyze_results()
    
    # Create and display animation
    print("\n🎬 Creating animation...")
    fig, anim = sim.create_animation()
    
    print("🎥 Displaying animation...")
    plt.show()
    
    # Option to save
    save_choice = input("\n💾 Save animation as GIF? (y/n): ")
    if save_choice.lower() == 'y':
        print("💾 Saving animation...")
        try:
            anim.save('realistic_gravity_slingshot.gif', writer='pillow', fps=15,
                     savefig_kwargs={'facecolor': 'black'})
            print("✅ Animation saved as 'realistic_gravity_slingshot.gif'")
        except Exception as e:
            print(f"❌ Error saving animation: {e}")

if __name__ == "__main__":
    main() 