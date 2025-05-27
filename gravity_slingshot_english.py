import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import math

class GravitySlingshot:
    def __init__(self):
        # Physical constants (adjusted for better visualization)
        self.G = 6.67430e-11 * 1e6  # Amplified gravity constant for demonstration
        self.dt = 50  # Time step (seconds)
        
        # Moon parameters
        self.moon_mass = 7.342e22  # Moon mass (kg)
        self.moon_radius = 1737000  # Moon radius (m)
        self.moon_pos = np.array([0, 0], dtype=float)  # Moon initial position (m)
        self.moon_vel = np.array([800, 0], dtype=float)  # Moon velocity (m/s) - moving right
        
        # Spacecraft parameters - approaching from bottom-left
        self.spacecraft_mass = 1000  # Spacecraft mass (kg)
        self.spacecraft_pos = np.array([-8e7, -6e7], dtype=float)  # Spacecraft initial position
        self.spacecraft_vel = np.array([1200, 800], dtype=float)  # Spacecraft initial velocity
        
        # Trajectory recording
        self.moon_trajectory = [self.moon_pos.copy()]
        self.spacecraft_trajectory = [self.spacecraft_pos.copy()]
        self.time_steps = [0]
        self.spacecraft_speeds = [np.linalg.norm(self.spacecraft_vel)]
        
        # Scale factor (for display)
        self.scale = 1e7
        
    def gravitational_force(self, pos1, pos2, mass1, mass2):
        """Calculate gravitational force between two objects"""
        r_vector = pos2 - pos1
        r_magnitude = np.linalg.norm(r_vector)
        
        # Avoid division by zero
        min_distance = self.moon_radius * 2
        if r_magnitude < min_distance:
            r_magnitude = min_distance
            
        # Gravitational force magnitude
        force_magnitude = self.G * mass1 * mass2 / (r_magnitude ** 2)
        
        # Force direction
        force_direction = r_vector / r_magnitude
        
        return force_magnitude * force_direction
    
    def update_physics(self):
        """Update physics state"""
        # Calculate gravitational force on spacecraft from moon
        force_on_spacecraft = self.gravitational_force(
            self.spacecraft_pos, self.moon_pos, 
            self.spacecraft_mass, self.moon_mass
        )
        
        # Moon is much more massive than spacecraft, so moon is barely affected
        # But for physical accuracy, we still calculate it
        force_on_moon = -force_on_spacecraft
        
        # Update acceleration
        spacecraft_acc = force_on_spacecraft / self.spacecraft_mass
        moon_acc = force_on_moon / self.moon_mass
        
        # Update velocity
        self.spacecraft_vel += spacecraft_acc * self.dt
        self.moon_vel += moon_acc * self.dt
        
        # Update position
        self.spacecraft_pos += self.spacecraft_vel * self.dt
        self.moon_pos += self.moon_vel * self.dt
        
        # Record trajectory
        self.moon_trajectory.append(self.moon_pos.copy())
        self.spacecraft_trajectory.append(self.spacecraft_pos.copy())
        self.time_steps.append(len(self.time_steps) * self.dt)
        self.spacecraft_speeds.append(np.linalg.norm(self.spacecraft_vel))
    
    def run_simulation(self, duration=5000):
        """Run simulation"""
        steps = int(duration / self.dt)
        
        for step in range(steps):
            self.update_physics()
            
            # Stop if spacecraft gets too far away
            distance = np.linalg.norm(self.spacecraft_pos - self.moon_pos)
            if distance > 3e8:  # 300 million meters
                break
                
            # Progress display
            if step % 10 == 0:
                print(f"Simulation progress: {step}/{steps} ({step/steps*100:.1f}%)")
    
    def analyze_trajectory(self):
        """Analyze trajectory data"""
        speeds = np.array(self.spacecraft_speeds)
        initial_speed = speeds[0]
        max_speed = np.max(speeds)
        final_speed = speeds[-1]
        
        speed_gain = final_speed - initial_speed
        
        print(f"\n=== Trajectory Analysis ===")
        print(f"Initial speed: {initial_speed:.2f} m/s")
        print(f"Maximum speed: {max_speed:.2f} m/s")
        print(f"Final speed: {final_speed:.2f} m/s")
        print(f"Speed gain: {speed_gain:.2f} m/s ({speed_gain/initial_speed*100:.1f}%)")
        
        return {
            'initial_speed': initial_speed,
            'max_speed': max_speed,
            'final_speed': final_speed,
            'speed_gain': speed_gain
        }
    
    def create_animation(self):
        """Create animation"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor('black')
        
        # Main animation panel
        ax1.set_facecolor('black')
        
        # Setup coordinate system
        moon_traj = np.array(self.moon_trajectory)
        spacecraft_traj = np.array(self.spacecraft_trajectory)
        
        # Find appropriate display range
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
        ax1.set_title('Gravity Slingshot Simulation', color='white', fontsize=14)
        ax1.set_xlabel('Distance (10 Mm)', color='white')
        ax1.set_ylabel('Distance (10 Mm)', color='white')
        ax1.tick_params(colors='white')
        
        # Speed plot panel
        ax2.set_facecolor('black')
        ax2.set_title('Spacecraft Speed Change', color='white', fontsize=14)
        ax2.set_xlabel('Time (seconds)', color='white')
        ax2.set_ylabel('Speed (m/s)', color='white')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.3)
        
        # Create graphic elements
        moon_circle = Circle((0, 0), 0.3, color='lightgray', label='Moon', alpha=0.8)
        spacecraft_point, = ax1.plot([], [], 'wo', markersize=6, label='Spacecraft')
        moon_trail, = ax1.plot([], [], 'gray', alpha=0.6, linewidth=1.5, label='Moon trajectory')
        spacecraft_trail, = ax1.plot([], [], 'cyan', alpha=0.8, linewidth=2, label='Spacecraft trajectory')
        
        # Velocity vectors
        moon_vel_arrow = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                                    arrowprops=dict(arrowstyle='->', 
                                                  color='white', lw=2))
        spacecraft_vel_arrow = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                                          arrowprops=dict(arrowstyle='->', 
                                                        color='yellow', lw=2))
        
        # Speed curve
        speed_line, = ax2.plot([], [], 'cyan', linewidth=2)
        speed_point, = ax2.plot([], [], 'ro', markersize=5)
        
        ax1.add_patch(moon_circle)
        ax1.legend(loc='upper left')
        
        # Add information text
        info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                           color='white', verticalalignment='top',
                           fontfamily='monospace', fontsize=10)
        
        def animate(frame):
            if frame >= len(self.moon_trajectory):
                return
                
            # Current positions
            moon_pos = moon_traj[frame] / self.scale
            spacecraft_pos = spacecraft_traj[frame] / self.scale
            
            # Update positions
            moon_circle.center = (moon_pos[0], moon_pos[1])
            spacecraft_point.set_data([spacecraft_pos[0]], [spacecraft_pos[1]])
            
            # Update trajectories
            if frame > 0:
                moon_trail.set_data(moon_traj[:frame, 0] / self.scale,
                                  moon_traj[:frame, 1] / self.scale)
                spacecraft_trail.set_data(spacecraft_traj[:frame, 0] / self.scale,
                                        spacecraft_traj[:frame, 1] / self.scale)
            
            # Calculate and display velocity vectors
            if frame < len(self.moon_trajectory) - 1:
                moon_vel = (moon_traj[frame+1] - moon_traj[frame]) / (self.dt * self.scale)
                spacecraft_vel = (spacecraft_traj[frame+1] - spacecraft_traj[frame]) / (self.dt * self.scale)
                
                # Scale velocity vectors for display
                vel_scale = 0.3
                moon_vel_end = moon_pos + moon_vel * vel_scale
                spacecraft_vel_end = spacecraft_pos + spacecraft_vel * vel_scale
                
                moon_vel_arrow.set_position(moon_pos)
                moon_vel_arrow.xy = moon_vel_end
                
                spacecraft_vel_arrow.set_position(spacecraft_pos)
                spacecraft_vel_arrow.xy = spacecraft_vel_end
            
            # Update speed plot
            times = np.array(self.time_steps[:frame+1])
            speeds = np.array(self.spacecraft_speeds[:frame+1])
            
            speed_line.set_data(times, speeds)
            if frame < len(speeds):
                speed_point.set_data([times[frame]], [speeds[frame]])
            
            ax2.relim()
            ax2.autoscale_view()
            
            # Update information
            distance = np.linalg.norm(spacecraft_traj[frame] - moon_traj[frame]) / 1000000
            time = self.time_steps[frame] / 3600  # Convert to hours
            current_speed = self.spacecraft_speeds[frame]
            speed_change = current_speed - self.spacecraft_speeds[0]
            
            info_text.set_text(f'Time: {time:.2f} hours\n'
                             f'Distance: {distance:.1f} thousand km\n'
                             f'Current speed: {current_speed:.0f} m/s\n'
                             f'Speed change: {speed_change:+.0f} m/s\n'
                             f'Frame: {frame}/{len(self.moon_trajectory)-1}')
            
            return [moon_circle, spacecraft_point, moon_trail, spacecraft_trail, 
                   moon_vel_arrow, spacecraft_vel_arrow, speed_line, speed_point, info_text]
        
        # Create animation
        frames = min(len(self.moon_trajectory), 500)  # Limit frames
        anim = animation.FuncAnimation(fig, animate, frames=frames,
                                     interval=100, blit=True, repeat=True)
        
        plt.tight_layout()
        return fig, anim

def main():
    print("🚀 Starting Gravity Slingshot Simulation...")
    
    # Create simulation instance
    sim = GravitySlingshot()
    
    # Run physics simulation
    print("⚡ Running physics calculations...")
    sim.run_simulation(duration=4000)
    
    print(f"✅ Simulation complete! Total {len(sim.moon_trajectory)} time steps")
    
    # Analyze results
    analysis = sim.analyze_trajectory()
    
    # Create animation
    print("🎬 Creating animation...")
    fig, anim = sim.create_animation()
    
    # Show animation
    print("🎥 Displaying animation...")
    plt.show()
    
    # Optional: Save as GIF
    save_choice = input("\n💾 Save animation as GIF? (y/n): ")
    if save_choice.lower() == 'y':
        print("📁 Saving animation as gravity_slingshot_english.gif...")
        try:
            anim.save('gravity_slingshot_english.gif', writer='pillow', fps=10)
            print("✅ Animation saved successfully!")
        except Exception as e:
            print(f"❌ Save failed: {e}")

if __name__ == "__main__":
    main() 