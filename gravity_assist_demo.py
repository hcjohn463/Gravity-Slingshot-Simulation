import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import math

class GravityAssistDemo:
    def __init__(self):
        # Physical constants
        self.G = 6.67430e-11
        self.dt = 100  # Time step (seconds)
        
        # Moon parameters
        self.moon_mass = 7.342e22 * 500  # Amplified for demonstration
        self.moon_radius = 1737000
        self.moon_pos = np.array([0, 0], dtype=float)
        self.moon_vel = np.array([1000, 0], dtype=float)  # Moving right (like in your image)
        
        # Spacecraft parameters - positioned to create slingshot effect
        self.spacecraft_mass = 1000
        self.spacecraft_pos = np.array([-1e8, -8e7], dtype=float)  # Approaching from bottom-left
        self.spacecraft_vel = np.array([1400, 900], dtype=float)  # Initial velocity toward moon
        
        # Trajectory recording
        self.moon_trajectory = [self.moon_pos.copy()]
        self.spacecraft_trajectory = [self.spacecraft_pos.copy()]
        self.time_steps = [0]
        self.spacecraft_speeds = [np.linalg.norm(self.spacecraft_vel)]
        self.distances = [np.linalg.norm(self.spacecraft_pos - self.moon_pos)]
        
        # Scale factor
        self.scale = 1e7
        
    def gravitational_force(self, pos1, pos2, mass1, mass2):
        """Calculate gravitational force"""
        r_vector = pos2 - pos1
        r_magnitude = np.linalg.norm(r_vector)
        
        # Prevent collision
        min_distance = self.moon_radius * 1.5
        if r_magnitude < min_distance:
            r_magnitude = min_distance
            
        force_magnitude = self.G * mass1 * mass2 / (r_magnitude ** 2)
        force_direction = r_vector / r_magnitude
        
        return force_magnitude * force_direction
    
    def update_physics(self):
        """Update physics"""
        # Calculate forces
        force_on_spacecraft = self.gravitational_force(
            self.spacecraft_pos, self.moon_pos, 
            self.spacecraft_mass, self.moon_mass
        )
        
        # Update spacecraft (moon moves at constant velocity for simplicity)
        spacecraft_acc = force_on_spacecraft / self.spacecraft_mass
        self.spacecraft_vel += spacecraft_acc * self.dt
        self.spacecraft_pos += self.spacecraft_vel * self.dt
        
        # Moon continues moving right
        self.moon_pos += self.moon_vel * self.dt
        
        # Record data
        self.moon_trajectory.append(self.moon_pos.copy())
        self.spacecraft_trajectory.append(self.spacecraft_pos.copy())
        self.time_steps.append(len(self.time_steps) * self.dt)
        self.spacecraft_speeds.append(np.linalg.norm(self.spacecraft_vel))
        self.distances.append(np.linalg.norm(self.spacecraft_pos - self.moon_pos))
    
    def run_simulation(self, duration=3000):
        """Run the simulation"""
        steps = int(duration / self.dt)
        
        for step in range(steps):
            self.update_physics()
            
            # Stop if spacecraft is far away
            distance = np.linalg.norm(self.spacecraft_pos - self.moon_pos)
            if distance > 4e8:
                break
                
            if step % 5 == 0:
                print(f"Progress: {step}/{steps} ({step/steps*100:.1f}%)")
    
    def analyze_trajectory(self):
        """Analyze the trajectory"""
        speeds = np.array(self.spacecraft_speeds)
        distances = np.array(self.distances)
        
        initial_speed = speeds[0]
        max_speed = np.max(speeds)
        final_speed = speeds[-1]
        min_distance = np.min(distances)
        
        speed_gain = final_speed - initial_speed
        energy_gain = 0.5 * self.spacecraft_mass * (final_speed**2 - initial_speed**2)
        
        print(f"\n=== Gravity Assist Analysis ===")
        print(f"Initial speed: {initial_speed:.0f} m/s")
        print(f"Maximum speed: {max_speed:.0f} m/s")
        print(f"Final speed: {final_speed:.0f} m/s")
        print(f"Speed gain: {speed_gain:.0f} m/s ({speed_gain/initial_speed*100:.1f}%)")
        print(f"Closest approach: {min_distance/1000:.0f} km")
        print(f"Energy gain: {energy_gain/1e6:.1f} MJ")
        
        return {
            'initial_speed': initial_speed,
            'max_speed': max_speed,
            'final_speed': final_speed,
            'speed_gain': speed_gain,
            'min_distance': min_distance
        }
    
    def create_animation(self):
        """Create the animation"""
        # Set up the figure with black background
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        fig.patch.set_facecolor('black')
        
        # Main trajectory plot
        ax1.set_facecolor('black')
        
        # Convert trajectories to arrays
        moon_traj = np.array(self.moon_trajectory)
        spacecraft_traj = np.array(self.spacecraft_trajectory)
        
        # Set up plot limits
        all_x = np.concatenate([moon_traj[:, 0], spacecraft_traj[:, 0]]) / self.scale
        all_y = np.concatenate([moon_traj[:, 1], spacecraft_traj[:, 1]]) / self.scale
        
        x_center = np.mean(all_x)
        y_center = np.mean(all_y)
        plot_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) * 0.6
        
        ax1.set_xlim(x_center - plot_range, x_center + plot_range)
        ax1.set_ylim(y_center - plot_range, y_center + plot_range)
        ax1.set_aspect('equal')
        ax1.set_title('Gravity Assist Maneuver', fontsize=16, color='white')
        ax1.set_xlabel('Distance (10 Mm)', fontsize=12, color='white')
        ax1.set_ylabel('Distance (10 Mm)', fontsize=12, color='white')
        
        # Speed analysis plot
        ax2.set_facecolor('black')
        ax2.set_title('Spacecraft Speed Profile', fontsize=16, color='white')
        ax2.set_xlabel('Time (hours)', fontsize=12, color='white')
        ax2.set_ylabel('Speed (km/s)', fontsize=12, color='white')
        ax2.grid(True, alpha=0.3)
        
        # Create plot elements
        moon_circle = Circle((0, 0), 0.4, color='lightgray', alpha=0.9, label='Moon')
        spacecraft_dot, = ax1.plot([], [], 'wo', markersize=8, label='Spacecraft')
        
        # Trajectory lines
        moon_trail, = ax1.plot([], [], 'gray', alpha=0.7, linewidth=2, label='Moon path')
        spacecraft_trail, = ax1.plot([], [], 'cyan', alpha=0.9, linewidth=3, label='Spacecraft path')
        
        # Velocity arrows
        moon_arrow = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle='->', color='white', lw=3))
        spacecraft_arrow = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                                      arrowprops=dict(arrowstyle='->', color='yellow', lw=3))
        
        # Speed plot elements
        speed_curve, = ax2.plot([], [], 'cyan', linewidth=3)
        current_speed_dot, = ax2.plot([], [], 'ro', markersize=8)
        
        ax1.add_patch(moon_circle)
        ax1.legend(loc='upper right', fontsize=10)
        
        # Information display
        info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                           color='lime', fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        def animate(frame):
            if frame >= len(self.moon_trajectory):
                return
                
            # Current positions
            moon_pos = moon_traj[frame] / self.scale
            spacecraft_pos = spacecraft_traj[frame] / self.scale
            
            # Update positions
            moon_circle.center = moon_pos
            spacecraft_dot.set_data([spacecraft_pos[0]], [spacecraft_pos[1]])
            
            # Update trajectory trails
            moon_trail.set_data(moon_traj[:frame+1, 0] / self.scale,
                              moon_traj[:frame+1, 1] / self.scale)
            spacecraft_trail.set_data(spacecraft_traj[:frame+1, 0] / self.scale,
                                    spacecraft_traj[:frame+1, 1] / self.scale)
            
            # Update velocity arrows
            if frame < len(self.moon_trajectory) - 1:
                # Moon velocity arrow
                moon_vel_display = np.array([1, 0]) * 2  # Fixed right direction
                moon_arrow.set_position(moon_pos)
                moon_arrow.xy = moon_pos + moon_vel_display
                
                # Spacecraft velocity arrow
                spacecraft_vel = (spacecraft_traj[frame+1] - spacecraft_traj[frame]) / self.scale
                vel_magnitude = np.linalg.norm(spacecraft_vel)
                if vel_magnitude > 0:
                    spacecraft_vel_display = spacecraft_vel / vel_magnitude * 2
                    spacecraft_arrow.set_position(spacecraft_pos)
                    spacecraft_arrow.xy = spacecraft_pos + spacecraft_vel_display
            
            # Update speed plot
            times_hours = np.array(self.time_steps[:frame+1]) / 3600
            speeds_kms = np.array(self.spacecraft_speeds[:frame+1]) / 1000
            
            speed_curve.set_data(times_hours, speeds_kms)
            if frame < len(speeds_kms):
                current_speed_dot.set_data([times_hours[frame]], [speeds_kms[frame]])
                
            ax2.relim()
            ax2.autoscale_view()
            
            # Update info text
            current_time = self.time_steps[frame] / 3600
            current_distance = self.distances[frame] / 1000
            current_speed = self.spacecraft_speeds[frame]
            speed_change = current_speed - self.spacecraft_speeds[0]
            
            info_text.set_text(
                f'Time: {current_time:.1f} hours\n'
                f'Distance: {current_distance:.0f} km\n'
                f'Speed: {current_speed:.0f} m/s\n'
                f'Speed Δ: {speed_change:+.0f} m/s\n'
                f'Frame: {frame+1}/{len(self.moon_trajectory)}'
            )
            
            return [moon_circle, spacecraft_dot, moon_trail, spacecraft_trail,
                   moon_arrow, spacecraft_arrow, speed_curve, current_speed_dot, info_text]
        
        # Create animation
        total_frames = min(len(self.moon_trajectory), 400)
        anim = animation.FuncAnimation(
            fig, animate, frames=total_frames,
            interval=80, blit=True, repeat=True
        )
        
        plt.tight_layout()
        return fig, anim

def main():
    print("🌙 Starting Gravity Assist Demonstration...")
    
    # Create and run simulation
    demo = GravityAssistDemo()
    
    print("🔄 Running simulation...")
    demo.run_simulation(duration=2500)
    
    print(f"✅ Simulation complete! {len(demo.moon_trajectory)} time steps")
    
    # Analyze results
    analysis = demo.analyze_trajectory()
    
    # Create and show animation
    print("🎬 Creating animation...")
    fig, anim = demo.create_animation()
    
    print("🎥 Showing animation...")
    plt.show()
    
    # Save option
    save = input("\n💾 Save animation? (y/n): ")
    if save.lower() == 'y':
        print("💾 Saving as gravity_assist_demo.gif...")
        try:
            anim.save('gravity_assist_demo.gif', writer='pillow', fps=12, 
                     savefig_kwargs={'facecolor': 'black'})
            print("✅ Animation saved!")
        except Exception as e:
            print(f"❌ Save error: {e}")

if __name__ == "__main__":
    main() 