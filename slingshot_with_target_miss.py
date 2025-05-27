# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
import os
from PIL import Image

class SlingshotWithTarget:
    def __init__(self):
        # Enhanced physics constants for dramatic effect
        self.G = 6.67430e-11
        self.dt = 200  # Time step (seconds)
        
        # Enhanced mode - dramatic effect with clear U-turn
        self.moon_mass = 7.342e22 * 50  # 50x real moon mass for dramatic effect
        self.moon_radius = 1737000  # Real moon radius
        self.moon_pos = np.array([0, 0], dtype=float)
        self.moon_vel = np.array([1500, 0], dtype=float)  # 1.5 km/s rightward
        
        # Spacecraft parameters - optimized for dramatic slingshot from behind
        self.spacecraft_mass = 1000
        self.spacecraft_pos = np.array([3e7, -2e7], dtype=float)  # 30M km ahead, 20M km below (right-bottom)
        self.spacecraft_vel = np.array([-2600, 1900], dtype=float)  # Flying left-up to catch moon from behind
        
        # Target zone in the upper right area
        self.target_center = np.array([4e7, 3e7], dtype=float)  # 40M km right, 30M km up
        self.target_radius = 4e6  # 4M km radius
        self.target_hit = False
        self.target_hit_time = None
        
        # Load custom images if available
        self.moon_image = self.load_image('moon.png', 'moon.jpg', 'moon.jpeg')
        self.spacecraft_image = self.load_image('spacecraft.png', 'spacecraft.jpg', 'spacecraft.jpeg', 'spaceship.png')
        self.flag_image = self.load_image('flag.png', 'flag.jpg', 'flag.jpeg')
        
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
        print(f"🎯 Slingshot with Target Challenge")
        print(f"🌙 Enhanced Gravity Slingshot (Moon mass x50)")
        print(f"Escape velocity at initial distance ({initial_distance/1000:.0f} km): {self.escape_velocity/1000:.2f} km/s")
        print(f"Initial speed: {np.linalg.norm(self.spacecraft_vel)/1000:.2f} km/s")
        print(f"Sufficient for escape: {'Yes' if np.linalg.norm(self.spacecraft_vel) > self.escape_velocity else 'No'}")
        print(f"🎯 Target location: ({self.target_center[0]/1000:.0f}, {self.target_center[1]/1000:.0f}) km")
        print(f"🎯 Target radius: {self.target_radius/1000:.0f} km")
        
        # Print image status
        if self.moon_image is not None:
            print("✅ Moon image loaded successfully")
        else:
            print("ℹ️ Using default moon circle (no moon image found)")
            
        if self.spacecraft_image is not None:
            print("✅ Spacecraft image loaded successfully")
        else:
            print("ℹ️ Using default spacecraft dot (no spacecraft image found)")
            
        if self.flag_image is not None:
            print("✅ Flag image loaded successfully")
        else:
            print("ℹ️ Using default target circle (no flag image found)")
    
    def load_image(self, *filenames):
        """Try to load image from multiple possible filenames"""
        for filename in filenames:
            if os.path.exists(filename):
                try:
                    img = Image.open(filename)
                    # Convert to RGBA if not already
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    return np.array(img)
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")
                    continue
        return None
    
    def calculate_escape_velocity(self, distance):
        """Calculate escape velocity at given distance"""
        return np.sqrt(2 * self.G * self.moon_mass / distance)
    
    def check_target_hit(self):
        """Check if spacecraft has hit the target"""
        if self.target_hit:
            return  # Already hit
            
        distance_to_target = np.linalg.norm(self.spacecraft_pos - self.target_center)
        if distance_to_target <= self.target_radius:
            self.target_hit = True
            self.target_hit_time = len(self.time_steps) - 1
            print(f"🎯 TARGET HIT! Time: {self.time_steps[-1]/3600:.1f} hours")
    
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
        
        # Check target hit
        self.check_target_hit()
        
        # Record data
        self.moon_trajectory.append(self.moon_pos.copy())
        self.spacecraft_trajectory.append(self.spacecraft_pos.copy())
        self.time_steps.append(len(self.time_steps) * self.dt)
        self.spacecraft_speeds.append(np.linalg.norm(self.spacecraft_vel))
        self.spacecraft_directions.append(np.arctan2(self.spacecraft_vel[1], self.spacecraft_vel[0]))
    
    def run_simulation(self, max_steps=100):
        """Run simulation with fixed step limit"""
        print("Running slingshot with target simulation...")
        
        for step in range(max_steps):
            self.update_physics()
            
            # Progress reporting
            if step % 20 == 0:
                distance = np.linalg.norm(self.spacecraft_pos - self.moon_pos)
                dist_km = distance / 1000
                speed_kms = self.spacecraft_speeds[-1] / 1000
                target_dist = np.linalg.norm(self.spacecraft_pos - self.target_center) / 1000
                print(f"Step {step}: Distance = {dist_km:.0f} km, Speed = {speed_kms:.2f} km/s, Target dist = {target_dist:.0f} km")
        
        print(f"Simulation finished! Total steps: {len(self.spacecraft_trajectory)}")
        if self.target_hit:
            print(f"🎯 Target was hit during simulation!")
        else:
            print(f"❌ Target was not hit during simulation")
    
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
        
        # Find closest approach to moon
        min_distance = float('inf')
        for i in range(len(self.spacecraft_trajectory)):
            dist = np.linalg.norm(
                np.array(self.spacecraft_trajectory[i]) - 
                np.array(self.moon_trajectory[i])
            )
            min_distance = min(min_distance, dist)
        
        # Find closest approach to target
        min_target_distance = float('inf')
        for pos in self.spacecraft_trajectory:
            dist = np.linalg.norm(np.array(pos) - self.target_center)
            min_target_distance = min(min_target_distance, dist)
        
        print(f"\n=== Slingshot with Target Analysis ===")
        print(f"Initial speed: {initial_speed/1000:.2f} km/s")
        print(f"Maximum speed: {max_speed/1000:.2f} km/s")
        print(f"Final speed: {final_speed/1000:.2f} km/s")
        print(f"Speed gain: {speed_gain:.0f} m/s ({speed_percentage:.1f}%)")
        print(f"Direction change: {direction_change:.1f}°")
        print(f"Closest approach to moon: {min_distance/1000:.0f} km")
        print(f"Closest approach to target: {min_target_distance/1000:.0f} km")
        print(f"Target hit: {'✅ YES' if self.target_hit else '❌ NO'}")
        if self.target_hit:
            print(f"Target hit time: {self.time_steps[self.target_hit_time]/3600:.1f} hours")
        
        return {
            'speed_gain': speed_gain,
            'direction_change': direction_change,
            'closest_approach': min_distance/1000,
            'target_hit': self.target_hit,
            'closest_target_approach': min_target_distance/1000
        }
    
    def create_animation(self):
        """Create visualization"""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        fig.patch.set_facecolor('black')
        
        # Convert to display coordinates
        moon_traj = np.array(self.moon_trajectory) / self.scale
        spacecraft_traj = np.array(self.spacecraft_trajectory) / self.scale
        target_center_display = self.target_center / self.scale
        target_radius_display = self.target_radius / self.scale
        
        # Setup main plot
        ax.set_facecolor('black')
        
        # Calculate bounds including target
        all_x = np.concatenate([moon_traj[:, 0], spacecraft_traj[:, 0], [target_center_display[0]]])
        all_y = np.concatenate([moon_traj[:, 1], spacecraft_traj[:, 1], [target_center_display[1]]])
        
        margin = 1.0
        x_center = np.mean(all_x)
        y_center = np.mean(all_y)
        plot_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) + 2*margin
        
        ax.set_xlim(x_center - plot_range/2, x_center + plot_range/2)
        ax.set_ylim(y_center - plot_range/2, y_center + plot_range/2)
        ax.set_aspect('equal')
        
        # Create mixed color title
        result_text = "Success" if self.target_hit else "Fail"
        title_color = 'lime' if self.target_hit else 'red'
        
        # Calculate the full title for positioning
        full_title = f'Slingshot with Target Challenge - {result_text}'
        
        # Main title text in white (center aligned)
        ax.text(0.5, 1.02, 'Slingshot with Target Challenge - ', 
                transform=ax.transAxes, fontsize=16, weight='bold',
                horizontalalignment='center', verticalalignment='bottom',
                color='white')
        
        # Calculate position for the result text to align properly
        # Position it right after the dash
        ax.text(0.71, 1.02, result_text, 
                transform=ax.transAxes, fontsize=16, weight='bold',
                horizontalalignment='left', verticalalignment='bottom',
                color=title_color)
        
        ax.set_xlabel('Distance (10 Million km)', fontsize=14, color='white')
        ax.set_ylabel('Distance (10 Million km)', fontsize=14, color='white')
        ax.grid(True, alpha=0.3, color='gray')
        
        # Add target marker (flag or cross) - no circle background
        if self.flag_image is not None:
            # Use flag image
            flag_img = OffsetImage(self.flag_image, zoom=0.08)
            flag_visual = AnnotationBbox(flag_img, target_center_display, frameon=False, zorder=15)
            ax.add_artist(flag_visual)
            target_color = 'lime' if self.target_hit else 'orange'  # Color for legend only
        else:
            # Use default cross marker with target zone circle
            target_color = 'lime' if self.target_hit else 'red'
            target_alpha = 0.8 if self.target_hit else 0.5
            target_circle = Circle(target_center_display, target_radius_display, 
                                 color=target_color, alpha=target_alpha, 
                                 label='Target Zone', zorder=3)
            ax.add_patch(target_circle)
            ax.plot(target_center_display[0], target_center_display[1], 
                    'x', color='white', markersize=15, markeredgewidth=3, zorder=15)
        
        # Create visual elements
        moon_size = 0.25
        
        # Moon visual element
        if self.moon_image is not None:
            # Use custom moon image
            moon_img = OffsetImage(self.moon_image, zoom=0.06)
            moon_visual = AnnotationBbox(moon_img, (0, 0), frameon=False, zorder=10)
            ax.add_artist(moon_visual)
            moon_circle = None  # No circle needed
        else:
            # Use default circle
            moon_circle = Circle((0, 0), moon_size, color='lightgray', alpha=0.9, 
                               label='Moon', zorder=10)
            ax.add_patch(moon_circle)
            moon_visual = moon_circle
        
        # Spacecraft visual element
        if self.spacecraft_image is not None:
            # Use custom spacecraft image
            spacecraft_img = OffsetImage(self.spacecraft_image, zoom=0.05)
            spacecraft_visual = AnnotationBbox(spacecraft_img, (0, 0), frameon=False, zorder=11)
            ax.add_artist(spacecraft_visual)
            spacecraft_dot = None  # No dot needed
        else:
            # Use default dot
            spacecraft_dot, = ax.plot([], [], 'o', color='white', markersize=12, 
                                     label='Spacecraft', zorder=11)
            spacecraft_visual = spacecraft_dot
        
        # Path trails
        moon_path, = ax.plot([], [], color='gray', alpha=0.7, linewidth=3, 
                            label='Moon Path', zorder=5)
        spacecraft_path, = ax.plot([], [], color='cyan', alpha=1.0, linewidth=4, 
                                  label='Spacecraft Path', zorder=6)
        
        # Velocity indicators
        arrow_scale = 1.2
        moon_velocity = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                   arrowprops=dict(arrowstyle='->', color='yellow', 
                                                 lw=4, alpha=0.9), zorder=8)
        spacecraft_velocity = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                         arrowprops=dict(arrowstyle='->', color='lime', 
                                                       lw=4, alpha=1.0), zorder=9)
        
        # Status display
        status_display = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                                color='lime', fontsize=14, weight='bold',
                                verticalalignment='top',
                                bbox=dict(boxstyle="round,pad=0.6", facecolor='black', 
                                        alpha=0.8, edgecolor='lime', linewidth=2))
        
        # Target status display
        target_status = ax.text(0.02, 0.02, '', transform=ax.transAxes, 
                               color='white', fontsize=16, weight='bold',
                               verticalalignment='bottom',
                               bbox=dict(boxstyle="round,pad=0.6", facecolor='red', 
                                       alpha=0.8, edgecolor='white', linewidth=2))
        
        def animate(frame):
            if frame >= len(moon_traj):
                return []
            
            # Current positions
            moon_x, moon_y = moon_traj[frame]
            spacecraft_x, spacecraft_y = spacecraft_traj[frame]
            
            # Update positions
            if self.moon_image is not None:
                moon_visual.xybox = (moon_x, moon_y)
            else:
                moon_circle.center = (moon_x, moon_y)
                
            if self.spacecraft_image is not None:
                spacecraft_visual.xybox = (spacecraft_x, spacecraft_y)
            else:
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
            
            # Update status
            current_time = self.time_steps[frame] / 3600
            distance_km = np.linalg.norm((np.array([spacecraft_x, spacecraft_y]) - 
                                        np.array([moon_x, moon_y])) * self.scale) / 1000
            current_speed_kms = self.spacecraft_speeds[frame] / 1000
            speed_delta = (self.spacecraft_speeds[frame] - self.spacecraft_speeds[0]) / 1000
            
            # Distance to target
            target_distance = np.linalg.norm((np.array([spacecraft_x, spacecraft_y]) * self.scale - 
                                            self.target_center)) / 1000
            
            status_display.set_text(
                f'Time: {current_time:.1f} hrs\n'
                f'Distance: {distance_km:.0f} km\n'
                f'Speed: {current_speed_kms:.2f} km/s\n'
                f'Δ Speed: {speed_delta:+.2f} km/s\n'
                f'Target dist: {target_distance:.0f} km\n'
                f'Frame: {frame+1}/{len(moon_traj)}'
            )
            
            # Update target status
            if self.target_hit and frame >= self.target_hit_time:
                target_status.set_text('🎯 TARGET HIT! 🎯')
                target_status.get_bbox_patch().set_facecolor('lime')
            else:
                target_status.set_text('🎯 TARGET ZONE 🎯')
                target_status.get_bbox_patch().set_facecolor('red')
            
            # Return all animated elements
            animated_elements = [moon_path, spacecraft_path, moon_velocity, spacecraft_velocity, 
                               status_display, target_status]
            
            if self.moon_image is not None:
                animated_elements.append(moon_visual)
            else:
                animated_elements.append(moon_circle)
                
            if self.spacecraft_image is not None:
                animated_elements.append(spacecraft_visual)
            else:
                animated_elements.append(spacecraft_dot)
                
            return animated_elements
        
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
    print("🎯 Slingshot Target Challenge")
    print("=" * 50)
    print("Can the spacecraft hit the target using gravity assist?")
    
    # Initialize simulation
    sim = SlingshotWithTarget()
    
    print(f"\n{'='*60}")
    print("Mission Objective:")
    print("• Use gravity slingshot to reach the target zone")
    print("• Target is located in the upper-right area")
    print("• Red zone = missed target, Green zone = target hit!")
    print("• Watch the spacecraft trajectory carefully!")
    print(f"{'='*60}")
    
    # Run simulation
    sim.run_simulation(max_steps=330)
    
    # Analyze results
    results = sim.analyze_slingshot()
    
    # Analyze escape success
    final_distance = np.linalg.norm(
        np.array(sim.spacecraft_trajectory[-1]) - 
        np.array(sim.moon_trajectory[-1])
    )
    final_speed = sim.spacecraft_speeds[-1]
    escape_vel_at_final = sim.calculate_escape_velocity(final_distance)
    
    print(f"\n=== Mission Results ===")
    print(f"Final distance from moon: {final_distance/1000:.0f} km")
    print(f"Final speed: {final_speed/1000:.2f} km/s") 
    print(f"Escape velocity at final distance: {escape_vel_at_final/1000:.2f} km/s")
    print(f"Successfully escaped: {'Yes' if final_speed > escape_vel_at_final else 'No'}")
    
    if sim.target_hit:
        print("🎉 MISSION SUCCESS! Target achieved!")
        print("✅ The spacecraft successfully used gravity assist to reach the target!")
    else:
        print("❌ Mission failed - Target missed")
        print(f"💡 Closest approach to target: {results['closest_target_approach']:.0f} km")
        print("💡 Try adjusting initial position or velocity for better trajectory")
    
    # Create visualization
    print("\n🎬 Generating mission animation...")
    fig, anim = sim.create_animation()
    
    print("🎥 Displaying slingshot target challenge...")
    plt.show()
    
    # Save option
    save_option = input("\n💾 Save animation as GIF? (y/n): ")
    if save_option.lower() == 'y':
        print("💾 Saving animation...")
        try:
            filename = 'slingshot_target_hit.gif' if sim.target_hit else 'slingshot_target_miss.gif'
            anim.save(filename, writer='pillow', fps=12,
                     savefig_kwargs={'facecolor': 'black'})
            print(f"✅ Animation saved as '{filename}'")
        except Exception as e:
            print(f"❌ Save error: {e}")
    
    print("\n🎯 Target challenge complete!")
    return sim, results

if __name__ == "__main__":
    main() 