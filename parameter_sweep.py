# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import os
from PIL import Image

class SlingshotParameterSweep:
    def __init__(self):
        # Enhanced physics constants for dramatic effect
        self.G = 6.67430e-11
        self.dt = 200  # Time step (seconds)
        
        # Enhanced mode - dramatic effect with clear U-turn
        self.moon_mass = 7.342e22 * 50  # 50x real moon mass for dramatic effect
        self.moon_radius = 1737000  # Real moon radius
        
        # Fixed parameters (same for all tests)
        self.spacecraft_mass = 1000
        self.spacecraft_pos = np.array([3e7, -2e7], dtype=float)  # 30M km ahead, 20M km below
        
        # Target zone in the upper right area
        self.target_center = np.array([4e7, 3e7], dtype=float)  # 40M km right, 30M km up
        self.target_radius = 1e5  # 100k km radius - small target zone
        
        # Results storage
        self.successful_params = []
        self.all_results = []
        
    def calculate_escape_velocity(self, distance):
        """Calculate escape velocity at given distance"""
        return np.sqrt(2 * self.G * self.moon_mass / distance)
    
    def check_target_hit(self, spacecraft_pos):
        """Check if spacecraft has hit the target"""
        distance_to_target = np.linalg.norm(spacecraft_pos - self.target_center)
        return distance_to_target <= self.target_radius
    
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
    
    def run_single_simulation(self, spacecraft_vel, max_steps=100):
        """Run a single simulation with given initial velocity"""
        # Initialize positions and velocities
        moon_pos = np.array([0, 0], dtype=float)
        moon_vel = np.array([1500, 0], dtype=float)  # 1.5 km/s rightward
        spacecraft_pos = self.spacecraft_pos.copy()
        spacecraft_vel = np.array(spacecraft_vel, dtype=float)
        
        target_hit = False
        target_hit_time = None
        
        # Run simulation
        for step in range(max_steps):
            # Calculate gravitational force
            force = self.gravitational_force(
                spacecraft_pos, moon_pos,
                self.spacecraft_mass, self.moon_mass
            )
            
            # Update spacecraft
            acceleration = force / self.spacecraft_mass
            spacecraft_vel += acceleration * self.dt
            spacecraft_pos += spacecraft_vel * self.dt
            
            # Update moon (constant velocity)
            moon_pos += moon_vel * self.dt
            
            # Check target hit
            if self.check_target_hit(spacecraft_pos):
                target_hit = True
                target_hit_time = step
                break
        
        return {
            'target_hit': target_hit,
            'target_hit_time': target_hit_time,
            'final_pos': spacecraft_pos.copy(),
            'final_distance_to_target': np.linalg.norm(spacecraft_pos - self.target_center)
        }
    
    def parameter_sweep(self):
        """Sweep through parameter space"""
        print("🔍 Starting parameter sweep...")
        print("Testing spacecraft velocity ranges:")
        print("X: -4000 to -1000 m/s (step: 10)")
        print("Y: 1000 to 4000 m/s (step: 10)")
        print()
        
        # Define parameter ranges
        x_velocities = np.arange(-4000, -1000 + 10, 10)  # -4000 to -1000, step 10
        y_velocities = np.arange(1000, 4000 + 10, 10)    # 1000 to 4000, step 10
        
        total_tests = len(x_velocities) * len(y_velocities)
        print(f"Total parameter combinations to test: {total_tests}")
        print(f"X velocities: {len(x_velocities)} points")
        print(f"Y velocities: {len(y_velocities)} points")
        print()
        
        successful_count = 0
        test_count = 0
        
        # Progress tracking
        progress_interval = max(1, total_tests // 20)  # Show progress every 5%
        
        for i, vx in enumerate(x_velocities):
            for j, vy in enumerate(y_velocities):
                test_count += 1
                
                # Run simulation
                result = self.run_single_simulation([vx, vy])
                
                # Store result
                result_data = {
                    'vx': vx,
                    'vy': vy,
                    'speed': np.sqrt(vx**2 + vy**2),
                    'target_hit': result['target_hit'],
                    'target_hit_time': result['target_hit_time'],
                    'final_distance': result['final_distance_to_target'] / 1000  # km
                }
                
                self.all_results.append(result_data)
                
                if result['target_hit']:
                    successful_count += 1
                    self.successful_params.append(result_data)
                
                # Progress reporting
                if test_count % progress_interval == 0 or test_count == total_tests:
                    progress = (test_count / total_tests) * 100
                    print(f"Progress: {progress:.1f}% ({test_count}/{total_tests}) - "
                          f"Successful: {successful_count}")
        
        print(f"\n✅ Parameter sweep completed!")
        print(f"Total tests: {total_tests}")
        print(f"Successful parameters: {successful_count}")
        print(f"Success rate: {(successful_count/total_tests)*100:.2f}%")
        
        return self.successful_params, self.all_results
    
    def analyze_results(self):
        """Analyze the sweep results"""
        if not self.successful_params:
            print("❌ No successful parameters found!")
            return
        
        print(f"\n📊 Analysis of {len(self.successful_params)} successful parameters:")
        print("=" * 60)
        
        # Convert to arrays for analysis
        vx_success = [p['vx'] for p in self.successful_params]
        vy_success = [p['vy'] for p in self.successful_params]
        speeds_success = [p['speed'] for p in self.successful_params]
        hit_times = [p['target_hit_time'] for p in self.successful_params]
        
        print(f"X velocity range: {min(vx_success)} to {max(vx_success)} m/s")
        print(f"Y velocity range: {min(vy_success)} to {max(vy_success)} m/s")
        print(f"Speed range: {min(speeds_success):.0f} to {max(speeds_success):.0f} m/s")
        print(f"Hit time range: {min(hit_times)} to {max(hit_times)} steps")
        print()
        
        # Show first 10 successful parameters
        print("🎯 First 10 successful parameters:")
        print("Vx (m/s) | Vy (m/s) | Speed (m/s) | Hit Time (steps)")
        print("-" * 55)
        for i, param in enumerate(self.successful_params[:10]):
            print(f"{param['vx']:8.0f} | {param['vy']:8.0f} | {param['speed']:11.0f} | {param['target_hit_time']:13}")
        
        if len(self.successful_params) > 10:
            print(f"... and {len(self.successful_params) - 10} more")
        
        return vx_success, vy_success, speeds_success
    
    def create_visualization(self):
        """Create visualization of parameter space"""
        if not self.all_results:
            print("No results to visualize!")
            return
        
        print("\n🎨 Creating parameter space visualization...")
        
        # Extract data
        vx_all = [r['vx'] for r in self.all_results]
        vy_all = [r['vy'] for r in self.all_results]
        success_all = [r['target_hit'] for r in self.all_results]
        
        # Create figure
        plt.style.use('dark_background')
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Plot all points
        successful_vx = [vx_all[i] for i in range(len(vx_all)) if success_all[i]]
        successful_vy = [vy_all[i] for i in range(len(vy_all)) if success_all[i]]
        failed_vx = [vx_all[i] for i in range(len(vx_all)) if not success_all[i]]
        failed_vy = [vy_all[i] for i in range(len(vy_all)) if not success_all[i]]
        
        # Plot failed points (red)
        ax.scatter(failed_vx, failed_vy, c='red', s=1, alpha=0.3, label='Failed')
        
        # Plot successful points (green)
        ax.scatter(successful_vx, successful_vy, c='lime', s=3, alpha=0.8, label='Success')
        
        # Mark known successful cases
        ax.scatter([-2800], [1600], c='yellow', s=100, marker='*', 
                  label='Known Success', edgecolors='black', linewidth=1)
        ax.scatter([-2600], [1900], c='orange', s=100, marker='x', 
                  label='Known Fail', linewidth=3)
        
        ax.set_xlabel('X Velocity (m/s)', fontsize=14, color='white')
        ax.set_ylabel('Y Velocity (m/s)', fontsize=14, color='white')
        ax.set_title('Slingshot Parameter Space - Success/Fail Map', 
                    fontsize=16, color='white', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Add statistics text
        success_rate = (len(successful_vx) / len(vx_all)) * 100
        stats_text = f'Total tests: {len(vx_all)}\nSuccessful: {len(successful_vx)}\nSuccess rate: {success_rate:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=12, color='white', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot before showing (to avoid potential issues)
        try:
            plt.savefig('parameter_sweep_results.png', dpi=300, 
                       facecolor='black', edgecolor='none', 
                       bbox_inches='tight', pad_inches=0.1)
            print("📁 Visualization saved as 'parameter_sweep_results.png'")
        except Exception as e:
            print(f"Warning: Could not save plot: {e}")
        
        # Also save as PDF for better quality
        try:
            plt.savefig('parameter_sweep_results.pdf', 
                       facecolor='black', edgecolor='none', 
                       bbox_inches='tight', pad_inches=0.1)
            print("📁 High-quality version saved as 'parameter_sweep_results.pdf'")
        except Exception as e:
            print(f"Warning: Could not save PDF: {e}")
        
        # Save a white background version as backup
        try:
            # Temporarily change colors for white background version
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            ax.set_title('Slingshot Parameter Space - Success/Fail Map', 
                        fontsize=16, color='black', pad=20)
            ax.set_xlabel('X Velocity (m/s)', fontsize=14, color='black')
            ax.set_ylabel('Y Velocity (m/s)', fontsize=14, color='black')
            ax.tick_params(colors='black')
            
            # Update stats text color
            for text in ax.texts:
                if 'Total tests:' in text.get_text():
                    text.set_color('black')
                    text.get_bbox_patch().set_facecolor('white')
                    text.get_bbox_patch().set_edgecolor('black')
            
            plt.savefig('parameter_sweep_results_white.png', dpi=300, 
                       facecolor='white', edgecolor='none', 
                       bbox_inches='tight', pad_inches=0.1)
            print("📁 White background version saved as 'parameter_sweep_results_white.png'")
            
            # Restore original colors
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            ax.set_title('Slingshot Parameter Space - Success/Fail Map', 
                        fontsize=16, color='white', pad=20)
            ax.set_xlabel('X Velocity (m/s)', fontsize=14, color='white')
            ax.set_ylabel('Y Velocity (m/s)', fontsize=14, color='white')
            ax.tick_params(colors='white')
            
        except Exception as e:
            print(f"Warning: Could not save white background version: {e}")
        
        plt.show()
    
    def save_results(self, filename='successful_parameters.txt'):
        """Save successful parameters to file"""
        if not self.successful_params:
            print("No successful parameters to save!")
            return
        
        with open(filename, 'w') as f:
            f.write("# Successful Slingshot Parameters\n")
            f.write("# Format: Vx(m/s), Vy(m/s), Speed(m/s), HitTime(steps)\n")
            f.write("# Target: [4e7, 3e7] km, Radius: 100,000 km\n")
            f.write("# Spacecraft initial position: [3e7, -2e7] km\n\n")
            
            for param in self.successful_params:
                f.write(f"{param['vx']:6.0f}, {param['vy']:6.0f}, {param['speed']:7.0f}, {param['target_hit_time']:3}\n")
        
        print(f"💾 Successful parameters saved to '{filename}'")

def main():
    print("🚀 Slingshot Parameter Sweep Analysis")
    print("=" * 50)
    
    # Initialize sweep
    sweep = SlingshotParameterSweep()
    
    # Run parameter sweep
    successful_params, all_results = sweep.parameter_sweep()
    
    # Analyze results
    if successful_params:
        sweep.analyze_results()
        
        # Create visualization
        sweep.create_visualization()
        
        # Save results
        sweep.save_results()
    else:
        print("\n❌ No successful parameters found in the tested range!")
        print("Consider:")
        print("- Expanding the search range")
        print("- Reducing the step size")
        print("- Increasing the target radius")
        print("- Increasing the simulation steps")
    
    print("\n🎯 Parameter sweep analysis complete!")

if __name__ == "__main__":
    main() 