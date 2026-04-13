import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.pyplot as plt  <- Removed duplicate import
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
import random

# Apply a professional plot style
plt.style.use('seaborn-v0_8-darkgrid')

# --- QUINTIC POLYNOMIAL HELPER ---
class QuinticPolynomial:
    """
    Calculates the 6 coefficients (a0-a5) for a 5th-degree polynomial.
    
    This is the core of the trajectory generator, solving the boundary
    value problem for a given start/end state and time duration [cite: 18-19].
    """
    def __init__(self, start_pos, start_vel, start_acc, end_pos, end_vel, end_acc, T):
        
        # [cite: 34-36]
        self.a0 = start_pos
        self.a1 = start_vel
        self.a2 = start_acc / 2.0
        
        # Setup the 3x3 matrix 'A' and 3x1 vector 'b' to solve for a3, a4, a5 [cite: 42-43]
        A = np.array([[T**3, T**4, T**5],
                      [3*T**2, 4*T**3, 5*T**4],
                      [6*T, 12*T**2, 20*T**3]])
        
        b = np.array([end_pos - (self.a0 + self.a1*T + self.a2*T**2),
                      end_vel - (self.a1 + 2*self.a2*T),
                      end_acc - (2*self.a2)])
        
        try:
            # Solve the linear system Ax = b for x = [a3, a4, a5]
            self.a3, self.a4, self.a5 = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Handle cases where the matrix is singular (e.g., T=0)
            self.a3, self.a4, self.a5 = 0.0, 0.0, 0.0
            print("Warning: Singular matrix in QuinticPolynomial. Using zero coefficients.")

    def calc_pos(self, t):
        # s(t) = a0 + a1*t + a2*t^2 + ... [cite: 22]
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5*t**5

    def calc_vel(self, t):
        # v(t) = ds/dt [cite: 24]
        return self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3 + 5*self.a5*t**4

    def calc_acc(self, t):
        # a(t) = d^2s/dt^2 [cite: 25]
        return 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3

    def calc_jerk(self, t):
        # j(t) = d^3s/dt^3 [cite: 26]
        return 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2

# --- RANDOMIZED ROAD DEFINITION ---
# Randomize road shape for robust testing [cite: 59-60]
ROAD_AMP = random.uniform(4.0, 8.0)
ROAD_FREQ = random.uniform(25.0, 40.0) 
ROAD_WIDTH = 14.0 

def get_road_boundaries(x_vals):
    """Calculates the center, left, and right road boundaries based on a sine wave."""
    y_center = ROAD_AMP * np.sin(x_vals / ROAD_FREQ) + 5.0
    y_left = y_center + ROAD_WIDTH / 2.0
    y_right = y_center - ROAD_WIDTH / 2.0
    return y_center, y_left, y_right

def get_road_yaw(x):
    """Calculates the road's angle (yaw) at a given x-position."""
    # Yaw is the angle of the tangent, found using the derivative dy/dx
    dy_dx = (ROAD_AMP / ROAD_FREQ) * np.cos(x / ROAD_FREQ)
    return np.arctan2(dy_dx, 1.0)

# --- COLLISION CHECKING & COST ---
def check_collision_details(rx, ry, obstacle, road_func):
    """
    Checks a given path (rx, ry) for collisions against an obstacle and road boundaries.
    Returns: (bool collision_detected, float min_distance_to_obstacle)
    """
    ox, oy, width, height = obstacle
    min_dist = float('inf') 
    collision = False
    
    # Safety margin for car size (approx 1.3m buffer) [cite: 116-117]
    margin = 1.3
    x_min, x_max = ox - width/2 - margin, ox + width/2 + margin
    y_min, y_max = oy - height/2 - margin, oy + height/2 + margin

    for px, py in zip(rx, ry):
        # Approximate distance to rectangle center
        dist = np.hypot(px - ox, py - oy)
        if dist < min_dist: 
            min_dist = dist
        
        # 1. Rectangle Collision Check [cite: 124-126]
        if x_min <= px <= x_max and y_min <= py <= y_max:
            collision = True
            
        # 2. Road boundary check [cite: 127-129]
        # (Check if we've driven off the road)
        _, y_l, y_r = road_func(np.array([px]))
        if py > y_l[0] - 0.5 or py < y_r[0] + 0.5: # 0.5m buffer from edge
            collision = True

    return collision, min_dist

def plan_single_segment(start, end_nominal, obstacle, road_func):
    """
    Generates and tests multiple trajectories to find the optimal one.
    This is the "generate-and-test" local planner. [cite: 64-65]
    """
    sx, sy, syaw, sv, sa = start
    gx_nom, gy_nom, gyaw_nom, gv, ga = end_nominal
    
    # Decompose start/end states into x and y components
    svx, svy = sv * np.cos(syaw), sv * np.sin(syaw)
    sax, say = sa * np.cos(syaw), sa * np.sin(syaw)

    best_safe_path = None
    min_safe_cost = float('inf')
    # Removed best_unsafe_path and max_unsafe_dist to fix crashing bug [cite: 138-139]

    # 1. Generate Candidates
    # Sample different lateral positions (d_lat) [cite: 67]
    lateral_offsets = np.linspace(-5.0, 5.0, 15) 
    # Sample different maneuver durations (T) [cite: 68]
    t_range = np.arange(4.0, 10.0, 1.0) 

    for lat_offset in lateral_offsets:
        # Calculate the new target (x, y) based on the lateral offset
        gy = gy_nom + lat_offset * np.cos(gyaw_nom)
        gx = gx_nom - lat_offset * np.sin(gyaw_nom)
        gvx, gvy = gv * np.cos(gyaw_nom), gv * np.sin(gyaw_nom)
        gax, gay = ga * np.cos(gyaw_nom), ga * np.sin(gyaw_nom)

        for T in t_range:
            # Create two 1D quintic polynomials (one for x, one for y)
            xp = QuinticPolynomial(sx, svx, sax, gx, gvx, gax, T)
            yp = QuinticPolynomial(sy, svy, say, gy, gvy, gay, T)
            
            # --- FIX (RELIABILITY): Increased collision checks from 20 to 40 ---
            t_check = np.linspace(0, T, 40) # 
            rx = [xp.calc_pos(i) for i in t_check]
            ry = [yp.calc_pos(i) for i in t_check]
            
            # 2. Evaluate Candidates
            collision, _ = check_collision_details(rx, ry, obstacle, road_func) # min_dist no longer needed [cite: 153]
            
            if not collision:
                # Calculate cost function J *only* for safe paths
                jx = [xp.calc_jerk(i) for i in t_check]
                jy = [yp.calc_jerk(i) for i in t_check]
                
                # J = (wt * T) + (wj * Jerk) + (wd * d_lat) [cite: 74]
                T_cost = T  # wt = 1.0 (implied) - Prefers shorter maneuvers
                
                # --- FIX (EFFICIENCY): Jerk cost reduced from 0.1 to 0.01 to favor faster paths ---
                jerk_cost = 0.01 * np.sum(np.square(jx) + np.square(jy)) # wj = 0.01 - Prefers comfort 
                lat_cost = 2.0 * abs(lat_offset) # wd = 2.0 - Prefers staying near center
                
                base_cost = T_cost + jerk_cost + lat_cost

                path_data = {'T': T, 'end_state': [gx, gy, gyaw_nom, gv, ga], 'polys': (xp, yp), 'cost': base_cost}

                # 3. Select Optimal Path
                # This path is safe. Is it the best *safe* path?
                if base_cost < min_safe_cost:
                    min_safe_cost = base_cost
                    best_safe_path = path_data
            # Removed 'else' block that saved unsafe paths [cite: 162-165]

    # --- FIX (CRASHING): Only accept safe paths. Removed the 'else best_unsafe_path' fallback. ---
    final_path = best_safe_path # [cite: 166]

    if final_path:
        # Generate a high-resolution path for the *chosen* trajectory
        T_best = final_path['T']
        xp, yp = final_path['polys']
        t_high_res = np.linspace(0, T_best, 60) # 60 points for smooth animation
        
        final_path['x'] = [xp.calc_pos(i) for i in t_high_res]
        final_path['y'] = [yp.calc_pos(i) for i in t_high_res]
        # Calculate yaw by taking the angle of the path's gradient (dy/dx)
        final_path['yaw'] = np.arctan2(np.gradient(final_path['y']), np.gradient(final_path['x']))
        
        # Removed the "Warning: Using fallback path" print [cite: 174-175]
        return final_path
    
    return None # No *safe* path could be generated

# --- ADDED FOR RUBRIC ---
def plot_illustrative_example():
    """
    Generates the Position, Velocity, and Acceleration time-series plots.
    This uses the "Stop" maneuver example from the report [cite: 45-53]
    to satisfy the rubric's visualization requirement.
    """
    print("Generating P-V-A diagnostic plot...")
    
    # 1. Define Boundary Conditions from report example [cite: 46-48]
    s0 = 0.0   # m
    v0 = 20.0  # m/s
    a0 = 0.0   # m/s^2
    
    sT = 50.0  # m
    vT = 0.0   # m/s
    aT = 0.0   # m/s^2
    
    T = 5.0    # seconds
    
    # 2. Solve for the 1D trajectory
    traj = QuinticPolynomial(s0, v0, a0, sT, vT, aT, T)
    
    # 3. Sample the trajectory over time
    t = np.linspace(0, T, 100)
    pos = [traj.calc_pos(i) for i in t]
    vel = [traj.calc_vel(i) for i in t]
    acc = [traj.calc_acc(i) for i in t]
    
    # 4. Create the 3-panel plot
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Kinematic Profile for 'Stop' Maneuver (T={T}s)", fontsize=16)

    # Position Plot
    ax[0].plot(t, pos, 'b-', linewidth=2, label='Position')
    ax[0].axhline(sT, color='b', linestyle='--', label=f'Target Pos: {sT}m')
    ax[0].set_ylabel('Position (m)')
    ax[0].legend()
    ax[0].grid(True)
    
    # Velocity Plot
    ax[1].plot(t, vel, 'g-', linewidth=2, label='Velocity')
    ax[1].axhline(v0, color='g', linestyle='--', label=f'Start Vel: {v0}m/s')
    ax[1].axhline(vT, color='r', linestyle='--', label=f'Target Vel: {vT}m/s')
    ax[1].set_ylabel('Velocity (m/s)')
    ax[1].legend()
    ax[1].grid(True)

    # Acceleration Plot
    ax[2].plot(t, acc, 'r-', linewidth=2, label='Acceleration')
    ax[2].axhline(a0, color='r', linestyle='--', label=f'Start/End Acc: {a0}m/s²')
    ax[2].set_ylabel('Acceleration (m/s²)')
    ax[2].set_xlabel('Time (s)')
    ax[2].legend()
    ax[2].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    # Note: plt.show() is called at the end of main()

# =========================================
# Main Execution
# =========================================
if __name__ == "__main__":
    
    # --- 1. Generate the P-V-A plot required by the rubric ---
    plot_illustrative_example()
    
    # --- 2. Setup the Main Simulation Environment ---
    x_road = np.linspace(0, 150, 750)
    yc, yl, yr = get_road_boundaries(x_road)
    
    # Define high-level waypoints
    wp_x = [0.0, 50.0, 100.0, 150.0]
    waypoints = []
    for i, wx in enumerate(wp_x):
        wy, _, _ = get_road_boundaries(np.array([wx]))
        # Target 12 m/s, but 0 m/s at the final waypoint
        v_target = 12.0 if i < len(wp_x)-1 else 0.0
        waypoints.append([wx, wy[0], get_road_yaw(wx), v_target, 0.0]) # [x, y, yaw, v, a]

    # Procedurally generate obstacles [cite: 190-191]
    obstacles = []
    for i in range(len(waypoints) - 1):
        # Place an obstacle somewhere between two waypoints
        ox = random.uniform(wp_x[i] + 20, wp_x[i+1] - 20)
        lane_pos = random.choice([-1, 0, 1]) # Left, center, or right lane
        oy_center, _, _ = get_road_boundaries(np.array([ox]))
        oy = oy_center[0] + lane_pos * (ROAD_WIDTH / 4.5)
        
        # Randomize "brick" dimensions [cite: 196-200]
        if random.random() > 0.5:
            width, height = random.uniform(3.0, 5.0), random.uniform(1.5, 2.5) # Horizontal
        else:
            width, height = random.uniform(1.5, 2.5), random.uniform(3.0, 5.0) # Vertical
            
        obstacles.append((ox, oy, width, height))

    # Generate a simple, unoptimized path for comparison [cite: 202-206]
    unopt_x, unopt_y = [], [] 
    for i in range(len(waypoints) - 1):
        p1, p2 = waypoints[i], waypoints[i+1]
        unopt_x.extend(np.linspace(p1[0], p2[0], 15))
        unopt_y.extend(np.linspace(p1[1], p2[1], 15))

    # --- 3. Run the Planner ---
    # These lists will store the final, concatenated path
    opt_x, opt_y, opt_yaw = [], [], []
    current_state = waypoints[0]
    print(f"Planning on Randomized Road (Amp: {ROAD_AMP:.1f}, Freq: {ROAD_FREQ:.1f})...")
    
    # Plan segment by segment (waypoint to waypoint)
    for i in range(len(waypoints) - 1):
        print(f"Processing Segment {i+1}/{len(waypoints)-1}...")
        path = plan_single_segment(current_state, waypoints[i+1], obstacles[i], get_road_boundaries)
        
        if path:
            # Add the new path segment to the total path
            opt_x.extend(path['x'])
            opt_y.extend(path['y'])
            opt_yaw.extend(path['yaw'])
            # The end of this path is the start of the next
            current_state = path['end_state']
        else:
            # This is now a true failure case, as the planner found NO safe path.
            print(f"CRITICAL FAIL at segment {i+1}! No safe path found. Stopping.")
            break

    # --- 4. VISUALIZATION ---
    print("Planning complete. Launching visualization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # --- Plot 1: Static Comparison --- [cite: 222-232]
    ax1.set_title("Comparison: Straight-Line vs. Optimized Trajectory")
    ax1.fill_between(x_road, yl, yr, color='gray', alpha=0.3, label='Road')
    ax1.plot(x_road, yc, 'w--', linewidth=1, label='Centerline')
    ax1.plot(x_road, yl, 'k-'); ax1.plot(x_road, yr, 'k-') # Road boundaries
    for ox, oy, w, h in obstacles: 
        ax1.add_patch(Rectangle((ox - w/2, oy - h/2), w, h, color='firebrick', alpha=0.8, label='Obstacle'))
    ax1.plot(unopt_x, unopt_y, 'r--', linewidth=2, label='Unoptimized')
    ax1.plot(opt_x, opt_y, 'g-', linewidth=3, label='Optimized')
    ax1.set_aspect('equal')
    ax1.set_ylim(np.min(yr) - 5, np.max(yl) + 5)
    # Manually create a legend to avoid duplicate labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    ax1.grid(True)

    # --- Plot 2: Animation --- [cite: 233-252]
    ax2.set_title("Animation: Optimized Autonomous Driving")
    ax2.fill_between(x_road, yl, yr, color='gray', alpha=0.3)
    ax2.plot(x_road, yc, 'w--', linewidth=1)
    ax2.plot(x_road, yl, 'k-'); ax2.plot(x_road, yr, 'k-')
    for ox, oy, w, h in obstacles: 
        ax2.add_patch(Rectangle((ox - w/2, oy - h/2), w, h, color='firebrick'))
    ax2.set_aspect('equal'); ax2.set_ylim(np.min(yr) - 5, np.max(yl) + 5); ax2.grid(True)
    
    # Path trace (what's behind the car)
    path_line, = ax2.plot([], [], 'b-', linewidth=2, alpha=0.5)
    # Car body
    car_body = Rectangle((0, 0), 3.0, 1.5, color='blue', zorder=10)
    ax2.add_patch(car_body)

    def update(frame):
        """Animation update function"""
        if frame >= len(opt_x): 
            return path_line, car_body
        
        x, y, yaw = opt_x[frame], opt_y[frame], opt_yaw[frame]
        
        # Calculate the car's transformation matrix
        tr = transforms.Affine2D().rotate_around(x, y, yaw) + ax2.transData
        car_body.set_transform(tr)
        car_body.set_xy((x - 3.0/2, y - 1.5/2)) # Set car's center
        
        # Update the path trace
        path_line.set_data(opt_x[:frame], opt_y[:frame])
        
        # Camera follows the car
        ax2.set_xlim(x - 50, x + 50)
        return path_line, car_body

    ani = animation.FuncAnimation(fig, update, frames=len(opt_x), blit=False, interval=30, repeat=False)
    
    plt.tight_layout()
    plt.show()