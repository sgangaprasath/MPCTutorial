import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon
from matplotlib.transforms import Affine2D
import matplotlib.cm as cm

class RocketMPC:
    
    def __init__(self):
        # --- Physical Parameters ---
        self.m_wet = 35.0          
        self.m_dry = 20.0          
        self.v_e = 2500.0          
        self.l = 0.8               # m — Distance from COM to Engine Gimbal
        self.l_cop = 0.4           # m — Distance from COM to Center of Pressure (Positive = COP is BELOW COM)
        self.g = 9.81     
        
        # --- MPC Time Parameters ---
        self.dt = 0.1          
        self.N_horizon = 40        
        self.sim_time = 15.0        
        self.max_steps = int(self.sim_time / self.dt)
        
        # --- Actuator & Aero Parameters ---
        self.T_min = 0.0          
        self.rho = 1.225           
        
        self.T_max = 700.0
        self.dT_max = 2500.0
        self.delta_max = np.radians(15)
        self.ddelta_max = np.radians(60)
        self.theta_max = np.radians(15)

        self.I_base = 10.0
        self.cd_area = 0.04
                
        # --- States: [y, z, theta, y_dot, z_dot, theta_dot, mass, T, delta] ---
        # Stress Test Initialization
        self.start_state = np.array([15.0, 200.0,np.radians(45), 3.0, -35.0, 0.05, 28.0, 0.0, 0.0])
        self.target_kinematics = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Flags
        self.crashed = False
        self.Windgust = True # Turned on to test the new torques!

        self.setup_mpc_controller()
    
    def get_dynamics(self, x, u):
        
        theta = x[2]
        m = x[6]           
        T_raw = x[7]  # We now treat this as the 'commanded' thrust      
        delta = x[8]       
        
        T_dot = u[0]
        delta_dot = u[1]
        
        y_dot = x[3]
        z_dot = x[4]
        theta_dot = x[5]
        
        v_mag = ca.sqrt(y_dot**2 + z_dot**2 + 1e-5) 
        D_y = 0.5 * self.rho * self.cd_area * v_mag * y_dot
        D_z = 0.5 * self.rho * self.cd_area * v_mag * z_dot
        
        I = self.I_base * (m / self.m_wet)

        # --- NEW: Tanh Smoothing for Engine Cutoff ---
        # k_sharp controls how abrupt the cutoff is. Higher = sharper.
        # We shift by a small margin (e.g., 0.1 kg) so it fully shuts down near m_dry
        k_sharp = 10.0 
        margin = 0.1
        
        # alpha is ~1.0 when m > m_dry, and smoothly drops to ~0.0 as m -> m_dry
        alpha = 0.5 * (1.0 + ca.tanh(k_sharp * (m - (self.m_dry + margin))))
        
        # Effective thrust applies the smooth shutdown
        T_eff = T_raw * alpha

        # Use T_eff for all physical forces and mass depletion
        y_ddot = (T_eff / m) * ca.sin(theta + delta) - (D_y / m)
        z_ddot = (T_eff / m) * ca.cos(theta + delta) - self.g - (D_z / m)
        
        tau_engine = -self.l * T_eff * ca.sin(delta)
        tau_aero = self.l_cop * (D_y * ca.cos(theta) - D_z * ca.sin(theta))
        theta_ddot = (tau_engine + tau_aero) / I
        
        m_dot = -T_eff / self.v_e
        
        return ca.vertcat(y_dot, z_dot, theta_dot, y_ddot, z_ddot, theta_ddot, m_dot, T_dot, delta_dot)

    def rk4_step(self, x, u, dt):
        k1 = self.get_dynamics(x, u)
        k2 = self.get_dynamics(x + dt/2.0 * k1, u)
        k3 = self.get_dynamics(x + dt/2.0 * k2, u)
        k4 = self.get_dynamics(x + dt * k3, u)
        return x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def setup_mpc_controller(self):
        self.opti = ca.Opti()
        
        self.X = self.opti.variable(9, self.N_horizon + 1) 
        self.U = self.opti.variable(2, self.N_horizon)
        
        self.P_x0 = self.opti.parameter(9)
        self.opti.subject_to(self.X[:, 0] == self.P_x0)
        
        cost = 0
        
        # Heavily increased penalty on pitch (index 2) and pitch rate (index 5) 
        # to fight the new aerodynamic wind torque!
        Q = np.diag([0, 20, 500, 10, 10, 100])  
        P = Q * 5.0                                             
        
        W_T = 0.1     
        R_dT = 0.05   
        R_dd = 10.0
        pad_radius = 5.0 # +/- 5 meters

        for k in range(self.N_horizon):
            x_k = self.X[:, k]
            u_k = self.U[:, k]
            
            err = x_k[:6] - self.target_kinematics
            cost += ca.mtimes([err.T, Q, err])
            
            # --- NEW: Landing Pad Deadband Cost ---
            y_pos = x_k[0]
            # Tiny pull toward dead-center to keep the solver math stable
            cost += 0.1 * y_pos**2 
            # MASSIVE penalty if it strays outside the +/- 5m landing pad bounds
            cost += 100.0 * ca.fmax(0, ca.fabs(y_pos) - pad_radius)**2
            
            # Actuator Effort
            cost += W_T * x_k[7] + 0.001 * (x_k[7])**2 + R_dT * (u_k[0])**2 + R_dd * (u_k[1])**2
            
            # Dynamics Constraints
            x_next = self.rk4_step(x_k, u_k, self.dt)
            self.opti.subject_to(self.X[:, k+1] == x_next)
            
            # Limits
            self.opti.subject_to(self.opti.bounded(-self.theta_max,self.X[2,k],self.theta_max))
            self.opti.subject_to(self.opti.bounded(-self.dT_max, u_k[0], self.dT_max))
            self.opti.subject_to(self.opti.bounded(-self.ddelta_max, u_k[1], self.ddelta_max))
            self.opti.subject_to(self.opti.bounded(self.T_min, self.X[7, k], self.T_max))
            self.opti.subject_to(self.opti.bounded(-self.delta_max, self.X[8, k], self.delta_max))
            self.opti.subject_to(self.X[1, k] >= -0.05)                             
            self.opti.subject_to(self.opti.bounded(self.m_dry, self.X[6, k], self.m_wet))
      
        # Terminal Cost
        err_N = self.X[:6, self.N_horizon] - self.target_kinematics
        cost += ca.mtimes([err_N.T, P, err_N])
        
        # Terminal Deadband Cost (even stricter)
        y_pos_N = self.X[0, self.N_horizon]
        cost += 1.0 * y_pos_N**2
        cost += 500.0 * ca.fmax(0, ca.fabs(y_pos_N) - pad_radius)**2
        
        self.opti.subject_to(self.X[1, self.N_horizon] >= -0.05)

        self.opti.subject_to(self.opti.bounded(self.T_min, self.X[7, self.N_horizon], self.T_max))
        self.opti.subject_to(self.opti.bounded(-self.delta_max, self.X[8, self.N_horizon], self.delta_max))
        self.opti.subject_to(self.opti.bounded(self.m_dry, self.X[6, self.N_horizon], self.m_wet))

        self.opti.minimize(cost)

        p_opts = {"expand": True}
        s_opts = {"max_iter": 100, "print_level": 0, "print_timing_statistics": "no", "sb": "yes"}
        self.opti.solver("ipopt", p_opts, s_opts)

    def run_simulation(self):
        print("Starting v2 Simulation: Full Aero Torques...")
        
        current_state = self.start_state.copy()
        self.history_X = [current_state]
        self.history_U = []
        
        u_guess = np.tile([0.0, 0.0], (self.N_horizon, 1)).T
        x_guess = np.tile(self.start_state, (self.N_horizon + 1, 1)).T
        
        if self.Windgust:
            random_time = np.random.uniform(0.5, 3.5)
            self.wind_hit_time = round(random_time / self.dt) * self.dt
            print(f"Weather Update: Random wind gust forecasted around t={self.wind_hit_time:.1f}s!")
        else:
            self.wind_hit_time = -1.0
        
        for step in range(self.max_steps):
            current_time = step * self.dt
            
            self.opti.set_value(self.P_x0, current_state)
            self.opti.set_initial(self.X, x_guess)
            self.opti.set_initial(self.U, u_guess)
            
            try:
                sol = self.opti.solve()
                x_opt = sol.value(self.X)
                u_opt = sol.value(self.U)
            except Exception as e:
                x_opt = self.opti.debug.value(self.X)
                u_opt = self.opti.debug.value(self.U)

            u_applied = u_opt[:, 0].copy() 
            
            if current_state[6] <= self.m_dry:
                u_applied[0] = 0.0 
                u_applied[1] = 0.0 
                current_state[7] = 0.0 
                
            self.history_U.append(u_applied) 
            
            if abs(current_time - self.wind_hit_time) < 1e-3:
                print(f"[{current_time}s] MASSIVE WIND GUST DETECTED!")
                # The wind gust adds raw horizontal velocity, which triggers the aero torque!
                current_state[3] -= 8.0  
                current_state[5] += 0.5  

            next_state_dm = self.rk4_step(current_state, u_applied, self.dt)
            current_state = np.array(next_state_dm).flatten()
            
            self.history_X.append(current_state)
            
            u_guess = np.hstack((u_opt[:, 1:], u_opt[:, -1:]))
            x_guess = np.hstack((x_opt[:, 1:], x_opt[:, -1:]))
            
            if current_state[1] < -0.1:
                print(f"\nFATAL: CRASHED below the landing pad at t={current_time:.2f}s!")
                self.crashed = True
                break

            if np.linalg.norm(current_state[:3]) < 0.1 and np.linalg.norm(current_state[3:6]) < 0.5:
                print(f"\nLanded safely at t={current_time:.2f}s! Remaining Fuel: {(current_state[6]-self.m_dry):.2f} kg")
                break
                
            if step % 10 == 0:
                print(f"Simulating... t = {current_time:.1f}s | Mass: {current_state[6]:.2f}kg")

        self.history_X = np.array(self.history_X).T
        self.history_U = np.array(self.history_U).T
        self.actual_steps = len(self.history_U[0])
        print("Simulation Complete!")

    def plot_results(self):
        """Highly detailed diagnostic plotting for debugging MPC behavior"""
        # Note: States (X) have actual_steps + 1 entries. Controls (U) have actual_steps entries.
        time_x = np.linspace(0, self.actual_steps * self.dt, self.actual_steps + 1)
        time_u = np.linspace(0, (self.actual_steps - 1) * self.dt, self.actual_steps)
        
        fig, axs = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle("MPC Diagnostic Dashboard: Flight Dynamics & Controls", fontsize=16, fontweight='bold')
        
        # --- ROW 1: KINEMATICS ---
        # 1. Trajectory
        axs[0, 0].plot(self.history_X[0, :], self.history_X[1, :], 'b.-', label="Flight Path")
        axs[0, 0].plot(self.start_state[0], self.start_state[1], 'go', markersize=8, label="Start")
        axs[0, 0].plot(self.target_kinematics[0], self.target_kinematics[1], 'rx', markersize=10, label="Target")
        axs[0, 0].set_title("1. Spatial Trajectory (y vs z)")
        axs[0, 0].set_xlabel("y (m)"); axs[0, 0].set_ylabel("z (m)")
        axs[0, 0].axis('equal'); axs[0, 0].grid(True); axs[0, 0].legend()

        # 2. Velocities
        axs[0, 1].plot(time_x, self.history_X[3, :], 'c-', linewidth=2, label="Vy (Horizontal)")
        axs[0, 1].plot(time_x, self.history_X[4, :], 'm-', linewidth=2, label="Vz (Vertical)")
        axs[0, 1].axvline(self.wind_hit_time, color='r', linestyle='--', alpha=0.5)
        axs[0, 1].set_title("2. Velocities over Time")
        axs[0, 1].set_xlabel("Time (s)"); axs[0, 1].set_ylabel("Velocity (m/s)")
        axs[0, 1].grid(True); axs[0, 1].legend()

        # 3. Pitch Dynamics
        axs[0, 2].plot(time_x, np.degrees(self.history_X[2, :]), 'k-', linewidth=2, label="Pitch (deg)")
        axs[0, 2].plot(time_x, np.degrees(self.history_X[5, :]), 'orange', linewidth=2, label="Pitch Rate (deg/s)")
        axs[0, 2].axvline(self.wind_hit_time, color='r', linestyle='--', alpha=0.5)
        axs[0, 2].set_title("3. Pitch Behavior")
        axs[0, 2].set_xlabel("Time (s)"); axs[0, 2].set_ylabel("Degrees")
        axs[0, 2].grid(True); axs[0, 2].legend()

        # --- ROW 2: STATES (Physics Output) ---
        # 4. Phase Portrait (Pitch vs Pitch Rate)
        axs[1, 0].plot(np.degrees(self.history_X[2, :]), np.degrees(self.history_X[5, :]), 'g.-')
        axs[1, 0].plot(np.degrees(self.start_state[2]), np.degrees(self.start_state[5]), 'go', label="Start")
        axs[1, 0].set_title("4. Pitch Phase Portrait")
        axs[1, 0].set_xlabel("Pitch (deg)"); axs[1, 0].set_ylabel("Pitch Rate (deg/s)")
        axs[1, 0].grid(True); axs[1, 0].legend()

        # 5. Thrust State
        axs[1, 1].step(time_x, self.history_X[7, :], 'tab:red', linewidth=2, where='post', label="Thrust")
        hover_history = self.history_X[6, :] * self.g
        axs[1, 1].plot(time_x, hover_history, 'g--', label="Hover T")
        axs[1, 1].set_title("5. Thrust State (T)")
        axs[1, 1].set_xlabel("Time (s)"); axs[1, 1].set_ylabel("Force (N)")
        axs[1, 1].grid(True); axs[1, 1].legend()

        # 6. Gimbal State
        axs[1, 2].step(time_x, np.degrees(self.history_X[8, :]), 'tab:blue', linewidth=2, where='post', label="Gimbal")
        axs[1, 2].axhline(np.degrees(self.delta_max), color='k', linestyle=':', label="Max limit")
        axs[1, 2].axhline(-np.degrees(self.delta_max), color='k', linestyle=':')
        axs[1, 2].set_title("6. Gimbal State (delta)")
        axs[1, 2].set_xlabel("Time (s)"); axs[1, 2].set_ylabel("Angle (deg)")
        axs[1, 2].grid(True); axs[1, 2].legend()

        # --- ROW 3: CONTROLS & MASS (Solver Output) ---
        # 7. Mass Depletion
        axs[2, 0].plot(time_x, self.history_X[6, :], 'purple', linewidth=2, label="Mass")
        axs[2, 0].axhline(self.m_dry, color='k', linestyle='--', label="Empty")
        axs[2, 0].set_title("7. Propellant Mass")
        axs[2, 0].set_xlabel("Time (s)"); axs[2, 0].set_ylabel("Mass (kg)")
        axs[2, 0].grid(True); axs[2, 0].legend()

        # 8. Commanded Thrust Rate
        axs[2, 1].step(time_u, self.history_U[0, :], 'r-', linewidth=1.5, where='post', label="dT Command")
        axs[2, 1].set_title("8. Control: Thrust Rate (U0)")
        axs[2, 1].set_xlabel("Time (s)"); axs[2, 1].set_ylabel("Rate (N/s)")
        axs[2, 1].grid(True); axs[2, 1].legend()

        # 9. Commanded Gimbal Rate
        axs[2, 2].step(time_u, np.degrees(self.history_U[1, :]), 'b-', linewidth=1.5, where='post', label="ddelta Command")
        axs[2, 2].set_title("9. Control: Gimbal Rate (U1)")
        axs[2, 2].set_xlabel("Time (s)"); axs[2, 2].set_ylabel("Rate (deg/s)")
        axs[2, 2].grid(True); axs[2, 2].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('debug.jpg')
        # plt.show()
    
    def animate_results(self):

        frames = self.actual_steps

        # -----------------------------
        # Visual Settings
        # -----------------------------
        VISUAL_SCALE = 6.0
        MAX_FLAME_LENGTH = 4.0
        CAMERA_WIDTH = 25
        CAMERA_HEIGHT_UP = 40
        CAMERA_HEIGHT_DOWN = 10

        # -----------------------------
        # Create Figure with 2 Panels
        # -----------------------------
        fig, (ax_world, ax_follow) = plt.subplots(
            1, 2, figsize=(14, 8), num="MPC Rocket Dual View"
        )
        crash_text = ax_world.text(0.5, 0.5, '', transform=ax_world.transAxes, color='darkred', 
                             fontsize=50, fontweight='bold', ha='center', va='center')
        
        telemetry_text = ax_follow.text(
            0.02, 0.98, "",
            transform=ax_follow.transAxes,
            fontsize=10,
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
            color='lime'
        )

        # -----------------------------
        # WORLD VIEW (Fixed)
        # -----------------------------
        ax_world.set_title("World View (Fixed Frame)")
        ax_world.set_xlim(-30, 30)
        ax_world.set_ylim(-10, self.start_state[1] + 10)
        ax_world.set_aspect('equal')
        ax_world.grid(True)

        traj_world, = ax_world.plot([], [], 'g--', alpha=0.6)
        pad_world = Rectangle((-10, -6.2), 20, 0.5, fc='darkgray')
        ax_world.add_patch(pad_world)

        # -----------------------------
        # FOLLOW VIEW (Dynamic)
        # -----------------------------
        ax_follow.set_title("Rocket Follow View")
        ax_follow.set_aspect('equal')
        ax_follow.grid(True)

        traj_follow, = ax_follow.plot([], [], 'g--', alpha=0.6)
        pad_follow = Rectangle((-10, -6.2), 20, 0.5, fc='darkgray')
        ax_follow.add_patch(pad_follow)

        # -----------------------------
        # Rocket Geometry (shared logic)
        # -----------------------------
        def create_rocket(ax):
            body = Rectangle(
                (-0.2*VISUAL_SCALE, -0.6*VISUAL_SCALE),
                0.4*VISUAL_SCALE,
                1.2*VISUAL_SCALE,
                fc='white', ec='black'
            )

            nose = Polygon(
                [[-0.2*VISUAL_SCALE, 0.6*VISUAL_SCALE],
                [ 0.2*VISUAL_SCALE, 0.6*VISUAL_SCALE],
                [ 0.0,              1.0*VISUAL_SCALE]],
                fc='black'
            )

            engine = Polygon(
                [[-0.06*VISUAL_SCALE, 0],
                [ 0.06*VISUAL_SCALE, 0],
                [ 0.1*VISUAL_SCALE, -0.4*VISUAL_SCALE],
                [-0.1*VISUAL_SCALE, -0.4*VISUAL_SCALE]],
                fc='gray', ec='black'
            )

            flame = Polygon(
                [[-0.08*VISUAL_SCALE, -0.4*VISUAL_SCALE],
                [ 0.08*VISUAL_SCALE, -0.4*VISUAL_SCALE],
                [ 0.0,               -0.4*VISUAL_SCALE]],
                fc='orange', ec='none', alpha=0.0
            )

            for p in [body, nose, engine, flame]:
                ax.add_patch(p)

            return body, nose, engine, flame

        rocket_world = create_rocket(ax_world)
        rocket_follow = create_rocket(ax_follow)

        cmap = cm.get_cmap('YlOrRd')

        # -----------------------------
        # Animation Update
        # -----------------------------
        def update(i):
            time = i * self.dt

            y = self.history_X[0, i]
            z = self.history_X[1, i]
            theta = self.history_X[2, i]
            thrust = self.history_X[7, i]
            delta = self.history_X[8, i]
            y_dot = self.history_X[3, i]
            z_dot = self.history_X[4, i]
            mass = self.history_X[6, i]
            
            telemetry_text.set_text(
                f"TIME:   {time:5.2f} s\n"
                f"ALT:    {z:6.2f} m\n"
                f"VZ:     {z_dot:6.2f} m/s\n"
                f"VY:     {y_dot:6.2f} m/s\n"
                f"PITCH:  {np.degrees(theta):6.2f} deg\n"
                f"THRUST: {thrust:6.1f} N\n"
                f"MASS:   {mass:6.2f} kg"
            )

            # Update trajectories
            traj_world.set_data(self.history_X[0, :i],
                                self.history_X[1, :i])
            traj_follow.set_data(self.history_X[0, :i],
                                self.history_X[1, :i])

            # Follow camera window
            ax_follow.set_xlim(y - CAMERA_WIDTH/2,
                               y + CAMERA_WIDTH/2)
            ax_follow.set_ylim(z - CAMERA_HEIGHT_DOWN,
                               z + CAMERA_HEIGHT_UP)

            thrust_frac = np.clip(thrust / self.T_max, 0.0, 1.0)
            flame_len = MAX_FLAME_LENGTH * thrust_frac
            
            if self.crashed and i == frames - 1:
                crash_text.set_text("CRASHED")
            else:
                crash_text.set_text("")

            for (body, nose, engine, flame), ax in [
                (rocket_world, ax_world),
                (rocket_follow, ax_follow)
            ]:

                # Flame shape
                flame.set_xy([
                    [-0.08*VISUAL_SCALE, -0.4*VISUAL_SCALE],
                    [ 0.08*VISUAL_SCALE, -0.4*VISUAL_SCALE],
                    [ 0.0, -0.4*VISUAL_SCALE - flame_len]
                ])

                flame.set_facecolor(cmap(thrust_frac))
                flame.set_alpha(float(thrust_frac))

                # Body transform
                body_trans = (
                    Affine2D()
                    .rotate(theta)
                    .translate(y, z)
                    + ax.transData
                )

                body.set_transform(body_trans)
                nose.set_transform(body_trans)

                engine_trans = (
                    Affine2D()
                    .rotate(delta)
                    .translate(0, -0.6*VISUAL_SCALE)
                    + body_trans
                )

                engine.set_transform(engine_trans)
                flame.set_transform(engine_trans)

            return (
                traj_world, traj_follow,
                *rocket_world, *rocket_follow,telemetry_text
            )

        ani = animation.FuncAnimation(
            fig, update,
            frames=frames,
            interval=int(self.dt * 1000),
            blit=False
        )
        
        ani.save(
        "rocket_landing_try.gif",
        writer="pillow",
        fps=15
    )

        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":
    mpc = RocketMPC()
    mpc.run_simulation()
    
    mpc.plot_results()
    mpc.animate_results() # Uncomment to render the gif