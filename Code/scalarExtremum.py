# Extremum.py
# Author        : Biagio Trimarchi
# License       : None
# Description   : 

### MODULES
import numpy as np              # Numerical methods module
import matplotlib.pyplot as plt   # Plotting module

class Simulation:
    def __init__(self):
        self.lim = 4.0                              # Limit
        self.x0 = 5.0                               # Initial condition
        self.x = self.x0                            # Actual state
        self.t0 = 0.0                               # Initial time
        self.dt = 0.001                             # Simulation time step
        self.tF = 100.0                             # Final time
        self.steps = int(self.tF/self.dt) + 1       # Number of simulation steps
        self.l = 0.01                               # Small gain
        self.r = 0.05                               # Dither
        self.w = 10                                 # Oscillation frequency
        self.y0 = self.J(self.x)                    # Estimated cost
        self.y1 = self.dJ(self.x)                   # Estimated gradient
        self.y2 = self.ddJ(self.x)                  # Estimated hessian

        self.time_span = np.linspace(self.t0, self.tF, self.steps)      # Simulation time span
        
        self.x_traj = np.zeros((self.steps, ))                          # State trajectory 
        self.x_traj[0] = self.x                                         # Initial state
        
        self.J_traj = np.zeros((self.steps, ))                          # Cost along the trajectory
        self.J_traj[0] = self.J(self.x)                                 # Initial cost
        
        self.dJ_traj = np.zeros((self.steps, ))                         # Gradient cost function along trajectory
        self.dJ_traj[0] = self.dJ(self.x)                               # Initial gradient cost function

        self.ddJ_traj = np.zeros((self.steps, ))                        # Hessian cost function along trajectory
        self.ddJ_traj[0] = self.ddJ(self.x)                             # Initial hessian cost function

        self.y0_traj = np.zeros((self.steps, )) 
        self.y0_traj[0] = self.J(self.x)     

        self.y1_traj = np.zeros((self.steps, ))
        self.y1_traj[0] = self.dJ(self.x)     

        self.y2_traj = np.zeros((self.steps, )) 
        self.y2_traj[0] = self.ddJ(self.x)     

    def J(self, x):
        """
            Cost function
        """

        return x**4.0 / 650

    def dJ(self, x):
        """
            Gradient cost function
        """

        return 4*x**3 / 650

    def ddJ(self, x):
        """
            Hessian cost function
        """

        return 12*x**2 / 650

    def B(self, x):
        """
            Infinite cost barrier
        """

        return 1.0/(x-self.lim)**2.0

    def run(self):
        for tt in range(0, self.steps-1):
            t = self.time_span[tt]                                     # Actual time
            self.x = self.x0 + self.r*np.sin(self.w * t)               # Update state
            
            J = self.J(self.x)                                    # Cost on new state
            dJ = self.dJ(self.x)                                  # Gradient on new state
            ddJ = self.ddJ(self.x)                                # Hessian on new state

            self.y0 = self.y0 + self.l * ( -self.y0 + J ) * self.dt
            self.y1 = self.y1 + self.l * ( -self.y1 +  (-self.y0 + J) * np.sin(self.w * t) ) * self.dt
            self.y2 = self.y2 + self.l * ( -self.y2 + (-self.y0 + J) * np.sin(self.w * t) ** 2 ) * self.dt

            self.x_traj[tt+1] = self.x
            self.J_traj[tt+1] = J
            self.dJ_traj[tt+1] = dJ
            self.ddJ_traj[tt+1] = ddJ
            self.y0_traj[tt+1] = self.y0
            self.y1_traj[tt+1] = self.y1
            self.y2_traj[tt+1] = self.y2
            

    def plot(self):
        # Plot state trajectory
        fig, ax = plt.subplots()
        plt.plot(self.time_span, self.x_traj)
        ax.set_title("x(t)")
        
        # Plot cost
        fig, ax = plt.subplots()
        plt.plot(self.time_span, self.J_traj, label="Cost")
        plt.plot(self.time_span, self.y0_traj, label="Mean")
        ax.set_title("J(t)")

        # Plot gradient
        fig, ax = plt.subplots()
        plt.plot(self.time_span, self.dJ_traj, label="Gradient")
        plt.plot(self.time_span, self.y1_traj, label="Mean")
        ax.legend()
        ax.set_title("dJ(t)")

        # Plot hessian
        fig, ax = plt.subplots()
        plt.plot(self.time_span, self.ddJ_traj, label="Hessian")
        plt.plot(self.time_span, self.y2_traj, label="Mean")
        ax.legend()
        ax.set_title("ddJ(t)")

        plt.show()
        plt.close()

def main():
    sim = Simulation()

    sim.run()
    sim.plot()

if __name__=='__main__':
    main()
