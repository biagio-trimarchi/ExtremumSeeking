import numpy as np
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self):
        self.x0 = 4.0
        self.x = self.x0
        self.t0 = 0.0
        self.tF = 1000.0
        self.dt = 0.001
        self.steps = int((self.tF - self.t0)/self.dt)
        self.time_span = np.linspace(self.t0, self.tF, self.steps)
        self.l = 0.5
        self.r = 0.05
        self.w = 50
        self.theta = self.x
        self.J = self.f(self.theta)

        self.x_traj = np.zeros((self.steps, ))
        self.x_traj[0] = self.x

        self.J_traj = np.zeros((self.steps, ))
        self.J_traj[0] = self.f(self.theta) 



    def f(self, x):
        return x**2.0 + 25.0

    def df(self, x):
        return 2.0*x

    def ddf(self, x):
        return 2.0

    def run(self):
        for ii in range(self.steps - 1):
            t = self.time_span[ii+1]
            self.x = self.x - self.r * self.l * np.sin (self.w * t) * self.J * self.dt
            
            self.theta = self.x + np.sin(self.w * t)
            self.J = self.f(self.theta)

            self.x_traj[ii + 1] = self.x

    def plot(self):
        # State
        fig, ax = plt.subplots()
        plt.plot(self.time_span, self.x_traj)
        ax.set_title("x(t)")

        plt.show()
        plt.close()

def main():
    sim = Simulation()
    sim.run()
    sim.plot()

if __name__ == '__main__':
    main()

