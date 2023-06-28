import numpy as np
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self):
        self.x0 = 4.0
        self.x = self.x0
        self.t0 = 0.0
        self.tF = 100.0
        self.dt = 0.0001
        self.steps = int((self.tF - self.t0)/self.dt)
        self.time_span = np.linspace(self.t0, self.tF, self.steps)
        self.a = 0.55
        self.h = 1
        self.l = 0.5
        self.r = 12.0
        self.w = 150
        self.theta = self.x
        self.J = self.f(self.theta)

        self.y0 = self.J
        self.y1 = 8.0
        self.y2 = 2.0

        self.x_traj = np.zeros((self.steps, ))
        self.x_traj[0] = self.x

        self.J_traj = np.zeros((self.steps, ))
        self.J_traj[0] = self.f(self.theta) 

        self.dJ_traj = np.zeros((self.steps, ))
        self.dJ_traj[0] = self.df(self.theta) 

        self.ddJ_traj = np.zeros((self.steps, ))
        self.ddJ_traj[0] = self.ddf(self.theta) 

        self.y0_traj = np.zeros((self.steps, ))
        self.y0_traj[0] = self.y0

        self.y1_traj = np.zeros((self.steps, ))
        self.y1_traj[0] = self.y1

        self.y2_traj = np.zeros((self.steps, ))
        self.y2_traj[0] = self.y2

    def f(self, x):
        return x**2.0 

    def df(self, x):
        return 2.0*x

    def ddf(self, x):
        return 2.0

    def run(self):
        for ii in range(self.steps - 1):
            t = self.time_span[ii+1]
            self.x = self.x - self.r * self.a * np.sin (self.w * t) * self.J * self.dt

            self.y0 = self.y0 + self.h * ( -self.y0 + self.J ) * self.dt
            self.y1 = self.y1 + self.l * ( -self.y1 + (-self.y0 + self.J ) * 2.0 * (1.0/self.a) * np.sin(self.w * t) ) * self.dt
            self.y2 = self.y2 + self.l * ( -self.y2 + (-self.y0 + self.J ) * 16.0 * (1.0/self.a ** 2.00) * (np.sin(self.w * t)**2.0 - 0.5)) * self.dt
            
            self.theta = self.x + self.a * np.sin(self.w * t)
            self.J = self.f(self.theta)

            self.x_traj[ii + 1] = self.x
            self.J_traj[ii + 1] = self.J
            self.dJ_traj[ii + 1] = self.df(self.theta)
            self.ddJ_traj[ii + 1] = self.ddf(self.theta)
            self.y0_traj[ii + 1] = self.y0
            self.y1_traj[ii + 1] = self.y1
            self.y2_traj[ii + 1] = self.y2

    def plot(self):
        # State
        fig, ax = plt.subplots()
        plt.plot(self.time_span, self.x_traj)
        ax.set_title("x(t)")

        fig, ax = plt.subplots()
        plt.plot(self.time_span, self.J_traj, label="J")
        plt.plot(self.time_span, self.y0_traj, label="y0")
        ax.set_title("J(t)")
        ax.legend()

        fig, ax = plt.subplots()
        plt.plot(self.time_span, self.dJ_traj, label="dJ")
        plt.plot(self.time_span, self.y1_traj, label="y1")
        ax.set_title("dJ(t)")
        ax.legend()

        fig, ax = plt.subplots()
        plt.plot(self.time_span, self.ddJ_traj, label="ddJ")
        plt.plot(self.time_span, self.y2_traj, label="y2")
        ax.set_title("ddJ(t)")
        ax.legend()

        plt.show()
        plt.close()

def main():
    sim = Simulation()
    sim.run()
    sim.plot()

if __name__ == '__main__':
    main()

