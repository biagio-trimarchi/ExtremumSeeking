# testCircle.py
# Author        :
# License       :
# Description   :

### TO DOs:
# --- Add p movements
# --- Store cost values on p trajectory
# --- Plot against filtered and theta value
# --- Fix error on second component of Hessian

### MODULES
import numpy as np
import matplotlib.pyplot as plt

class Simulation:

  def __init__(self):
    self.radius = 5                                 # Circle radius
    self.p = np.array([2.0, 0.0])                   # Position
    self.J = self.distanceFromCircularWall(self.p)  # Cost

    # Simulation parameters
    self.t0 = 0.0                                                 # Initial time
    self.tF = 100.0                                               # Final time
    self.dt = 0.01                                              # Simulation time step
    self.steps = int((self.tF-self.t0)/self.dt) + 1               # Number of simulation steps
    self.time_span = np.linspace(self.t0, self.tF, self.steps)    # Time vector

    # Algorithm parameters
    self.l = 0.5                            # Low pass gain
    self.h = 10.0                           # High pass gain
    self.k = 12.0                           # Integrator gain
    self.w = 150                            # Oscillation frequency
    self.a = 0.55                           # Dither
    self.theta = self.p.copy()              # Evaluation point

    # Algorithm quantities
    self.y0 = self.distanceFromCircularWall(self.p)   # Initial filtered cost
    self.y1 = self.gradient(self.p) 
    self.y2 = 2.0*np.eye(2)

    # Plotting quantities
    self.p_traj = np.zeros((self.steps, 2))
    self.p_traj[0, :] = self.p.copy()

    self.theta_traj = np.zeros((self.steps, 2))
    self.theta_traj[0, :] = self.p.copy()

    self.J_traj = np.zeros((self.steps, ))
    self.J_traj[0] = self.J

    self.y0_traj = np.zeros((self.steps, ))
    self.y0_traj[0] = self.J

    self.gradJ_traj = np.zeros((self.steps, 2))
    self.gradJ_traj[0, :] = self.y1.copy() 

    self.hessJ_traj = np.zeros((self.steps, 2, 2))
    self.hessJ_traj[0, :] = self.y2.copy() 

    self.y1_traj = np.zeros((self.steps, 2))
    self.y1_traj[0, :] = self.y1.copy() 

    self.y2_traj = np.zeros((self.steps, 2, 2))
    self.y2_traj[0, :, :] = self.y2.copy() 

  def distanceFromCircularWall(self, p):
    norm = np.linalg.norm(p)
    # return np.abs(norm - self.radius)
    return np.linalg.norm(p) ** 2

  def gradient(self, p):
    return 2.0*p.copy() 

  def hessian(self, p):
    return 2.0 * np.eye(2)

  def run(self):
    for ii in range(self.steps - 1):
      t = self.time_span[ii+1]
      
      # Modulation signal
      S = np.array([
        np.sin(self.w * t),
        np.sin(self.w/2.0 * t)
      ])

      # Demodulation signal (Gradient)
      M = np.array([
        2.0/self.a * np.sin(self.w * t),
        2.0/self.a * np.sin(self.w/2.0 * t)
      ])

      # Demodulation signal (Hessian)
      N = np.array([
        [16.0 / self.a ** 2.0 * (np.sin(self.w * t)**2.0 - 1.0/2.0), 4.0 / (self.a * self.a) * np.sin(self.w * t) * np.sin(self.w / 2.0 * t)], 
        [4.0 / (self.a * self.a) * np.sin(self.w * t) * np.sin(self.w / 2.0 * t), 16.0 / self.a ** 2.0 * (np.sin(self.w/2.0 * t)**2.0 - 1.0/2.0)] 
      ])

      # Update position
      self.p = self.p #- self.k * self.a * S * self.J * self.dt

      # Update sampling positin and cost
      self.theta = self.p + self.a * S
      self.J = self.distanceFromCircularWall(self.theta)

      # Update filters
      diff = self.J - self.y0
      self.y0 = self.y0 + self.h * diff * self.dt
      self.y1 = self.y1 + self.l * ( -self.y1 + diff * M ) * self.dt
      self.y2 = self.y2 + self.l * ( -self.y2 + diff * N ) * self.dt 
      
      # Update plots
      self.theta_traj[ii+1, :] = self.theta.copy()
      self.J_traj[ii+1] = self.J
      self.gradJ_traj[ii+1] = self.gradient(self.theta)
      self.hessJ_traj[ii+1] = self.hessian(self.theta)
      self.y0_traj[ii+1] = self.y0
      self.y1_traj[ii+1] = self.y1.copy()
      self.y2_traj[ii+1] = self.y2.copy()

  def plot(self):
    # State
    fig, ax = plt.subplots()
    plt.plot(self.time_span, self.theta_traj[:, 0], label="Virtual sensor")
    plt.plot(self.time_span, self.theta_traj[:, 1], label="Virtual sensor")
    ax.set_title("theta")

    fig, ax = plt.subplots()
    plt.plot(self.theta_traj[:, 0], self.theta_traj[:, 1], label="Virtual sensor")
    ax.set_title("theta")

    fig, ax = plt.subplots()
    plt.plot(self.time_span, self.J_traj, label="Real value")
    plt.plot(self.time_span, self.y0_traj, label="Filtered value")
    ax.legend()
    ax.set_title("Distance")

    fig, ax = plt.subplots(1, 2, sharey='row')
    ax[0].plot(self.time_span, self.gradJ_traj[:, 0], label="Real value")
    ax[0].plot(self.time_span, self.y1_traj[:, 0], label="Filtered value")
    ax[1].plot(self.time_span, self.gradJ_traj[:, 1], label="Real value")
    ax[1].plot(self.time_span, self.y1_traj[:, 1], label="Filtered value")
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title("Gradient 1")
    ax[1].set_title("Gradient 2")

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(self.time_span, self.hessJ_traj[:, 0, 0], label="Real value")
    ax[0, 0].plot(self.time_span, self.y2_traj[:, 0, 0], label="Filtered value")
    ax[0, 1].plot(self.time_span, self.hessJ_traj[:, 0, 1], label="Real value")
    ax[0, 1].plot(self.time_span, self.y2_traj[:, 0, 1], label="Filtered value")
    ax[1, 0].plot(self.time_span, self.hessJ_traj[:, 1, 0], label="Real value")
    ax[1, 0].plot(self.time_span, self.y2_traj[:, 1, 0], label="Filtered value")
    ax[1, 1].plot(self.time_span, self.hessJ_traj[:, 1, 1], label="Real value")
    ax[1, 1].plot(self.time_span, self.y2_traj[:, 1, 1], label="Filtered value")

    plt.show()
    plt.close()

def main():
  sim = Simulation()

  sim.run()
  sim.plot()

if __name__=='__main__':
  main()
