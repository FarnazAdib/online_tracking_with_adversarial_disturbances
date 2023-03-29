import jax.numpy as np
import jax
from jax.numpy.linalg import inv, pinv, matrix_power
from scipy.linalg import solve_discrete_are as dare
# TODO one change I made to make it work is to use for I = self.I + self.past_e instead of e according to Ben Recht
#TODO previously I used -Kd D to make the control but it does not make sense.
#TODO I decreased the d gain and now pid works smoothly
class PID:
    def __init__(self, KP, KI, KD, Ref, dx, dt=1.0, Ilim=1000, beta=0.1):
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.Ref = Ref
        self.dx = dx
        self.past_e = np.zeros([dx, 1])
        self.P = np.zeros([dx, 1])
        self.I = np.zeros([dx, 1])
        self.D = np.zeros([dx, 1])
        self.Ilim = Ilim
        self.t = 0
        self.dt = dt
        self.beta = beta

    def pid_control(self, state):
        e = (self.Ref[self.t] - state).reshape([self.dx, 1])
        P = e

        # I part

        I = self.I + self.dt * self.past_e
        for i in range(self.dx):
            if I[i, 0] > self.Ilim:
                I = jax.ops.index_update(I, i, self.Ilim)
            if I[i, 0] < -self.Ilim:
                I = jax.ops.index_update(I, i, -self.Ilim)

        # D part
        D = self.beta*self.D + (1 - self.beta) * (e - self.past_e) / self.dt
        self.past_e = e
        self.t = self.t + 1
        self.P, self.I, self.D = P, I, D

        return (self.KP @ e + self.KI @ I + self.KD @ D).flatten()

    def run_pid(self):
        return lambda x, z, r, t: (self.pid_control(x), z)