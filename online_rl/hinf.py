import jax.numpy as np
import jax
from jax.numpy.linalg import inv, pinv, matrix_power
from scipy.linalg import solve_discrete_are as dare


class HINF:
    def __init__(self, A, B, Q, R, RA, RC, Ref, gamma=1.0, l: int = 2,
                 T_change: int = None, RA2: np.ndarray = None, RC2: np.ndarray = None):
        self.A = A
        self.B = B
        dx, du = B.shape
        self.dx = dx
        self.l = l
        self.Ref = Ref
        self.t = 0
        self.T_change = T_change
        self.RA2 = RA2
        self.RC2 = RC2
        self.P = dare(A, np.concatenate([B, np.eye(dx)], axis=1), Q,
                      np.block([[R, np.zeros([du, dx])], [np.zeros([dx, du]), -gamma**2 * np.eye(dx)]]))
        X = B.T @ self.P @ inv(gamma**2 * np.eye(dx)-self.P) @ self.P
        self.Kx = -inv(R + B.T @ self.P @ B + X @ B) @ (B.T @ self.P @ A + X @ A)
        lambda_cl, _ = np.linalg.eig(self.A + self.B @ self.Kx)
        if max(np.absolute(lambda_cl)) > (1.0-0.0001):
            print("Closed-loop system is not stable for gamma=", gamma)
        self.Pi, self.Lam, self.Kz, self.N = self._tracking_par(RA, RC, self.Kx)
        self.R_his = np.zeros((l, dx, 1))


    def _tracking_par(self, RA, RC, Kx):
        Pi = RC
        Lam = Pi @ RA - self.A @ Pi # Only if B is eye
        Kz = Lam - Kx @ Pi
        N = self._build_N(RA, RC)
        return Pi, Lam, Kz, N

    def _build_N(self,RA, RC):
        Ol = RC
        for i in range(1, self.l):
            Ol = np.concatenate([Ol, RC @ matrix_power(RA, i)])

        Ol_p = inv(Ol.T @ Ol) @ Ol.T
        return matrix_power(RA, self.l-1) @ Ol_p

    def hinf_control(self, state):
        # self.R_his = jax.ops.index_update(self.R_his, 0, (self.Ref[self.t]).reshape([self.dx, 1]))
        self.R_his = self.R_his.at[0].set((self.Ref[self.t]).reshape([self.dx, 1]))
        self.R_his = np.roll(self.R_his, -1, axis=0)
        ref_state = self.N @ self.R_his.reshape([self.dx * self.l, 1])
        u = (self.Kx @ state).flatten() + (self.Kz @ ref_state).flatten()
        self.t += 1
        if self.T_change == self.t:
            print('H infty-Dynamic changed.')
            self.Pi, self.Lam, self.Kz, self.N = self._tracking_par(self.RA2, self.RC2, self.Kx)
        return u

    def run_hinf(self):
        return lambda x, z, r, t: (self.hinf_control(x), z)

    def prepare_for_evaluation(self, new_Ref, T_change=None):
        self.Ref = new_Ref
        self.t = 0
        self.T_change = T_change
        self.R_his = np.zeros((self.l, self.dx, 1))

    def evaluate(self, new_Ref):
        self.Ref = new_Ref
        self.t = 0
        self.R_his = np.zeros((self.l, self.dx, 1))
        return lambda x, z, r, t: (self.hinf_control(x), z)