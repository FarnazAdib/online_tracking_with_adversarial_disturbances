import jax.numpy as np
import jax
from jax.numpy.linalg import inv, pinv, matrix_power
from scipy.linalg import solve_discrete_are as dare

class LQR:
    def __init__(self, A, B, Q, R, RA, RC, Ref, l: int = 2,
                 T_change: int = None, RA2: np.ndarray = None, RC2: np.ndarray = None):
        self.A = A
        self.dx, self.du = B.shape
        self.l = l
        self.Ref = Ref
        self.t = 0
        self.T_change = T_change
        self.RA2 = RA2
        self.RC2 = RC2
        self.P = dare(A, B, Q, R)
        self.Kx = -inv(R + B.T @ self.P @ B) @ B.T @ self.P @ A
        self.Pi, self.Lam, self.Kz, self.N = self._tracking_par(RA, RC, self.Kx)
        self.R_his = np.zeros((l, self.dx, 1))

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

    def build_Mr(self, mr):
        Mr = np.zeros((mr, self.du, self.dx))

        for i in range(self.l):
            Mr_i=self.Kz @ self.N[:, (self.l-i - 1) * self.dx:(self.l-i) * self.dx]
            Mr = jax.ops.index_update(Mr, i,Mr_i)
        print('Mr is', Mr)
        print('K_ff is', self.Kz)
        print('N is', self.N)
        print('K_ff@N is', self.Kz@self.N)

    def lqr_control(self, state):
        # self.R_his = jax.ops.index_update(self.R_his, 0, (self.Ref[self.t]).reshape([self.dx, 1]))
        self.R_his = self.R_his.at[0].set((self.Ref[self.t]).reshape([self.dx, 1]))
        self.R_his = np.roll(self.R_his, -1, axis=0)

        ref_state = self.N @ self.R_his.reshape([self.dx * self.l, 1])
        u = (self.Kx @ state).flatten() + (self.Kz @ ref_state).flatten()
        self.t += 1
        if self.T_change == self.t:
            print('LQR-Dynamic changed.')
            self.Pi, self.Lam, self.Kz, self.N = self._tracking_par(self.RA2, self.RC2, self.Kx)
        return u

    def run_lqr(self):
        return lambda x, z, r, t: (self.lqr_control(x), z)

    def prepare_for_evaluation(self, new_Ref, T_change=None):
        self.Ref = new_Ref
        self.T_change = T_change
        self.t = 0
        self.R_his = np.zeros((self.l, self.dx, 1))

    def evaluate(self, new_Ref):
        self.Ref = new_Ref
        self.t = 0
        self.R_his = np.zeros((self.l, self.dx, 1))
        return lambda x, z, r, t: (self.lqr_control(x), z)