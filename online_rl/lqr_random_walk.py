import jax.numpy as np
import jax
from jax.numpy.linalg import inv, pinv, matrix_power
from scipy.linalg import solve_discrete_are as dare

class LQR_Random_walk:
    def __init__(self, A, B, Q, R, RA, RC, Ref, l: int = 2,
                 T_change: int = None, RA2: np.ndarray = None, RC2: np.ndarray = None):
        self.A = A
        self.B = B
        self.dx, self.du = B.shape
        self.RA = RA
        self.RC = RC
        self.l = l
        self.Ref = Ref
        self.t = 0
        self.T_change = T_change
        self.RA2 = RA2
        self.RC2 = RC2

        self._extend_dynamic(A, B, Q)
        self._find_P_Kx(R)
        self.Pi, self.Lam, self.Kz, self.N = self._tracking_par(RA, RC, self.Kx)
        self.R_his = np.zeros((l, self.dx, 1))

    def _extend_dynamic(self, A, B, Q):
        self.A_ext = np.block([[A, 0.999*np.eye(self.dx)], [np.zeros((self.dx, self.dx)), 0.999*np.eye(self.dx)]])
        self.B_ext = np.block([[B], [np.zeros((self.du, self.dx))]])
        self.Q_ext = np.block([[Q, np.zeros((self.dx, self.dx))],
                               [np.zeros((self.dx, self.dx)), np.zeros((self.dx, self.dx))]])

    def _find_P_Kx(self, R):
        self.P = dare(self.A_ext, self.B_ext, self.Q_ext, R)
        self.Kx = -inv(R + self.B_ext.T @ self.P @ self.B_ext) @ self.B_ext.T @ self.P @ self.A_ext
    def _tracking_par(self, RA, RC, Kx):
        Kx_onlyx = self.Kx[:, :self.dx]
        Pi = RC
        Lam = Pi @ RA - self.A @ Pi # Only if B is eye
        Kz = Lam - Kx_onlyx @ Pi
        N = self._build_N(RA, RC)
        return Pi, Lam, Kz, N

    def _build_N(self,RA, RC):
        Ol = RC
        for i in range(1, self.l):
            Ol = np.concatenate([Ol, RC @ matrix_power(RA, i)])

        Ol_p = inv(Ol.T @ Ol) @ Ol.T
        return matrix_power(RA, self.l-1) @ Ol_p




    def lqr_control(self, state):
        # self.R_his = jax.ops.index_update(self.R_his, 0, (self.Ref[self.t]).reshape([self.dx, 1]))
        self.R_his = self.R_his.at[0].set((self.Ref[self.t]).reshape([self.dx, 1]))
        self.R_his = np.roll(self.R_his, -1, axis=0)
        ref_state = self.N @ self.R_his.reshape([self.dx * self.l, 1])
        u = (self.Kx @ state).flatten() + (self.Kz @ ref_state).flatten()
        self.t += 1
        if self.T_change == self.t:
            print('LQR for random walk-Dynamic changed.')
            self.Pi, self.Lam, self.Kz, self.N = self._tracking_par(self.RA2, self.RC2, self.Kx)
        return u

    def run_lqr_ext(self):
        def lqr_ext(x, z):
            if z is None:
                z = np.zeros(self.dx), np.zeros(self.du)
            xprev, uprev = z
            w = x - self.A @ xprev - self.B @ uprev
            u = self.lqr_control(np.block([x, w]))
            return u, (x, u)

        return lambda x, z, r, t: lqr_ext(x, z)

    def prepare_for_evaluation(self, new_Ref, T_change=None):
        self.Ref = new_Ref
        self.t = 0
        self.T_change = T_change
        self.R_his = np.zeros((self.l, self.dx, 1))

    def evaluate(self, new_Ref):
        self.Ref = new_Ref
        self.t = 0
        self.R_his = np.zeros((self.l, self.dx, 1))

        def lqr_ext(x, z):
            if z is None:
                z = np.zeros(self.dx), np.zeros(self.du)
            xprev, uprev = z
            w = x - self.A @ xprev - self.B @ uprev
            u = self.lqr_control(np.block([x, w]))
            return u, (x, u)
        return lambda x, z, r, t: lqr_ext(x, z)