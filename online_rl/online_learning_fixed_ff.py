# Copyright 2021 The Deluca Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from numbers import Real
from typing import Callable
from scipy.linalg import solve_discrete_are as dare
import jax
import jax.numpy as jnp
from jax.numpy.linalg import inv, matrix_power
import numpy as np
from jax import grad
from jax import jit


def quad_loss(x: jnp.ndarray, u: jnp.ndarray, r: jnp.ndarray) -> Real:
    """
    Quadratic loss.
    Args:
        x (jnp.ndarray):
        u (jnp.ndarray):
    Returns:
        Real
    """
    return jnp.sum((x - r).T @ (x - r) + u.T @ u)


class Online_Learning_Fixed_FF:
    def __init__(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        RA, RC,
        Ref: jnp.ndarray,
        Q: jnp.ndarray = None,
        R: jnp.ndarray = None,
        K: jnp.ndarray = None,
        T_change: int = 0,
        RA2: jnp.ndarray = None,
        RC2: jnp.ndarray = None,
        start_time: int = 0,
        cost_fn: Callable[[jnp.ndarray, jnp.ndarray], Real] = None,
        mw: int = 3,
        h: int = 2,
        lr_w: Real = 0.005,
        decay: bool = True,
        l: int = 2,

    ) -> None:
        """
        Description: Initialize the dynamics of the model.
        Args:
            A (jnp.ndarray): system dynamics
            B (jnp.ndarray): system dynamics
            Q (jnp.ndarray): cost matrices (i.e. cost = x^TQx + u^TRu)
            R (jnp.ndarray): cost matrices (i.e. cost = x^TQx + u^TRu)
            K (jnp.ndarray): Starting policy (optional). Defaults to LQR gain.
            start_time (int):
            cost_fn (Callable[[jnp.ndarray, jnp.ndarray], Real]):
            H (postive int): history of the controller
            HH (positive int): history of the system
            lr_scale (Real):
            lr_scale_decay (Real):
            decay (Real):
        """


        d_state, d_action = B.shape  # State & Action Dimensions
        self.d_state = d_state
        self.d_action = d_action
        self.A, self.B = A, B  # System Dynamics
        self.Ref = Ref # Reference trajectory
        self.t = 0  # Time Counter (for decaying learning rate)
        self.T_change = int(T_change)
        self.RA2 = RA2
        self.RC2 = RC2
        self.l = l

        self.mw, self.h = mw, h

        self.lr_w, self.decay = lr_w, decay

        self.bias = 0

        # Model Parameters
        # initial linear policy / perturbation contributions / bias
        # TODO: need to address problem of LQR with jax.lax.scan
        P = dare(A, B, Q, R)
        self.K = K if K is not None else - np.array(inv(R + B.T @ P @ B) @ (B.T @ P @ A))
        self.Pi, self.Lam, self.Kz, self.N = self._tracking_par(RA, RC, self.K)

        self.Mw = jnp.zeros((mw, d_action, d_state))


        # self.Mr = jax.ops.index_update(self.Mr, 1, np.array([[-0.0456, -0.02187], [0.0019, 0.479]]) )
        # self.Mr = jax.ops.index_update(self.Mr, 0, np.array([[-1, -0.022], [0, 0.4791]]))
        # Past mw+h noises and references ordered increasing in time
        self.W_his = jnp.zeros((h + mw, d_state, 1))
        self.R_his = jnp.zeros((self.l, d_state, 1))

        # past state and past action
        self.state, self.action = jnp.zeros((d_state, 1)), jnp.zeros((d_action, 1))
        cost_fn = lambda x, u, r: jnp.sum((x - r).T @ Q @ (x - r) + u.T @ R @ u)

        def proxy(Mw, W_his, t):
            y = np.zeros([self.d_state, 1])
            for i in range(h):
                v = self.K @ y + jnp.tensordot(Mw, W_his[i: i + mw], axes=([0, 2], [0, 1]))
                y = self.A @ y + self.B @ v + W_his[i + mw]

            v = self.K @ y + jnp.tensordot(Mw, W_his[i: i + mw], axes=([0, 2], [0, 1]))

            c = cost_fn(y.reshape([self.d_state, 1]), v.reshape(self.d_action, 1), Ref[t].reshape([d_state, 1]))
            return c

        def policy_loss(Mw, W_his, t):
            """Surrogate cost function"""

            def action(state, i):
                """Action function"""
                return self.K @ state \
                       + jnp.tensordot(Mw, jax.lax.dynamic_slice_in_dim(W_his, i, mw), axes=([0, 2], [0, 1]))

            def evolve(state, i):
                """Evolve function"""
                return self.A @ state + self.B @ action(state, i) + W_his[i + mw], None

            final_state, _ = jax.lax.scan(evolve, np.zeros((d_state, 1)), np.arange(h))
            return cost_fn(final_state, action(final_state, h - 1), Ref[t].reshape([d_state, 1]))

        self.policy_loss = policy_loss
        self.grad = jit(grad(policy_loss, (0)))

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

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Description: Return the action based on current state and internal parameters.
        Args:
            state (jnp.ndarray): current state
        Returns:
           jnp.ndarray: action to take
        """
        state = state.reshape([self.d_state, 1])
        self.update(state)
        action = self.get_action(state)
        self.state = state
        self.action = action
        self.t += 1
        if self.T_change == self.t:
            print('Online learning plus fixed ff-Dynamic changed.')
            self.Pi, self.Lam, self.Kz, self.N = self._tracking_par(self.RA2, self.RC2, self.K)
        return action.flatten()

    def update(self, state: jnp.ndarray) -> None:
        """
        Description: update agent internal state.
        Args:
            state (jnp.ndarray):
        Returns:
            None
        """

        # self.W_his = jax.ops.index_update(self.W_his, 0, state - self.A @ self.state - self.B @ self.action)
        self.W_his = self.W_his.at[0].set(state - self.A @ self.state - self.B @ self.action)
        self.W_his = jnp.roll(self.W_his, -1, axis=0)
        # self.R_his = jax.ops.index_update(self.R_his, 0, (self.Ref[self.t]).reshape([self.d_state, 1]))
        self.R_his = self.R_his.at[0].set((self.Ref[self.t]).reshape([self.d_state, 1]))
        self.R_his = jnp.roll(self.R_his, -1, axis=0)

        if self.t >= self.h +self.mw:
            Mw_delta = self.grad(self.Mw, self.W_his, self.t)
            self.Mw -= self.lr_w * Mw_delta
            # print("Synchronized Mw", self.Mw)
            # print("Synchronized Mr", self.Mr)
        # update state

    def get_action(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Description: get action from state.
        Args:
            state (jnp.ndarray):
        Returns:
            jnp.ndarray
        """
        ref_state = self.N @ self.R_his.reshape([self.d_state * self.l, 1])
        # Note here that Mr[l] is multiplied by r_{k}, Mr[0] is multipled by r_{k-l} so the order of Mr is the reverse of the paper.
        if self.t > 6 or self.t > self.T_change+6:
            u = self.K @ state + jnp.tensordot(self.Mw, self.W_his[-self.mw:], axes=([0, 2], [0, 1]))\
                + self.Kz @ ref_state
        else:
            u = self.K @ state + jnp.tensordot(self.Mw, self.W_his[-self.mw:], axes=([0, 2], [0, 1]))

        return u

    def run_online_learning_fixed_ff(self):
        return lambda x, z, Ref, t: (self.__call__(x), None)

    def prepare_for_evaluation(self, new_Ref, T_change=0):
        self.Ref = new_Ref
        self.t = 0
        self.T_change = T_change
        self.W_his = jnp.zeros((self.h + self.mw, self.d_state, 1))
        self.R_his = jnp.zeros((self.l, self.d_state, 1))
        self.lr_w, self.decay = 0.0, 0.0

    def evaluate(self, new_Ref):
        self.Ref = new_Ref
        self.t = 0
        self.W_his = jnp.zeros((self.h + self.mw, self.d_state, 1))
        self.R_his = jnp.zeros((self.l, self.d_state, 1))
        self.lr_w, self.decay = 0.0, 0.0
        return lambda x, z, Ref, t: (self.__call__(x), None)