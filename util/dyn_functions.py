import jax.numpy as np
import numpy as onp
onp.random.seed(2)


def gen_ref(RA, CA, s0, T):
    dyn_ref = lambda s, t: RA @ s
    s = s0
    for t in range(T):
        yield CA @ s
        s = dyn_ref(s, t)


def gen_randomwalk(T, dw, eta):
    W = onp.zeros((T, dw))
    W[0, :] = onp.random.normal(size=(1, dw))
    for t in range(T-1):
        W[t+1, :] = 0.999 * W[t, :] + eta * onp.random.normal(size=(1, dw))
    return W


def set_dyn(A, B):
    return lambda x, u, w, t: A @ x + B @ u + w


def set_cost(Q, R):
    return lambda x, u, r, t: (x - r).T @ Q @ (x - r) + u.T @ R @ u


def eval_with_ref(dyn, control, cost, W, Ref, dx, T):
    x, z = np.zeros(dx), None
    for t in range(T):
        u, z = control(x, z, Ref[t], t)
        c = cost(x, u, Ref[t], t)
        yield (x, u, W[t], c)
        x = dyn(x, u, W[t], t)