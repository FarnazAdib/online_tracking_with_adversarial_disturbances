import jax.numpy as np
import numpy as onp
onp.random.seed(1)
import math
from online_state_tracking_feedforward import tracking_alorithm
from util.dyn_functions import gen_ref, gen_randomwalk

# System dynamics
A, B = np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]])
dx, du = B.shape
dy = dx

# Cost function
Q, R = 20*np.eye(dy), np.eye(du)

# Reference
ref_A = np.array([[0.0, 1.0, 0.0], [-1, 1.5, 0.0], [0.0, 0.0, 1.0]])
ref_C = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
ref_init = np.array([[1.0], [-2.0], [0.5]])

general_par = {
    'A': A,
    'B': B,
    'Q': Q,
    'R': R,
    'ref_A': ref_A,
    'ref_C': ref_C,
    'ref_init': ref_init,
    'l': 2
}



# Parameters for constant disturbance
T = 10000
magnitude = 1.0
W1 = np.concatenate((magnitude * np.ones((int(T/2), 1)),  magnitude * np.ones((int(T/2), 1))))
W2 = np.concatenate((1 * magnitude * np.ones((int(T/4), 1)), 1 * magnitude * np.ones((int(3*T/4), 1))))
W = np.concatenate((W1, W2), axis=1)
Par_constant = {
    'T': T,
    'W': W,
    'W_label': 'Constant disturbance',
    'gpc_h': 5,
    'gpc_mw': 5,
    'gpc_mr': 5,
    'gpc_lrw': 1*10**(-4),
    'gpc_lrr': 1*10**(-4),
    'hinf_gamma': 1.5,
    'alg': ['Algorithm 1', 'online_learning_fixed_ff', 'Hinf Control', 'LQR']
}

# Parameters for amplitude modluation
T = 10000
W = 1*np.multiply(np.tile(np.sin(np.arange(T) * 2 * np.pi * 3.0 / 500), (2, 1)),
                  np.tile(np.sin(np.arange(T) * 2 * np.pi * 4.0 / 500), (2, 1)))
W = W.reshape([T, 2])
Par_amplitude_modulation = {
    'T': T,
    'W': W,
    'W_label': 'Amplitude modulation disturbance',
    'gpc_h': 5,
    'gpc_mw': 5,
    'gpc_mr': 5,
    'gpc_lrw': 1*10**(-4),
    'gpc_lrr': 1*10**(-4),
    'hinf_gamma': 1.5,
    'alg': ['Algorithm 1', 'online_learning_fixed_ff', 'Hinf Control', 'LQR']
}

# Parameters for sinus
T = 10000
W = 1* np.tile(np.sin(np.arange(T) * 2 * np.pi * 4 / 100), (2, 1)).T
W = W.reshape([T, 2])
Par_sinus = {
    'T': T,
    'W': W,
    'W_label': 'Sinusoidal disturbance',
    'gpc_h': 5,
    'gpc_mw': 5,
    'gpc_mr': 5,
    'gpc_lrw': 1*10**(-4),
    'gpc_lrr': 1*10**(-4),
    'hinf_gamma': 1.5,
    'alg': ['Algorithm 1', 'online_learning_fixed_ff', 'Hinf Control', 'LQR']
}

# Parameters for Gaussian
T = 10000
Par_gaussian = {
    'T': T,
    'W': 0.1 * onp.random.normal(size=(T, dx)),
    'W_label': 'Gaussian disturbance',
    'gpc_h': 5,
    'gpc_mw': 5,
    'gpc_mr': 5,
    'gpc_lrw': 1*10**(-4),
    'gpc_lrr': 1*10**(-4),
    'hinf_gamma': 1.5,
    'alg': ['Algorithm 1', 'online_learning_fixed_ff', 'Hinf Control', 'LQR']
}


# Parameters for Random walk. Set T =1000
T = 10000
W = gen_randomwalk(T, dx, 0.1)
Par_randomwalk = {
    'T': T,
    'W': W,
    'W_label': 'Random walk disturbance',
    'gpc_h': 5,
    'gpc_mw': 5,
    'gpc_mr': 5,
    'gpc_lrw': 1*10**(-4),
    'gpc_lrr': 1*10**(-4),
    'hinf_gamma': 1.5,
    'alg': ['Algorithm 1', 'online_learning_fixed_ff', 'Hinf Control', 'LQR_Random walk', 'LQR']
}

# Parameters for uniform
T = 10000
Par_uniform = {
    'T': T,
    'W': 1 * onp.random.random((T, dx)),
    'W_label': 'Uniform disturbance',
    'gpc_h': 5,
    'gpc_mw': 5,
    'gpc_mr': 5,
    'gpc_lrw': 1*10**(-4),
    'gpc_lrr': 1*10**(-4),
    'hinf_gamma': 1.5,
    'alg': ['Algorithm 1', 'online_learning_fixed_ff', 'Hinf Control', 'LQR']
}

# Evaluation
new_T = 30
new_Ref = np.array(list(gen_ref(ref_A, ref_C, ref_init, int(new_T))))
new_Ref = new_Ref.reshape([int(new_T), 2])

# Constant disturbance
# alg_constant = tracking_alorithm(general_par)
# alg_constant.tracking(Par_constant)
# alg_constant.show_summary(cumcmax=300.0)
# new_W = np.ones((int(new_T), 2))
# alg_constant.evaluat_alg(new_Ref, new_W, new_T, cumcmax=200.0, xmin=-6.0, xmax=6.0, emin=-6.0, emax=6.0)

# Amplitude modulation disturbance
# alg_amplitude = tracking_alorithm(general_par)
# alg_amplitude.tracking(Par_amplitude_modulation)
# alg_amplitude.show_summary(cumcmax=300.0)
# new_W = 1*np.multiply(np.tile(np.sin(np.arange(new_T) * 2 * np.pi * 3.0 / 500), (2, 1)),
#                       np.tile(np.sin(np.arange(new_T) * 2 * np.pi * 4.0 / 500), (2, 1)))
# new_W = new_W.reshape([new_T, 2])
# alg_amplitude.evaluat_alg(new_Ref, new_W, new_T, cumcmax=200.0, xmin=-6.0, xmax=6.0, emin=-6.0, emax=6.0)

# Sinus disturbance
alg_sinus = tracking_alorithm(general_par)
alg_sinus.tracking(Par_sinus)
alg_sinus.show_summary(cumcmax=300.0)
new_W = 1 * np.tile(np.sin(np.arange(new_T) * 2 * np.pi * 4 / 100), (2, 1)).T
new_W = new_W.reshape([new_T, 2])
alg_sinus.evaluat_alg(new_Ref, new_W, new_T, cumcmax=200.0, xmin=-6.0, xmax=6.0, emin=-6.0, emax=6.0)

# Gaussian disturbance
# alg_gaussian = tracking_alorithm(general_par)
# alg_gaussian.tracking(Par_gaussian)
# alg_gaussian.show_summary(cumcmax=300.0)
# new_W = 0.1 * onp.random.normal(size=(new_T, dx))
# alg_gaussian.evaluat_alg(new_Ref, new_W, new_T, cumcmax=200.0, xmin=-6.0, xmax=6.0, emin=-6.0, emax=6.0)

# Randomwalk disturbance
# alg_randomwalk = tracking_alorithm(general_par)
# alg_randomwalk.tracking(Par_randomwalk)
# alg_randomwalk.show_summary(cumcmax=500.0)
# new_W = gen_randomwalk(new_T, dx, 0.1)
# alg_randomwalk.evaluat_alg(new_Ref, new_W, new_T, cumcmax=200.0, xmin=-6.0, xmax=6.0, emin=-6.0, emax=6.0)

# Uniform disturbance
# alg_uniform = tracking_alorithm(general_par)
# alg_uniform.tracking(Par_uniform)
# alg_uniform.show_summary(cumcmax=300.0)
# new_W = 1 * onp.random.random((new_T, dx))
# alg_uniform.evaluat_alg(new_Ref, new_W, new_T, cumcmax=200.0, xmin=-6.0, xmax=6.0, emin=-6.0, emax=6.0)
