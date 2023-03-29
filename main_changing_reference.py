import jax.numpy as np
import numpy as onp
onp.random.seed(1)
import math
from online_state_tracking_feedforward import tracking_changing_reference_alorithm
from util.dyn_functions import gen_ref, gen_randomwalk

# System dynamics
A, B = np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]])
dx, du = B.shape
dy = dx

# Cost function
Q, R = 20*np.eye(dy), np.eye(du)

# Reference
ref_A1 = np.array([[0.0, 1.0, 0.0], [-1, 1.5, 0.0], [0.0, 0.0, 1.0]])
ref_C1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
ref_init1 = np.array([[1.0], [-2.0], [0.5]])

ref_A2 = np.array([[0.0, 1.0, 0.0], [-1, 0.0, 0.0], [0.0, 0.0, 0.0]])
ref_C2 = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 1.0]])
ref_init2 = np.array([[1.0], [0.0], [-1.0]])


general_par = {
    'A': A,
    'B': B,
    'Q': Q,
    'R': R,
    'ref_A1': ref_A1,
    'ref_C1': ref_C1,
    'ref_init1': ref_init1,
    'ref_A2': ref_A2,
    'ref_C2': ref_C2,
    'ref_init2': ref_init2,
    'l': 2,
    'T_change': 8000,
}



# Parameters for constant disturbance
T = 16000
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
T = 16000
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
T = 16000
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
T = 16000
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
T = 16000
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
T = 16000
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
new_Ref1 = np.array(list(gen_ref(ref_A1, ref_C1, ref_init1, int(new_T))))
new_Ref1 = new_Ref1.reshape([int(new_T), 2])
new_Ref2 = np.array(list(gen_ref(ref_A2, ref_C2, ref_init2, int(new_T))))
new_Ref2 = new_Ref2.reshape([int(new_T), 2])
new_Ref = np.concatenate((new_Ref1, new_Ref2), axis=0)
# Constant disturbance
alg_constant = tracking_changing_reference_alorithm(general_par)
alg_constant.tracking(Par_constant)
alg_constant.show_summary(T_change=general_par['T_change'], cumcmax=300.0)

# Amplitude modulation disturbance
# alg_amplitude = tracking_changing_reference_alorithm(general_par)
# alg_amplitude.tracking(Par_amplitude_modulation)
# alg_amplitude.show_summary(T_change=general_par['T_change'], cumcmax=300.0)

# Sinus disturbance
# alg_sinus = tracking_changing_reference_alorithm(general_par)
# alg_sinus.tracking(Par_sinus)
# alg_sinus.show_summary(T_change=general_par['T_change'], cumcmax=300.0)

# Gaussian disturbance
# alg_gaussian = tracking_changing_reference_alorithm(general_par)
# alg_gaussian.tracking(Par_gaussian)
# alg_gaussian.show_summary(T_change=general_par['T_change'], cumcmax=300.0)

# Randomwalk disturbance
# alg_randomwalk = tracking_changing_reference_alorithm(general_par)
# alg_randomwalk.tracking(Par_randomwalk)
# alg_randomwalk.show_summary(T_change=general_par['T_change'], cumcmax=500.0)

# Uniform disturbance
# alg_uniform = tracking_changing_reference_alorithm(general_par)
# alg_uniform.tracking(Par_uniform)
# alg_uniform.show_summary(T_change=general_par['T_change'], cumcmax=300.0)
