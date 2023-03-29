import jax.numpy as np
class ZERO:
    def __init__(self,du):
        self.du = du

    def run_zero(self):
        return lambda x, z, r, t: (np.zeros(self.du), z)