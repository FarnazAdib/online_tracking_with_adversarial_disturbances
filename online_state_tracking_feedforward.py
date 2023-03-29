import jax.numpy as np
from toolz.dicttoolz import valmap, itemmap
from online_rl.gpc import GPC
from online_rl.zero_control import ZERO
from online_rl.hinf import HINF
from online_rl.lqr import LQR
from online_rl.pid import PID
from online_rl.lqr_random_walk import LQR_Random_walk
from online_rl.online_learning_fixed_ff import Online_Learning_Fixed_FF
from util.dyn_functions import gen_ref, eval_with_ref, set_dyn, set_cost
from util.plot_functions import plot_costs, plot_errors, plot_immediate_cost, plot_states, plot_ref, \
    plot_errors_details, plot_ref_details, plot_costs2
from util.results_functions import report_ave_costs

class tracking_alorithm:
    def __init__(self, general_par: dict):
        self.A = general_par['A']
        self.B = general_par['B']
        self.Q = general_par['Q']
        self.R = general_par['R']
        self.ref_A = general_par['ref_A']
        self.ref_C = general_par['ref_C']
        self.ref_init = general_par['ref_init']
        self.l = general_par['l']

    def tracking(self, par: dict):
        Ref = np.array(list(gen_ref(self.ref_A, self.ref_C, self.ref_init, int(par['T']))))
        Ref = Ref.reshape([int(par['T']), 2])
        self.Ref = Ref
        self.dyn = set_dyn(self.A, self.B)
        self.cost = set_cost(self.Q, self.R)
        dx, du = self.B.shape
        self.Cs = {}
        self.alg = {}
        if 'Algorithm 1' in par['alg']:
            my_gpc = GPC(A=self.A, B=self.B, Ref=Ref, Q=self.Q, R=self.R, h=par['gpc_h'], mw=par['gpc_mw'],
                         mr=par['gpc_mr'], lr_w=par['gpc_lrw'], lr_r=par['gpc_lrr'])
            self.Cs['Algorithm 1'] = my_gpc.run_gpc()
            self.alg['Algorithm 1'] = my_gpc

        if 'online_learning_fixed_ff' in par['alg']:
            my_online_learning_fixed_ff = Online_Learning_Fixed_FF(A=self.A, B=self.B, RA=self.ref_A, RC=self.ref_C, Ref=Ref,
                                              Q=self.Q, R=self.R, h=par['gpc_h'], mw=par['gpc_mw'], lr_w=par['gpc_lrw'])
            self.Cs['online_learning_fixed_ff'] = my_online_learning_fixed_ff.run_online_learning_fixed_ff()
            self.alg['online_learning_fixed_ff'] = my_online_learning_fixed_ff


        if 'Hinf Control' in par['alg']:
            my_hinf = HINF(A=self.A, B=self.B, Q=self.Q, R=self.R, RA=self.ref_A, RC=self.ref_C,
                           Ref=Ref, gamma=par['hinf_gamma'], l=self.l)
            self.Cs['Hinf Control'] = my_hinf.run_hinf()
            self.alg['Hinf Control'] = my_hinf

        if 'LQR' in par['alg']:
            my_lqr = LQR(A=self.A, B=self.B, Q=self.Q, R=self.R, RA=self.ref_A, RC=self.ref_C,
                         Ref=Ref, l=self.l)
            self.Cs['LQR'] = my_lqr.run_lqr()
            self.alg['LQR'] = my_lqr

        if 'LQR_Random walk' in par['alg']:
            my_lqr_random_walk = LQR_Random_walk(A=self.A, B=self.B, Q=self.Q, R=self.R, RA=self.ref_A, RC=self.ref_C,
                                                 Ref=Ref, l=self.l)
            self.Cs['LQR_Random walk'] = my_lqr_random_walk.run_lqr_ext()
            self.alg['LQR_Random walk'] = my_lqr_random_walk

        if 'PID' in par['alg']:
            my_pid = PID(par['KP'], par['KI'], par['KD'], par['Ref'], dx)
            self.Cs['PID'] = my_pid.run_pid()
            self.alg['PID'] = my_pid

        if 'ZERO' in par['alg']:
            my_zero = ZERO(du)
            self.Cs['ZERO'] = my_zero.run_zero()
            self.alg['ZERO'] = my_zero

        print("Running controllers")
        self.traces = {Cstr: list(zip(*eval_with_ref(self.dyn, C, self.cost, par['W'], Ref, dx, par['T'])))
                       for Cstr, C in self.Cs.items()}
        self.xss = valmap(lambda x: x[0], self.traces)
        self.uss = valmap(lambda x: x[1], self.traces)
        self.costss = valmap(lambda x: x[3], self.traces)
        self.W_label = par['W_label']
        self.T = par['T']

        print('Running is done.')

    def tracking_monte_carlo(self, par: dict):
        Ref = np.array(list(gen_ref(self.ref_A, self.ref_C, self.ref_init, int(par['T']))))
        Ref = Ref.reshape([int(par['T']), 2])
        self.Ref = Ref
        self.dyn = set_dyn(self.A, self.B)
        self.cost = set_cost(self.Q, self.R)
        dx, du = self.B.shape
        self.Cs = {}
        self.alg = {}
        if 'Algorithm 1' in par['alg']:
            my_gpc = GPC(A=self.A, B=self.B, Ref=Ref, Q=self.Q, R=self.R, h=par['gpc_h'], mw=par['gpc_mw'],
                         mr=par['gpc_mr'], lr_w=par['gpc_lrw'], lr_r=par['gpc_lrr'])
            self.Cs['Algorithm 1'] = my_gpc.run_gpc()
            self.alg['Algorithm 1'] = my_gpc

        if 'online_learning_fixed_ff' in par['alg']:
            my_online_learning_fixed_ff = Online_Learning_Fixed_FF(A=self.A, B=self.B, RA=self.ref_A, RC=self.ref_C,
                                                                   Ref=Ref,
                                                                   Q=self.Q, R=self.R, h=par['gpc_h'], mw=par['gpc_mw'],
                                                                   lr_w=par['gpc_lrw'])
            self.Cs['online_learning_fixed_ff'] = my_online_learning_fixed_ff.run_online_learning_fixed_ff()
            self.alg['online_learning_fixed_ff'] = my_online_learning_fixed_ff

        if 'Hinf Control' in par['alg']:
            my_hinf = HINF(A=self.A, B=self.B, Q=self.Q, R=self.R, RA=self.ref_A, RC=self.ref_C,
                           Ref=Ref, gamma=par['hinf_gamma'], l=self.l)
            self.Cs['Hinf Control'] = my_hinf.run_hinf()
            self.alg['Hinf Control'] = my_hinf

        if 'LQR' in par['alg']:
            my_lqr = LQR(A=self.A, B=self.B, Q=self.Q, R=self.R, RA=self.ref_A, RC=self.ref_C,
                         Ref=Ref, l=self.l)
            self.Cs['LQR'] = my_lqr.run_lqr()
            self.alg['LQR'] = my_lqr

        if 'LQR_Random walk' in par['alg']:
            my_lqr_random_walk = LQR_Random_walk(A=self.A, B=self.B, Q=self.Q, R=self.R, RA=self.ref_A, RC=self.ref_C,
                                                 Ref=Ref, l=self.l)
            self.Cs['LQR_Random walk'] = my_lqr_random_walk.run_lqr_ext()
            self.alg['LQR_Random walk'] = my_lqr_random_walk

        if 'PID' in par['alg']:
            my_pid = PID(par['KP'], par['KI'], par['KD'], par['Ref'], dx)
            self.Cs['PID'] = my_pid.run_pid()
            self.alg['PID'] = my_pid

        if 'ZERO' in par['alg']:
            my_zero = ZERO(du)
            self.Cs['ZERO'] = my_zero.run_zero()
            self.alg['ZERO'] = my_zero

        print("Running controllers")
        self.traces = {Cstr: list(zip(*eval_with_ref(self.dyn, C, self.cost, par['W'], Ref, dx, par['T'])))
                       for Cstr, C in self.Cs.items()}
        self.xss = valmap(lambda x: x[0], self.traces)
        self.uss = valmap(lambda x: x[1], self.traces)
        self.costss = valmap(lambda x: x[3], self.traces)
        self.W_label = par['W_label']
        self.T = par['T']
        return self.costss
        print('Running is done.')

    def show_summary(self, cumcmin=0.0, cumcmax=200.0, cmin=0.0, cmax=100.0, emin=-2.5, emax=2.5):
        # Printing
        print('The summary for', self.W_label,':')
        report_ave_costs(self.costss)
        # statplot_state_tracking(costss, xss, Ref, W, tmax=T, xmax=4)
        plot_costs(self.costss, logscale=False, cumcmin=cumcmin, cumcmax=cumcmax, tmax=self.T)
        plot_immediate_cost(self.costss, cmin=cmin, cmax=cmax, tmax=self.T)
        plot_errors(self.xss, self.Ref, '', emin=emin, emax=emax, tmax=self.T, dx=self.B.shape[0])
    def show_summary2(self, my_data, is_stable_q, n_monte_carlo, cumcmin=0.0, cumcmax=200.0, cmin=0.0, cmax=100.0, emin=-2.5, emax=2.5):
        # Printing
        print('The summary for', self.W_label,':')
        # report_ave_costs(self.costss)
        # statplot_state_tracking(costss, xss, Ref, W, tmax=T, xmax=4)
        plot_costs2(my_data, is_stable_q, n_monte_carlo, logscale=False, cumcmin=cumcmin, cumcmax=cumcmax, tmax=self.T)



    def evaluat_alg(self, new_Ref, new_W, new_T,
                    cumcmin=0.0, cumcmax=200.0, cmin=0.0, cmax=100.0, emin=-2.5, emax=2.5, xmin=-2.5, xmax=2.5):
        print("Evaluating controllers")
        dx, du = self.B.shape
        for _, my_alg in self.alg.items():
            my_alg.prepare_for_evaluation(new_Ref)

        traces = {Cstr: list(zip(*eval_with_ref(self.dyn, C, self.cost, new_W, new_Ref, dx, new_T)))
                       for Cstr, C in self.Cs.items()}
        xss = valmap(lambda x: x[0], traces)
        uss = valmap(lambda x: x[1], traces)
        costss = valmap(lambda x: x[3], traces)
        print('The summary of evaluation for ', self.W_label, ':')
        report_ave_costs(costss)
        plot_costs(costss, logscale=False, cumcmin=cumcmin, cumcmax=cumcmax, tmax=new_T)
        plot_immediate_cost(costss, cmin=cmin, cmax=cmax, tmax=new_T)
        plot_errors(xss, new_Ref, '', emin=emin, emax=emax, tmax=new_T, dx=dx, use_marker=True)
        plot_states(xss, new_Ref, '', xmin=xmin, xmax=xmax, tmax=new_T, dx=dx, use_marker=True)
        plot_ref(new_Ref, '', xmin=xmin, xmax=xmax, tmax=new_T, dx=dx)
        print('Running is done.')

class tracking_changing_reference_alorithm:
    def __init__(self, general_par: dict):
        self.A = general_par['A']
        self.B = general_par['B']
        self.Q = general_par['Q']
        self.R = general_par['R']
        self.l = general_par['l']
        self.T_change = general_par['T_change']
        self.ref_A1 = general_par['ref_A1']
        self.ref_C1 = general_par['ref_C1']
        self.ref_init1 = general_par['ref_init1']
        self.ref_A2 = general_par['ref_A2']
        self.ref_C2 = general_par['ref_C2']
        self.ref_init2 = general_par['ref_init2']

    def tracking(self, par: dict):
        Ref1 = np.array(list(gen_ref(self.ref_A1, self.ref_C1, self.ref_init1, self.T_change)))
        Ref1 = Ref1.reshape([self.T_change, 2])
        Ref2 = np.array(list(gen_ref(self.ref_A2, self.ref_C2, self.ref_init2, int(par['T']) - self.T_change)))
        Ref2 = Ref2.reshape([int(par['T']) - self.T_change, 2])
        Ref = np.concatenate((Ref1, Ref2), axis=0)
        self.Ref = Ref
        self.dyn = set_dyn(self.A, self.B)
        self.cost = set_cost(self.Q, self.R)
        dx, du = self.B.shape
        self.Cs = {}
        self.alg = {}
        if 'Algorithm 1' in par['alg']:
            my_gpc = GPC(A=self.A, B=self.B, Ref=Ref, Q=self.Q, R=self.R, h=par['gpc_h'], mw=par['gpc_mw'],
                         mr=par['gpc_mr'], lr_w=par['gpc_lrw'], lr_r=par['gpc_lrr'])
            self.Cs['Algorithm 1'] = my_gpc.run_gpc()
            self.alg['Algorithm 1'] = my_gpc

        if 'online_learning_fixed_ff' in par['alg']:
            my_online_learning_fixed_ff = Online_Learning_Fixed_FF(A=self.A, B=self.B, RA=self.ref_A1, RC=self.ref_C1, Ref=Ref,
                                              Q=self.Q, R=self.R, h=par['gpc_h'], mw=par['gpc_mw'], lr_w=par['gpc_lrw'],
                                                                   RA2=self.ref_A2, RC2=self.ref_C2,
                                                                   T_change=self.T_change)
            self.Cs['online_learning_fixed_ff'] = my_online_learning_fixed_ff.run_online_learning_fixed_ff()
            self.alg['online_learning_fixed_ff'] = my_online_learning_fixed_ff


        if 'Hinf Control' in par['alg']:
            my_hinf = HINF(A=self.A, B=self.B, Q=self.Q, R=self.R, RA=self.ref_A1, RC=self.ref_C1,
                           Ref=Ref, gamma=par['hinf_gamma'], l=self.l,
                           RA2=self.ref_A2, RC2=self.ref_C2, T_change=self.T_change)
            self.Cs['Hinf Control'] = my_hinf.run_hinf()
            self.alg['Hinf Control'] = my_hinf

        if 'LQR' in par['alg']:
            my_lqr = LQR(A=self.A, B=self.B, Q=self.Q, R=self.R, RA=self.ref_A1, RC=self.ref_C1,
                         Ref=Ref, l=self.l,
                         RA2=self.ref_A2, RC2=self.ref_C2, T_change=self.T_change)
            self.Cs['LQR'] = my_lqr.run_lqr()
            self.alg['LQR'] = my_lqr

        if 'LQR_Random walk' in par['alg']:
            my_lqr_random_walk = LQR_Random_walk(A=self.A, B=self.B, Q=self.Q, R=self.R, RA=self.ref_A1, RC=self.ref_C1,
                                                 Ref=Ref, l=self.l,
                                                 RA2=self.ref_A2, RC2=self.ref_C2, T_change=self.T_change)
            self.Cs['LQR_Random walk'] = my_lqr_random_walk.run_lqr_ext()
            self.alg['LQR_Random walk'] = my_lqr_random_walk

        if 'PID' in par['alg']:
            my_pid = PID(par['KP'], par['KI'], par['KD'], par['Ref'], dx)
            self.Cs['PID'] = my_pid.run_pid()
            self.alg['PID'] = my_pid

        if 'ZERO' in par['alg']:
            my_zero = ZERO(du)
            self.Cs['ZERO'] = my_zero.run_zero()
            self.alg['ZERO'] = my_zero

        print("Running controllers")
        self.traces = {Cstr: list(zip(*eval_with_ref(self.dyn, C, self.cost, par['W'], Ref, dx, par['T'])))
                       for Cstr, C in self.Cs.items()}
        self.xss = valmap(lambda x: x[0], self.traces)
        self.uss = valmap(lambda x: x[1], self.traces)
        self.costss = valmap(lambda x: x[3], self.traces)
        self.W_label = par['W_label']
        self.T = par['T']

        print('Running is done.')

    def show_summary(self, T_change, cumcmin=0.0, cumcmax=200.0, cmin=0.0, cmax=100.0, emin=-2.5, emax=2.5):
        # Printing
        print('The summary for', self.W_label,':')
        report_ave_costs(self.costss)
        # statplot_state_tracking(costss, xss, Ref, W, tmax=T, xmax=4)
        plot_costs(self.costss, logscale=False, cumcmin=cumcmin, cumcmax=cumcmax, tmax=self.T)
        plot_immediate_cost(self.costss, cmin=cmin, cmax=cmax, tmax=self.T)
        # plot_errors(self.xss, self.Ref, '', emin=emin, emax=emax, tmax=self.T, dx=self.B.shape[0])
        plot_errors_details(self.xss, self.Ref, '', emin=emin, emax=emax, T=T_change, trange=30, dx=self.B.shape[0])
        plot_errors_details(self.xss, self.Ref, '', emin=emin, emax=emax, T=self.T, trange=30, dx=self.B.shape[0])
        plot_ref_details(self.Ref, '', T=T_change, trange=30, dx=self.B.shape[0])
        plot_ref_details(self.Ref, '', T=self.T, trange=30, dx=self.B.shape[0])

    def evaluat_alg(self, new_Ref, new_W, new_T, T_change=None,
                    cumcmin=0.0, cumcmax=200.0, cmin=0.0, cmax=100.0, emin=-2.5, emax=2.5, xmin=-2.5, xmax=2.5):
        print("Evaluating controllers")
        dx, du = self.B.shape
        for _, my_alg in self.alg.items():
            my_alg.prepare_for_evaluation(new_Ref, T_change)

        traces = {Cstr: list(zip(*eval_with_ref(self.dyn, C, self.cost, new_W, new_Ref, dx, new_T)))
                       for Cstr, C in self.Cs.items()}
        xss = valmap(lambda x: x[0], traces)
        uss = valmap(lambda x: x[1], traces)
        costss = valmap(lambda x: x[3], traces)
        print('The summary of evaluation for ', self.W_label, ':')
        report_ave_costs(costss)
        plot_costs(costss, logscale=False, cumcmin=cumcmin, cumcmax=cumcmax, tmax=new_T)
        plot_immediate_cost(costss, cmin=cmin, cmax=cmax, tmax=new_T)
        plot_errors(xss, new_Ref, '', emin=emin, emax=emax, tmax=new_T, dx=dx, use_marker=True)
        plot_states(xss, new_Ref, '', xmin=xmin, xmax=xmax, tmax=new_T, dx=dx, use_marker=True)
        print('Running is done.')



