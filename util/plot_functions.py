import jax.numpy as np
import matplotlib.pyplot as plt
from IPython import display
from toolz.dicttoolz import valmap, itemmap
from itertools import chain



def liveplot(costss, xss, W, cmax=30, cumcmax=15, wmax=2, xmax=20, logcmax=100, logcumcmax=1000):
    cummean = lambda x: np.cumsum(np.array(x))/np.arange(1, len(x)+1)
    cumcostss = valmap(cummean, costss)

    plt.style.use('seaborn')
    colors = {
        "Zero Control": "gray",
        "LQR / H2": "green",
        "LQR_Random walk": "teal",
        "Optimal LQG for GRW": "aqua",
        "Robust / Hinf Control": "orange",
        "GPC": "red",
        "online_learning_fixed_ff": "violet",
    }
    fig, ax = plt.subplots(3, 2, figsize=(21, 12))

    costssline = {}
    for Cstr, costs in costss.items():
        costssline[Cstr], = ax[0, 0].plot([], label=Cstr, color=colors[Cstr])
    ax[0, 0].set_xlabel("Time")
    ax[0, 0].set_ylabel("Instantaneous Cost")
    ax[0, 0].set_ylim([-1, cmax])
    ax[0, 0].set_xlim([0, 100])
    ax[0, 0].legend()

    cumcostssline = {}
    for Cstr, costs in cumcostss.items():
        cumcostssline[Cstr], = ax[0, 1].plot([], label=Cstr, color=colors[Cstr])
    ax[0, 1].set_xlabel("Time")
    ax[0, 1].set_ylabel("Average Cost")
    ax[0, 1].set_ylim([-1, cumcmax])
    ax[0, 1].set_xlim([0, 100])
    ax[0, 1].legend()

    perturbline, = ax[1, 0].plot([])
    ax[1, 0].set_xlabel("Time")
    ax[1, 0].set_ylabel("Perturbation")
    ax[1, 0].set_ylim([-wmax, wmax])
    ax[1, 0].set_xlim([0, 100])

    pointssline, trailssline = {}, {}
    for Cstr, C in xss.items():
        pointssline[Cstr], = ax[1,1].plot([], [], label=Cstr, color=colors[Cstr], ms=20, marker='s')
        trailssline[Cstr], = ax[1,1].plot([], [], label=Cstr, color=colors[Cstr], lw=2)
    ax[1, 1].set_xlabel("Position")
    ax[1, 1].set_ylabel("")
    ax[1, 1].set_ylim([-1, 6])
    ax[1, 1].set_xlim([-xmax, xmax])
    ax[1, 1].legend()

    logcostssline = {}
    for Cstr, costs in costss.items():
        logcostssline[Cstr], = ax[2, 0].plot([1], label=Cstr, color=colors[Cstr])
    ax[2, 0].set_xlabel("Time")
    ax[2, 0].set_ylabel("Instantaneous Cost (Log Scale)")
    ax[2, 0].set_xlim([0, 100])
    ax[2, 0].set_ylim([0.1, logcmax])
    ax[2, 0].set_yscale('log')
    ax[2, 0].legend()

    logcumcostssline = {}
    for Cstr, costs in cumcostss.items():
        logcumcostssline[Cstr], = ax[2, 1].plot([1], label=Cstr, color=colors[Cstr])
    ax[2, 1].set_xlabel("Time")
    ax[2, 1].set_ylabel("Average Cost (Log Scale)")
    ax[2, 1].set_xlim([0, 100])
    ax[2, 1].set_ylim([0.1, logcumcmax])
    ax[2, 1].set_yscale('log')
    ax[2, 1].legend()

    def livedraw(t):
        for Cstr, costsline in costssline.items():
            costsline.set_data(np.arange(t), costss[Cstr][:t])
        for Cstr, cumcostsline in cumcostssline.items():
            cumcostsline.set_data(np.arange(t), cumcostss[Cstr][:t])
        perturbline.set_data(np.arange(t), W[:t, 0])
        for i, (Cstr, pointsline) in enumerate(pointssline.items()):
            pointsline.set_data(xss[Cstr][t][0], i)
        for i, (Cstr, trailsline) in enumerate(trailssline.items()):
            trailsline.set_data(list(map(lambda x: x[0], xss[Cstr][max(t-10, 0):t])), i)
        for Cstr, logcostsline in logcostssline.items():
            logcostsline.set_data(np.arange(t), costss[Cstr][:t])
        for Cstr, logcumcostsline in logcumcostssline.items():
            logcumcostsline.set_data(np.arange(t), cumcostss[Cstr][:t])
        return chain(costssline.values(), cumcostssline.values(), [perturbline], pointssline.values(), trailssline.values(), logcostssline.values(), logcumcostssline.values())

    print("🧛 reanimating :) meanwhile...")
    livedraw(99)
    plt.show()

    from matplotlib import animation
    anim = animation.FuncAnimation(fig, livedraw, frames=100, interval=50, blit=True)
    from IPython.display import HTML
    display.clear_output(wait=True)
    return HTML(anim.to_html5_video())


def statplot_state_tracking(costss, xss, Ref, W, wmax=2, xmax=10, logcmax=100, logcumcmax=1000, tmax=100):
    cummean = lambda x: np.cumsum(np.array(x)) / np.arange(1, len(x) + 1)
    cumcostss = valmap(cummean, costss)

    plt.style.use('seaborn')
    colors = {
        "Zero Control": "gray",
        "LQR": "green",
        "LQR_Random walk": "teal",
        "PID": "aqua",
        "Hinf Control": "orange",
        "GPC": "red",
        "online_learning_fixed_ff": "violet",
    }

    fig, ax = plt.subplots(4, 2, figsize=(21, 12))

    for Cstr, C in xss.items():
        ax[0, 0].plot(np.arange(tmax), (np.asarray(xss[Cstr]))[:, 0] , label=Cstr, color=colors[Cstr], )
    ax[0, 0].plot(np.arange(tmax), Ref[:, 0], label='Reference', color='k', )
    ax[0, 0].set_xlabel("")
    ax[0, 0].set_ylabel("x_1")
    ax[0, 0].set_ylim([-xmax, xmax])
    ax[0, 0].set_xlim([0, tmax])
    ax[0, 0].legend()

    for Cstr, C in xss.items():
        ax[1, 0].plot(np.arange(tmax), (np.asarray(xss[Cstr]))[:, 1], label=Cstr, color=colors[Cstr], )
    ax[1, 0].plot(np.arange(tmax), Ref[:, 1], label='Reference', color='k', )
    ax[1, 0].set_xlabel("")
    ax[1, 0].set_ylabel("x_2")
    ax[1, 0].set_ylim([-xmax, xmax])
    ax[1, 0].set_xlim([0, tmax])
    ax[1, 0].legend()

    for Cstr, C in xss.items():
        ax[2, 0].plot(np.arange(tmax), (np.asarray(xss[Cstr]))[:, 0]-Ref[:, 0], label=Cstr, color=colors[Cstr], )
    ax[2, 0].set_xlabel("")
    ax[2, 0].set_ylabel("e_1")
    ax[2, 0].set_ylim([-xmax, xmax])
    ax[2, 0].set_xlim([0, tmax])
    ax[2, 0].legend()

    for Cstr, C in xss.items():
        ax[3, 0].plot(np.arange(tmax), (np.asarray(xss[Cstr]))[:, 1]-Ref[:, 1], label=Cstr, color=colors[Cstr], )
    ax[3, 0].set_xlabel("")
    ax[3, 0].set_ylabel("e_2")
    ax[3, 0].set_ylim([-xmax, xmax])
    ax[3, 0].set_xlim([0, tmax])
    ax[3, 0].legend()

    ax[0, 1].plot(np.arange(tmax), W[:, 0])
    ax[0, 1].set_xlabel("Time")
    ax[0, 1].set_ylabel("w_1")
    ax[0, 1].set_ylim([-wmax, wmax])
    ax[0, 1].set_xlim([0, tmax])

    ax[1, 1].plot(np.arange(tmax), W[:, 1])
    ax[1, 1].set_xlabel("Time")
    ax[1, 1].set_ylabel("w_2")
    ax[1, 1].set_ylim([-wmax, wmax])
    ax[1, 1].set_xlim([0, tmax])

    for Cstr, costs in costss.items():
        ax[2, 1].plot(np.arange(tmax), costss[Cstr], label=Cstr, color=colors[Cstr])
    ax[2, 1].set_xlabel("Time")
    ax[2, 1].set_ylabel("c_t (Log Scale)")
    ax[2, 1].set_xlim([0, tmax])
    ax[2, 1].set_ylim([0.1, logcmax])
    ax[2, 1].set_yscale('log')
    ax[2, 1].legend()

    for Cstr, costs in cumcostss.items():
        ax[3, 1].plot(np.arange(tmax), cumcostss[Cstr], label=Cstr, color=colors[Cstr])
    ax[3, 1].set_xlabel("Time")
    ax[3, 1].set_ylabel("Average Cost (Log Scale)")
    ax[3, 1].set_xlim([0, tmax])
    ax[3, 1].set_ylim([0.1, logcumcmax])
    ax[3, 1].set_yscale('log')
    ax[3, 1].legend()
    plt.show()
    return plt

def statplot(costss, xss, W, cmax=30, cumcmax=15, wmax=2, xmax=20, logcmax=100, logcumcmax=1000, tmax=100):
    cummean = lambda x: np.cumsum(np.array(x)) / np.arange(1, len(x) + 1)
    cumcostss = valmap(cummean, costss)

    plt.style.use('seaborn')
    colors = {
        "Zero Control": "gray",
        "LQR / H2": "green",
        "LQR_Random walk": "teal",
        "Optimal LQG for GRW": "aqua",
        "Robust / Hinf Control": "orange",
        "DRC": "red",
        "LQG": "green",
        "GPC": "teal",
        "online_learning_fixed_ff": "violet",
    }

    fig, ax = plt.subplots(3, 2, figsize=(21, 12))

    costssline = {}
    for Cstr, costs in costss.items():
        costssline[Cstr], = ax[0, 0].plot(np.arange(tmax), costss[Cstr], label=Cstr, color=colors[Cstr])
    ax[0, 0].set_xlabel("Time")
    ax[0, 0].set_ylabel("Instantaneous Cost")
    ax[0, 0].set_ylim([-1, cmax])
    ax[0, 0].set_xlim([0, tmax])
    ax[0, 0].legend()

    for Cstr, costs in cumcostss.items():
        ax[0, 1].plot(np.arange(tmax), cumcostss[Cstr], label=Cstr, color=colors[Cstr])
    ax[0, 1].set_xlabel("Time")
    ax[0, 1].set_ylabel("Average Cost")
    ax[0, 1].set_ylim([-1, cumcmax])
    ax[0, 1].set_xlim([0, tmax])
    ax[0, 1].legend()

    ax[1, 0].plot(np.arange(tmax), W[:, 0])
    ax[1, 0].set_xlabel("Time")
    ax[1, 0].set_ylabel("Perturbation")
    ax[1, 0].set_ylim([-wmax, wmax])
    ax[1, 0].set_xlim([0, tmax])


    for Cstr, C in xss.items():
        ax[1, 1].plot(np.arange(tmax), xss[Cstr][:], 0, label=Cstr, color=colors[Cstr], )
        # pointssline[Cstr], = ax[1, 1].plot([], [], label=Cstr, color=colors[Cstr], ms=20, marker='s')
        # trailssline[Cstr], = ax[1, 1].plot([], [], label=Cstr, color=colors[Cstr], lw=2)
    ax[1, 1].set_xlabel("Position")
    ax[1, 1].set_ylabel("")
    ax[1, 1].set_ylim([-10, 10])
    ax[1, 1].set_xlim([0, tmax])
    ax[1, 1].legend()

    for Cstr, costs in costss.items():
        ax[2, 0].plot(np.arange(tmax), costss[Cstr], label=Cstr, color=colors[Cstr])
    ax[2, 0].set_xlabel("Time")
    ax[2, 0].set_ylabel("Instantaneous Cost (Log Scale)")
    ax[2, 0].set_xlim([0, tmax])
    ax[2, 0].set_ylim([0.1, logcmax])
    ax[2, 0].set_yscale('log')
    ax[2, 0].legend()

    for Cstr, costs in cumcostss.items():
        ax[2, 1].plot(np.arange(tmax), cumcostss[Cstr], label=Cstr, color=colors[Cstr])
    ax[2, 1].set_xlabel("Time")
    ax[2, 1].set_ylabel("Average Cost (Log Scale)")
    ax[2, 1].set_xlim([0, tmax])
    ax[2, 1].set_ylim([0.1, logcumcmax])
    ax[2, 1].set_yscale('log')
    ax[2, 1].legend()
    plt.show()
    return plt


def plot_costs(costss, logscale=True, logcumcmin=0.1, logcumcmax=1000, cumcmin=0, cumcmax=20, tmax=100, use_marker=False):
    colors, labels, markers, _label_fontsize, _legend_fontsize = set_properties(use_marker)
    cummean = lambda x: np.cumsum(np.array(x)) / np.arange(1, len(x) + 1)
    cumcostss = valmap(cummean, costss)

    plt.style.use('seaborn')
    plt.figure()
    ax = plt.gca()
    for Cstr, costs in cumcostss.items():
        plt.plot(np.arange(tmax), cumcostss[Cstr], label=labels[Cstr], color=colors[Cstr], marker=markers[Cstr])
    ax.set_xlabel("Time", fontsize=_label_fontsize)
    if logscale:
        plt.axis([0, tmax-1, logcumcmin, logcumcmax])
        ax.set_ylabel("Average Cost (Log Scale)", fontsize=_label_fontsize)
        plt.yscale('log')
    else:
        plt.axis([0, tmax-1, cumcmin, cumcmax])
        ax.set_ylabel("Average Cost", fontsize=_label_fontsize)

    plt.legend(fontsize=_legend_fontsize)
    plt.show()
    return plt


def plot_costs2(costss, is_stable_q, n_monte_carlo, logscale=True, logcumcmin=0.1, logcumcmax=1000, cumcmin=0, cumcmax=20, tmax=100, use_marker=False):
    colors, labels, markers, _label_fontsize, _legend_fontsize = set_properties(use_marker)
    cummean = lambda x: np.cumsum(np.array(x)) / np.arange(1, len(x) + 1)

    plt.style.use('seaborn')
    plt.figure()
    ax = plt.gca()
    for Cstr, costs in costss.items():
        if Cstr == 'Q learning':
            stable_ind = 0
            cum_c = np.zeros([sum(is_stable_q), tmax])
            for i in range(n_monte_carlo):
                if is_stable_q[i]:
                    cum_c[stable_ind, :] = cummean(costs[i, :])
                    stable_ind = stable_ind + 1
        else:
            cum_c = np.zeros([n_monte_carlo, tmax])
            for i in range(n_monte_carlo):
                cum_c= cum_c.at[i, :].set(cummean(costs[i,:]))
                # cum_c[i, :] = cummean(costs[i,:])

        plt.plot(np.arange(tmax), np.nanmean(cum_c, axis=0), label=labels[Cstr], color=colors[Cstr], marker=markers[Cstr])
        # ax.fill_between(np.arange(tmax), np.percentile(cum_c, 25, axis=0), np.percentile(cum_c, 75, axis=0),
        #                 alpha=0.25)

    ax.set_xlabel("Time", fontsize=_label_fontsize)
    if logscale:
        plt.axis([0, tmax-1, logcumcmin, logcumcmax])
        ax.set_ylabel("Average Cost (Log Scale)", fontsize=_label_fontsize)
        plt.yscale('log')
    else:
        plt.axis([0, tmax-1, cumcmin, cumcmax])
        ax.set_ylabel("Average Cost", fontsize=_label_fontsize)

    plt.legend(fontsize=_legend_fontsize)
    plt.show()
    return plt


def plot_immediate_cost(costss, cmin=0, cmax=100, tmax=100, use_marker=False):
    colors, labels, markers, _label_fontsize, _legend_fontsize = set_properties(use_marker)
    my_c = lambda x: x
    costss = valmap(my_c, costss)

    plt.style.use('seaborn')
    plt.figure()
    ax = plt.gca()
    for Cstr, costs in costss.items():
        plt.plot(np.arange(tmax), costss[Cstr], label=labels[Cstr], color=colors[Cstr], marker=markers[Cstr])
    ax.set_xlabel("Time", fontsize=_label_fontsize)

    plt.axis([0, tmax-1, cmin, cmax])
    ax.set_ylabel("Immediate Cost", fontsize=_label_fontsize)

    plt.legend(fontsize=_legend_fontsize)
    plt.show()
    return plt


def plot_errors(xss, Ref, title, emin=-10.0, emax=10.0, tmax=100, dx=2, use_marker=False):
    colors, labels, markers, _label_fontsize, _legend_fontsize = set_properties(use_marker)

    for i in range(dx):
        plt.style.use('seaborn')
        plt.figure()
        ax = plt.gca()
        for Cstr, C in xss.items():
            plt.plot(np.arange(tmax), (np.asarray(xss[Cstr]))[:, i]-Ref[:, i], label=labels[Cstr], color=colors[Cstr],
                     marker=markers[Cstr])
        plt.axis([0, tmax-1, emin, emax])
        ax.set_xlabel("Time", fontsize=_label_fontsize)
        ax.set_ylabel('$e_{:1}$'.format(i+1)+'$_k$', fontsize=_label_fontsize)
        plt.legend(fontsize=_legend_fontsize)
        plt.title(title)
        plt.show()
    return plt


def plot_errors_details(xss, Ref, title, emin=-10.0, emax=10.0, trange=30, T=5000, dx=2, use_marker=False):
    colors, labels, markers, _label_fontsize, _legend_fontsize = set_properties(use_marker)

    for i in range(dx):
        plt.style.use('seaborn')
        plt.figure()
        ax = plt.gca()
        for Cstr, C in xss.items():
            plt.plot(np.arange(T-trange, T),
                     (np.asarray(xss[Cstr]))[T-trange:T, i]-Ref[T-trange:T, i],
                     label=labels[Cstr], color=colors[Cstr],
                     marker=markers[Cstr])
        plt.axis([T-trange, T-1, emin, emax])
        ax.set_xlabel("Time", fontsize=_label_fontsize)
        ax.set_ylabel('$e_{:1}$'.format(i+1)+'$_k$', fontsize=_label_fontsize)
        plt.legend(fontsize=_legend_fontsize)
        plt.title(title)
        plt.show()
    return plt

def plot_states(xss, Ref, title, xmin=-10.0, xmax=10.0, tmax=100, dx=2, use_marker=False):
    colors, labels, markers, _label_fontsize, _legend_fontsize = set_properties(use_marker)
    for i in range(dx):
        plt.style.use('seaborn')
        plt.figure()
        ax = plt.gca()
        plt.plot(np.arange(tmax), Ref[:, i], label=labels['Reference'], color=colors['Reference'],
                 marker=markers['Reference'])
        for Cstr, C in xss.items():
            plt.plot(np.arange(tmax), (np.asarray(xss[Cstr]))[:, i], label=labels[Cstr], color=colors[Cstr],
                     marker=markers[Cstr])
        plt.axis([0, tmax-1, xmin, xmax])
        ax.set_xlabel("Time", fontsize=_label_fontsize)
        ax.set_ylabel('$x_{:1}$'.format(i+1)+'$_k$', fontsize=_label_fontsize)
        plt.legend(fontsize=_legend_fontsize)
        plt.title(title)
        plt.show()
    return plt


def plot_ref(Ref, title, xmin=-10.0, xmax=10.0, tmax=100, dx=2, use_marker=False):
    colors, labels, markers, _label_fontsize, _legend_fontsize = set_properties(use_marker)
    for i in range(dx):
        plt.style.use('seaborn')
        plt.figure()
        ax = plt.gca()
        plt.plot(np.arange(tmax), Ref[:, i], label=labels['Reference'], color=colors['Reference'],
                 marker=markers['Reference'])

        plt.axis([0, tmax - 1, xmin, xmax])
        ax.set_xlabel("Time", fontsize=_label_fontsize)
        ax.set_ylabel('$r_{:1}$'.format(i + 1)+'$_k$', fontsize=_label_fontsize)
        plt.legend(fontsize=_legend_fontsize)
        plt.title(title)
        plt.show()
    return plt

def plot_ref_details(Ref, title, xmin=-10.0, xmax=10.0, trange=30, T=5000, dx=2, use_marker=False):
    colors, labels, markers, _label_fontsize, _legend_fontsize = set_properties(use_marker)
    for i in range(dx):
        plt.style.use('seaborn')
        plt.figure()
        ax = plt.gca()
        plt.plot(np.arange(T-trange, T), Ref[T-trange:T, i], label=labels['Reference'], color=colors['Reference'],
                 marker=markers['Reference'])

        plt.axis([T-trange, T-1, xmin, xmax])
        ax.set_xlabel("Time", fontsize=_label_fontsize)
        ax.set_ylabel('$r_{:1}$'.format(i + 1)+'$_k$', fontsize=_label_fontsize)
        plt.legend(fontsize=_legend_fontsize)
        plt.title(title)
        plt.show()
    return plt

def set_properties(use_marker):
    colors = {
        "Zero Control": "gray",
        "LQR": "orange",
        "LQR_Random walk": "teal",
        "PID": "aqua",
        "Hinf Control": "green",
        "Algorithm 1": "red",
        'Reference': 'black',
        "online_learning_fixed_ff": "violet",
        'Q learning': "aqua",
    }
    labels = {
        "Zero Control": 'Zero control',
        "LQR": 'LQR',
        "LQR_Random walk": 'LQR for random walk',
        "PID": 'PID control',
        "Hinf Control": '$H_{\infty}$-control',
        "Algorithm 1": 'Algorithm 1',
        'Reference': 'Reference',
        "online_learning_fixed_ff": 'Online control with fixed feedforward gain',
        'Q learning': "Q learning",
    }
    if use_marker:
        markers = {
            "Zero Control": "*",
            "LQR": "v",
            "LQR_Random walk": "d",
            "PID": 6,
            "Hinf Control": "o",
            "Algorithm 1": "*",
            'Reference': " ",
            "online_learning_fixed_ff": "s",
            'Q learning': 6,
        }
    else:
        markers = {
            "Zero Control": " ",
            "LQR": " ",
            "LQR_Random walk": "",
            "PID": " ",
            "Hinf Control": " ",
            "Algorithm 1": " ",
            'Reference': " ",
            "online_learning_fixed_ff": " ",
            'Q learning': " ",
        }
    _label_fontsize = 16
    _legend_fontsize = 16
    return colors, labels, markers, _label_fontsize, _legend_fontsize
