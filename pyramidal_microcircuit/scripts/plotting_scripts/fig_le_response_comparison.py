import sys

import matplotlib.pyplot as plt
import numpy as np
import src.plot_utils as plot_utils
from src.params import Params

p = Params()


# def phi(x):
#     return p.gamma * np.log(1 + np.exp(p.beta * (x - p.theta)))


def phi(x):
    return x


if __name__ == "__main__":

    out_file = sys.argv[1]
    plot_utils.setup_plt()
    t_stim = 250
    amps = np.concatenate([np.ones(t_stim), np.zeros(t_stim), np.ones(t_stim) * 0.5,
                          np.zeros(t_stim), np.ones(t_stim), np.sin(np.linspace(0, 14*np.pi, 2*t_stim)) + 1])

    voltage_trace_in = [(0, 0)]
    rate_sac = [(0, 0)]
    rate_le = [(0, 0)]
    trace_out_sac = [(0, 0)]
    trace_out_le = [(0, 0)]

    tau_eff = p.C_m_som / (p.g_l + p.g_a + p.g_d)
    p.tau_x = 1
    ux = 0
    uy_sac = 0
    uy_le = 0
    delta_ux = 0
    for i, amp in enumerate(amps):
        delta_ux = -ux + amp

        u_forward = ux + tau_eff * delta_ux
        ux = ux + (p.delta_t/p.tau_x) * delta_ux
        t_ms = i*p.delta_t
        r_forward = phi(u_forward)
        r_x = phi(ux)

        voltage_trace_in.append((t_ms, ux))
        rate_sac.append((t_ms, r_x))
        rate_le.append((t_ms, r_forward))

        delta_uy_sac = -p.g_l_eff * uy_sac + p.g_d * r_x
        uy_sac += p.delta_t * delta_uy_sac
        delta_uy_le = -p.g_l_eff * uy_le + p.g_d * r_forward
        uy_le += p.delta_t * delta_uy_le

        trace_out_le.append((t_ms, uy_le))
        trace_out_sac.append((t_ms, uy_sac))

    alpha_le = 0.7
    lw = 1
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(np.arange(0, len(amps)) * p.delta_t, amps, color="black", linewidth=lw)
    ax[1].plot(*zip(*rate_sac), label=r"$\phi(u_i)$", linewidth=lw)
    ax[1].plot(*zip(*rate_le), label=r"$\phi(\breve{u}_i)$", alpha=alpha_le, linewidth=lw)
    ax[2].plot(*zip(*trace_out_sac), label="Sacramento", linewidth=lw)
    ax[2].plot(*zip(*trace_out_le), label="Latent Equilibrium", alpha=alpha_le, linewidth=lw)
    ax[1].legend()
    # ax[2].legend()

    ax[0].set_title("A: Injected current")
    ax[1].set_title("B: Presynaptic activation")
    ax[2].set_title("C: Postsynaptic somatic potential")

    # ax[0].set_xlabel("Simulation time [ms]")
    # ax[1].set_xlabel("Simulation time [ms]")
    ax[2].set_xlabel("Simulation time [ms]")

    # ax[0].set_ylim(bottom=0)
    # ax[1].set_ylim(bottom=0)
    # ax[2].set_ylim(bottom=0)
    plt.savefig(out_file)
