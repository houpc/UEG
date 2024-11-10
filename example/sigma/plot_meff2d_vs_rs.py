#!/usr/bin/python3
import scienceplots
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib as mat
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cdict = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "magenta": "#EE3377",
    "grey": "#BBBBBB",
}
plt.switch_backend("TkAgg")
plt.style.use(["science", "std-colors"])
# mat.rcParams.update({'font.size': 28})
# mat.rcParams.update({"font.size": 16})
mat.rcParams["font.size"] = 16
mat.rcParams["mathtext.fontset"] = "cm"
mat.rcParams["font.family"] = "Times New Roman"
# size = 36
# sizein = 12

colors = [
    cdict["blue"],
    cdict["red"],
    cdict["orange"],
    cdict["magenta"],
    cdict["cyan"],
    "black",
    cdict["teal"],
    "grey",
]
pts = ['s', '^', 'v', 'p', 's', 'o', 'd']


def errorbar_mvsrs(rs, meff_data, merr, idx, label, ax=plt.gca(), zorder=None):
    if zorder is not None:
        handle = ax.errorbar(rs, meff_data, merr, fmt=pts[idx], capthick=1, capsize=4,
                             ms=5, color=colors[idx], label=label, zorder=zorder)
    else:
        handle = ax.errorbar(rs, meff_data, merr, fmt=pts[idx], capthick=1, capsize=4,
                             ms=5, color=colors[idx], label=label)
    return handle


def plot_mvsrs(rs, meff_data, idx, label, ls='-', ax=plt.gca(), rs_HDL=None, meff_HDL=None, zorder=None):
    # Add data in the high-density limit to the fit, if provided
    if (rs_HDL is not None) and (meff_HDL is not None):
        rslist = pd.unique(np.concatenate([rs, rs_HDL]))
        mdatalist = pd.unique(np.concatenate([meff_data, meff_HDL]))
        P = np.argsort(rslist)
        rs = rslist[P]
        meff_data = mdatalist[P]
    print(rs)
    print(meff_data)

    # mfitfunc = interpolate.PchipInterpolator(rs, meff_data)
    mfitfunc = interpolate.Akima1DInterpolator(rs, meff_data)
    # xgrid = np.arange(0, 6.2, 0.02)
    xgrid = np.arange(0, 1.2, 0.002)
    # ax.plot(rs, meff_data, 'o', ms=10, color=colors[idx])
    # ax.plot(xgrid, mfitfunc(xgrid), ls=ls, lw=2, color=colors[idx], label=label)
    if zorder is not None:
        handle, = ax.plot(xgrid, mfitfunc(xgrid), ls=ls,
                          color=colors[idx], label=label, zorder=zorder)
    else:
        handle, = ax.plot(xgrid, mfitfunc(xgrid), ls=ls,
                          color=colors[idx], label=label)

    yfit = np.ma.masked_invalid(mfitfunc(xgrid))
    # print(yfit)
    print("Turning point: rs = ", xgrid[np.argmin(yfit)])
    print("Effective mass ratio at turning point: ", np.min(yfit))
    return handle


# Taylor series for m* / m in the high-density limit to leading order in rs in 2D
# according to Giuliani & Vignale, Quantum Theory of the Electron Liquid (2008).
# This formula is incorrect; see below.
def high_density_limit_giuliani_vignale(rs):
    # c_2d = 1 / np.pi
    # alpha_2d = 1 / np.sqrt(2.0)  # original formula from G&V
    # # return 1 + c_2d * alpha_2d * rs * np.log(rs) / np.pi
    # return 1 + rs * np.log(rs) / (np.sqrt(2) * np.pi * np.pi)
    raise NotImplementedError("Incorrect formula for 2D high-density limit")


# Taylor series for m* / m in the high-density limit to leading order in rs in 2D
# (c.f. R. Asgari et al., PRB 79, 235324 (2009), doi:10.1103/PhysRevB.79.235324)
def high_density_limit_asgadi(rs):
    return 1 + rs * np.log(rs) / (np.sqrt(2) * np.pi)


# # This fits our data quite well; is there an error in [G&V] / Asgadi?
# # NOTE: No, seems that Asgadi's formula is correct, and G&V's is wrong.
# def high_density_limit_emp(rs):
#     c_2d = 1 / np.pi
#     alpha_2d = np.sqrt(2.0)
#     # alpha_2d = 1 / np.sqrt(2.0)  # original formula from G&V
#     return 1 + c_2d * alpha_2d * rs * np.log(rs) / np.pi
def main():
    fig, ax = plt.subplots(figsize=(5, 5))

    m_VDMC = [1.0, 0.9675, 0.951, 0.952, 0.970]
    m_VDMC_err = [0, 0.0010, 0.001, 0.002, 0.004]
    rs_VDMC = [0, 0.1, 0.3, 0.5, 1]

    rs_DMC = [1, 2, 3, 4, 5]
    m_DMC = [0.918, 0.879, 0.856, 0.842, 0.791]
    m_DMC_err = [0.006, 0.014, 0.014, 0.017, 0.01]

    # f_sigma_G0W0 = np.load("finalized_meff_results/2d/rpa/meff_2d_sigma_G0W0_large_rs.npz")
    f_sigma_G0W0 = np.load("finalized_meff_results/2d/rpa/meff_2d_sigma_G0W0.npz")
    rs_G0W0 = f_sigma_G0W0["rslist"]
    m_G0W0 = f_sigma_G0W0["mefflist"]

    # High-density limit
    rs_HDL_plot = np.linspace(1e-5, 0.387755102040817, num=101)
    meff_HDL_plot = np.array([high_density_limit_asgadi(rs) for rs in rs_HDL_plot])
    # meff_HDL_plot = np.array([high_density_limit_giuliani_vignale(rs) for rs in rs_HDL_plot])

    # # Use exact expression in the high-density limit for RPA(+FL) and RPT fits
    # cutoff_HDL = 0.1
    # rs_HDL = rs_HDL_plot[rs_HDL_plot <= cutoff_HDL]
    # meff_HDL = meff_HDL_plot[rs_HDL_plot <= cutoff_HDL]
    # print(rs_HDL)
    # print(meff_HDL)

    # Use RPA expression to fit the high-density limit
    # (the leading-order HDL expansion converges too slowly in 2D)
    cutoff_HDL = 0.05
    rs_HDL = rs_G0W0[rs_G0W0 <= cutoff_HDL]
    meff_HDL = m_G0W0[rs_G0W0 <= cutoff_HDL]
    print(rs_HDL)
    print(meff_HDL)

    # RPA
    handle1 = plot_mvsrs(rs_G0W0, m_G0W0, 7, r"RPA", '--', ax=ax)
    # handle1 = plot_mvsrs(rs_G0W0, m_G0W0, 2, r"RPA", '--', ax=ax, zorder=10000)

    # High-density limit
    # handle2 = plot_mvsrs(rs_HDL_plot, meff_HDL_plot, 4,
    #            "High-density limit [17]", "--", zorder=1000, ax=ax)
            #    "High-density limit [5]", "--", zorder=1000, ax=ax)

    # # High-density limit reported in Xie et al...this seems very wrong?
    # x_near0, y_near0 = np.array([
    #     [0., 1.],
    #     [0.11224489795918391, 0.9117854622441779],
    #     [0.18367346938775508, 0.8599153140437545],
    #     [0.26530612244897966, 0.8295695130557517],
    #     [0.387755102040817, 0.7985179957657023]]).T
    # ax.plot(x_near0, y_near0, "o--", color=colors[7],
    #         label="High-density limit [14]", zorder=1001)

    # Xie data
    rs_xie = [0.25, 0.5, 1.0, 3.0, 5.0, 10.0]
    # N = 29
    m_xie_N29 = [0.9493450590994533, 0.882463202728022, 0.8075088885361558,
                 0.6624456568146859, 0.5370396071086241, 0.3815095840237416]
    m_xie_N29_err = [0.012286238330642787, 0.014801924633879382, 0.012908898901033937,
                     0.00404809293349585, 0.0052229360479842165, 0.007245691592627612]
    # N = 49
    m_xie_N49 = [0.9697659283566704, 0.9209192115756524, 0.8640514413966484,
                 0.628293959920195, 0.5497751333980478, 0.4046921719547083]
    m_xie_N49_err = [0.021674990617977383, 0.00811342432541565, 0.011749023191077004,
                     0.009302572885392053, 0.0033228606617660738, 0.0022218086489334214]
    # N = 57
    m_xie_N57 = [0.8972423881298187, 0.7974929235841779, 0.7482061436659886,
                 0.5901231585641507, 0.4869549111001582, 0.3877645835702172]
    m_xie_N57_err = [0.025544403804267292, 0.01605710503593936, 0.029722120598245434,
                     0.018958500988648527, 0.017426480375743132, 0.003010231602124497]

    # Plot Xie data
    handle3 = errorbar_mvsrs(rs_xie, m_xie_N29, m_xie_N29_err,
                   0, r"NCT $(N_e = 29)$ [14]", ax=ax)
    handle4 = errorbar_mvsrs(rs_xie, m_xie_N49, m_xie_N49_err,
                   1, r"NCT $(N_e = 49)$ [14]", ax=ax)
    handle5 = errorbar_mvsrs(rs_xie, m_xie_N57, m_xie_N57_err,
                   6, r"NCT $(N_e = 57)$ [14]", ax=ax)

    print("\nPlotting our data")
    handle6 = errorbar_mvsrs(rs_VDMC[1:None], m_VDMC[1:None], m_VDMC_err[1:None],
                   5, "This work", zorder=30, ax=ax)
    # plot_mvsrs(rs_VDMC, m_VDMC, 5, "", '-', ax=ax)
    plot_mvsrs(rs_VDMC, m_VDMC, 5, "", '-', rs_HDL=rs_HDL, meff_HDL=meff_HDL, ax=ax)

    ax.set_xlabel(r"$r_s$")
    ax.set_ylabel(r"$m^*/m$")
    ax.annotate(r"2D", xy=(0.06, 0.9), xycoords="axes fraction")
    # ax.annotate(r"2D", xy=(0.875, 0.9), xycoords="axes fraction")

    # ax.set_xlim(0, 0.25)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.56, 1.12)
    # ax.set_ylim(0.56, 1.12)
    # ax.set_yticks([0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1])

    # ax.set_xlim(0, 5)
    # ax.set_ylim(0.4, 1.008)

    # ax.set_ylim(0.942, 1.008)
    # ax.set_ylim(0.765, 1.135)

    # Assemble legends
    # l1_handles = [handle1]
    # l1_handles = [handle1, handle2]
    l2_handles = [handle1, handle3, handle4, handle5, handle6]
    # top_legend = plt.legend(handles=l1_handles, loc="upper left", fontsize=14)
    bottom_legend = plt.legend(handles=l2_handles, loc="lower left", fontsize=14)
    # ax.add_artist(top_legend)
    ax.add_artist(bottom_legend)

    # Save the figure
    plt.savefig("meff_2DUEG.pdf")
    return


if __name__ == "__main__":
    main()
