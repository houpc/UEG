#!/usr/bin/python3
import scienceplots
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib as mat
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cdict = {
    "blue" : "#0077BB",
    "cyan" : "#33BBEE",
    "teal" : "#009988",
    "orange" : "#EE7733",
    "red" : "#CC3311",
    "magenta" : "#EE3377",
    "grey" : "#BBBBBB",
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

points = ['o','^','s','v','p','<','h','>']
# Colormap = ["blue", "red", "orange", "violet", "green", "black", "darkblue"]
# Color_beta = {0.125: "red", 0.25: "orange", 0.5: "violet", 1.0: "green", 2.0: "blue", 2.5:"darkblue", 16.0: "black"}
Colormap = [
    cdict["blue"],
    cdict["red"],
    cdict["orange"],
    cdict["magenta"],
    cdict["cyan"],
    "black",
    cdict["teal"],
    "grey",
]
# pt_beta = {0.125: 'o', 0.25: '^', 0.5: 'h', 1.0: 'v', 2.0: 'd', 2.5:'p', 16.0: 's'}
pts = ['s', '^', 'v', 'p', 's', 'o', 'd']
# fig, axes = plt.subplots(1,2, sharex='col')#, sharey=True, gridspec_kw={'wspace': 0})
# fig, ax1 = plt.subplots(figsize=(6, 4.6))
# fig, ax1 = plt.subplots(figsize=(6, 4))
fig, ax1 = plt.subplots(figsize=(5, 5))
# fig.set_size_inches(13,10)

# # N*=5 results without irreducible vertex
# rs_VDMC = [0, 0.5, 1, 2, 3, 4, 5, 6]
# m_VDMC = [1.0, 0.95893, 0.9514, 0.9516, 0.9597, 0.9692, 0.9771, 0.9842]
# m_VDMC_err = [0, 0.00067, 0.0016, 0.0018, 0.0016, 0.0026, 0.0028, 0.0029]

# N*=6 results with irreducible vertex
rs_VDMC = [0, 1, 2, 3, 4, 5]
m_VDMC = [1.0, 0.94947, 0.9521, 0.9604, 0.9703, 0.97872]
m_VDMC_err = [0.0, 0.00078, 0.0014, 0.0014, 0.00061, 0.00079]

rs_VMC = [1, 2, 4, 5, 10]
# m_BFVMC = [[1.00, 0.01], [0.98, 0.01], [1.00, 0.02], [1.09,0.03],[1.28, 0.03]]
m_BFVMC = [1.00, 0.98, 1.00, 1.09, 1.28]
m_BFVMC_err = [0.01, 0.01, 0.02, 0.03, 0.03]
# m_SJVMC = [[0.96,0.01], [0.94, 0.02], [0.94, 0.02], [1.02, 0.02], [1.13, 0.03]]
m_SJVMC = [0.96, 0.94, 0.94, 1.02, 1.13]
m_SJVMC_err = [0.01, 0.02, 0.02, 0.02, 0.03]

rs_DMC = [1, 2, 3, 4, 5]
m_DMC = [0.918, 0.879, 0.856, 0.842, 0.791]
m_DMC_err = [0.006, 0.014, 0.014, 0.017, 0.01]

rs_RPA = [1, 2, 3, 4, 5, 6]
# m_RPA = [[0.97, 0], [0.992,0], [1.016,0] [1.039,0], [1.059, 0]]
# m_Gp = [[0.952, 0], [0.951, 0], [0.956, 0], [0.962, 0], [0.968, 0]]
# m_Gpm = [[0.957, 0], [0.966, 0], [0.983, 0], [1.005, 0], [1.028,0]]
m_G0W0 = [0.970, 0.992, 1.016, 1.039, 1.059, 1.078]
m_Gp = [0.952, 0.951, 0.956, 0.962, 0.968, 0.973]
m_Gpm = [0.957, 0.966, 0.983, 1.005, 1.028, 1.055]

def errorbar_mvsrs(rs, meff_data, merr, idx, label, ax=ax1, zorder=None):
    if zorder is not None:
        handle = ax.errorbar(rs, meff_data, merr, fmt=pts[idx], capthick=1, capsize=4,
                ms=5, color=Colormap[idx], label=label, zorder=zorder)
    else:
        handle = ax.errorbar(rs, meff_data, merr, fmt=pts[idx], capthick=1, capsize=4,
                    ms=5, color=Colormap[idx], label=label)
    return handle

def plot_mvsrs(rs, meff_data, idx, label, ls='-', ax=ax1, rs_HDL=None, meff_HDL=None, zorder=None):
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
    xgrid = np.arange(0, 6.2, 0.02)
    # ax.plot(rs, meff_data, 'o', ms=10, color=Colormap[idx])
    # ax.plot(xgrid, mfitfunc(xgrid), ls=ls, lw=2, color=Colormap[idx], label=label)
    if zorder is not None:
        handle, = ax.plot(xgrid, mfitfunc(xgrid), ls=ls, color=Colormap[idx], label=label, zorder=zorder)
    else:
        handle, = ax.plot(xgrid, mfitfunc(xgrid), ls=ls, color=Colormap[idx], label=label)

    yfit = np.ma.masked_invalid(mfitfunc(xgrid))
    # print(yfit)
    print("Turning point: rs = ", xgrid[np.argmin(yfit)])
    print("Effective mass ratio at turning point: ", np.min(yfit))
    return handle

# Taylor series for m* / m in the high-density limit to leading order in rs
# (c.f. Giuliani & Vignale, Quantum Theory of the Electron Liquid, 2008, p. 500)
def low_density_limit(x):
    alpha = (4.0 / (9.0 * np.pi))**(1.0 / 3.0)
    return 1 + alpha * x * np.log(x) / (2.0 * np.pi)

# high-density limit
rs_HDL_plot = np.linspace(1e-5, 0.35, num=101)
meff_HDL_plot = np.array([low_density_limit(rs) for rs in rs_HDL_plot])

# Use exact expression in the high-density limit for RPA(+FL) and RPT fits
cutoff_HDL = 0.1
rs_HDL = rs_HDL_plot[rs_HDL_plot <= cutoff_HDL]
meff_HDL = meff_HDL_plot[rs_HDL_plot <= cutoff_HDL]
print(rs_HDL)
print(meff_HDL)

# high-density limit
# plot_mvsrs(rs_HDL_plot, meff_HDL_plot, 7, "HDL", "--", zorder=1000)

# RPA
handle0 = plot_mvsrs(rs_RPA, m_G0W0, 7, r"RPA", '--', rs_HDL=rs_HDL, meff_HDL=meff_HDL)

# m_QMC = [m_DMC, m_SJVMC]
# m_QMC_errs = [m_DMC_err, m_SJVMC_err]
# labels = ["DMC (2021)", "VMC (2023)"]
# idx = 0
# for (mdat, merr, label) in zip(m_QMC, m_QMC_errs, labels):
    # errorbar_mvsrs(rs_VMC, mdat, merr, idx, label)
    # idx = idx + 1
handle1 = errorbar_mvsrs(rs_DMC, m_DMC, m_DMC_err, 0, "DMC [11]", zorder=10)
handle2 = errorbar_mvsrs(rs_VMC, m_SJVMC, m_SJVMC_err, 1, "VMC [12]", zorder=20)

# l1_handles = []
# m_RPA = [m_G0W0, m_Gp, m_Gpm]
# # merr = np.zeros(len(rs_RPA))
# labels = [r"$G_0W_0$", r"$G_+$", r"$G_+$\,\&\,$G_-$"]
# # labels = [r"$G_0W_0$", r"$G_+$ [15]", r"$G_+$\,\&\,$G_-$ [15]"]
# idx = 2
# for (mdat, label) in zip(m_RPA, labels):
#     print("\nPlotting ", label)
#     handle = plot_mvsrs(rs_RPA, mdat, idx, label, '--', rs_HDL=rs_HDL, meff_HDL=meff_HDL)
#     l1_handles.append(handle)
#     idx = idx + 1

# Add points at rs = 0
# rs_HDL = np.insert(rs_HDL, 0, 0.0)
# meff_HDL = np.insert(meff_HDL, 0, 1.0)

handle3 = errorbar_mvsrs(rs_VDMC[1:None], m_VDMC[1:None], m_VDMC_err[1:None], 5, "This work", zorder=30)
# rs_VDMC.append(5.2)
# m_VDMC.append(0.98)
print("\nPlotting our data")
plot_mvsrs(rs_VDMC, m_VDMC, 5, "", '-', rs_HDL=rs_HDL, meff_HDL=meff_HDL)

ax1.set_xlabel(r"$r_s$")
ax1.set_ylabel(r"$m^*/m$")
# ax1.set_xlim(0, 6.2)
ax1.set_xlim(0, 5.2)
ax1.set_ylim(0.765, 1.07)
# ax1.set_ylim(0.765, 1.095)
# ax1.set_ylim(0.745, 1.095)
# ax1.set_ylim(0.78, 1.065)
# ax1.set_ylim(0.94, 0.99)
# ax1.legend(loc="upper left", fontsize=14, ncol=2)

# Assemble legends
l2_handles = [handle0, handle1, handle2, handle3]
# l2_handles = [handle1, handle2, handle3]
# top_legend = plt.legend(handles=l1_handles, loc="upper left", fontsize=14, title="Ref.[15]")
# top_legend = plt.legend(handles=l1_handles, loc="upper left", fontsize=14)
bottom_legend = plt.legend(handles=l2_handles, loc="lower left", fontsize=14)
# ax1.add_artist(top_legend)
ax1.add_artist(bottom_legend)
# ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
# ax1.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0, 1.05])
ax1.set_xticks([0, 1, 2, 3, 4, 5])
ax1.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0, 1.05])
ax1.annotate(r"3D", xy=(0.06, 0.9), xycoords="axes fraction")

# ax1.set_xlabel(r"$r_s$", size=34)
# ax1.set_ylabel(r"$m^*/m$", size=34)
# ax1.tick_params(labelsize=30)
# ax1.legend(loc=2, labelspacing=1)
# ax1.legend(loc=3, fontsize=26)

plt.savefig("meff_3DUEG.pdf")