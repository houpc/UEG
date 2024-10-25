using CSV
using DataFrames
using PyCall
using PyPlot
# using Printf
using ElectronGas, ElectronLiquid, Parameters
using Lehmann, GreenFunc, CompositeGrids, CurveFit

@pyimport numpy as np   # for saving/loading numpy data
@pyimport scienceplots  # for style "science"
@pyimport scipy.interpolate as interp

# Vibrant qualitative colour scheme from https://personal.sron.nl/~pault/
const cdict = Dict([
    "orange" => "#EE7733",
    "blue" => "#0077BB",
    "cyan" => "#33BBEE",
    "magenta" => "#EE3377",
    "red" => "#CC3311",
    "teal" => "#009988",
    "grey" => "#BBBBBB",
]);
style = PyPlot.matplotlib."style"
style.use(["science", "std-colors"])
const color = [
    "black",
    cdict["orange"],
    cdict["blue"],
    cdict["cyan"],
    cdict["magenta"],
    cdict["red"],
    cdict["teal"],
]
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 16
rcParams["mathtext.fontset"] = "cm"
rcParams["font.family"] = "Times New Roman"

# Plot F1 vs rs
# fig2 = figure(figsize=(5, 5))
fig2 = figure(figsize=(6, 6))
ax2 = fig2.add_subplot(111)

function plot_mvsrs(rs, meff_data, color, label, ls="-", ax=ax2; rs_HDL=nothing, meff_HDL=nothing, zorder=nothing)
    # Add data in the high-density limit to the fit, if provided
    if !isnothing(rs_HDL) && !isnothing(meff_HDL)
        _rslist = unique([rs; rs_HDL])
        mdatalist = unique([meff_data; meff_HDL])
        P = sortperm(_rslist)
        rs = _rslist[P]
        meff_data = mdatalist[P]
    end

    # mfitfunc = interp.PchipInterpolator(rs, meff_data)
    mfitfunc = interp.Akima1DInterpolator(rs, meff_data)
    xgrid = np.arange(0, maximum(_rslist) + 0.2, 0.01)
    # xgrid = np.arange(0, 6.2, 0.01)
    # ax.scatter(rs, meff_data; color=color, marker="o")
    if isnothing(zorder)
        handle, = ax.plot(xgrid, mfitfunc(xgrid), ls=ls, color=color, label=label)
    else
        handle, = ax.plot(xgrid, mfitfunc(xgrid), ls=ls, color=color, label=label, zorder=zorder)
    end
    yfit = np.ma.masked_invalid(mfitfunc(xgrid))
    # print(yfit)
    # print("Turning point: rs = ", xgrid[np.argmin(yfit)])
    # print("Effective mass ratio at turning point: ", np.min(yfit))
    return handle
end

# Taylor series for m* / m in the high-density limit to leading order in rs
# (c.f. Giuliani & Vignale, Quantum Theory of the Electron Liquid, 2008, p. 500)
function high_density_limit(x)
    alpha = (4 / 9π)^(1 / 3.0)
    return 1 + alpha * x * log(x) / 2π
end

# High-density limit
rs_HDL_plot = collect(LinRange(1e-5, 0.35, 101))
meff_HDL_plot = high_density_limit.(rs_HDL_plot)

# Use exact expression in the high-density limit for RPT and Simion & Giuliani fits
cutoff_HDL = 0.1
rs_HDL = rs_HDL_plot[rs_HDL_plot.≤cutoff_HDL]
meff_HDL = meff_HDL_plot[rs_HDL_plot.≤cutoff_HDL]
println(rs_HDL)
println(meff_HDL)

################
### Load data ##
################

# G0W0

f_sigma_G0W0 = np.load("results/finalized_meff_results/3d/rpa/meff_3d_sigma_G0W0.npz")
rs_sigma_G0W0 = f_sigma_G0W0.get("rslist")
m_sigma_G0W0 = f_sigma_G0W0.get("mefflist")

f_sigma_G0Wp = np.load("results/finalized_meff_results/3d/simion_giuliani/meff_3d_sigma_G0Wp.npz")
# f_sigma_G0Wp = np.load("results/finalized_meff_results/3d/const/meff_3d_sigma_G0Wp.npz")
rs_sigma_G0Wp = f_sigma_G0Wp.get("rslist")
m_sigma_G0Wp = f_sigma_G0Wp.get("mefflist")

f_sigma_G0Wpm = np.load("results/finalized_meff_results/3d/simion_giuliani/meff_3d_sigma_G0Wpm.npz")
# f_sigma_G0Wpm = np.load("results/finalized_meff_results/3d/const/meff_3d_sigma_G0Wpm.npz")
rs_sigma_G0Wpm = f_sigma_G0Wpm.get("rslist")
m_sigma_G0Wpm = f_sigma_G0Wpm.get("mefflist")

f_tree_level_G0W0 = np.load("results/finalized_meff_results/3d/rpa/meff_3d_tree_level_G0W0.npz")
rs_tree_level_G0W0 = f_tree_level_G0W0.get("rslist")
m_tree_level_G0W0 = f_tree_level_G0W0.get("mefflist")

f_tree_level_G0Wp = np.load("results/finalized_meff_results/3d/const/meff_3d_tree_level_G0Wp.npz")
rs_tree_level_G0Wp = f_tree_level_G0Wp.get("rslist")
m_tree_level_G0Wp = f_tree_level_G0Wp.get("mefflist")

# GW

f_sigma_GW0 = np.load("results/finalized_meff_results/3d/rpa/meff_3d_sigma_GW0.npz")
rs_sigma_GW0 = f_sigma_GW0.get("rslist")
m_sigma_GW0 = f_sigma_GW0.get("mefflist")

# f_sigma_GWp = np.load("results/finalized_meff_results/3d/simion_giuliani/meff_3d_sigma_GWp.npz")
# # f_sigma_GWp = np.load("results/finalized_meff_results/3d/const/meff_3d_sigma_GWp.npz")
# rs_sigma_GWp = f_sigma_GWp.get("rslist")
# m_sigma_GWp = f_sigma_GWp.get("mefflist")

# f_sigma_GWpm = np.load("results/finalized_meff_results/3d/simion_giuliani/meff_3d_sigma_GWpm.npz")
# # f_sigma_GWpm = np.load("results/finalized_meff_results/3d/const/meff_3d_sigma_GWpm.npz")
# rs_sigma_GWpm = f_sigma_GWpm.get("rslist")
# m_sigma_GWpm = f_sigma_GWpm.get("mefflist")

println(m_sigma_G0W0)
println(m_sigma_GW0)

println(m_sigma_G0Wp)
# TODO: Fix these outliers
println(rs_sigma_G0Wp[4])
deleteat!(rs_sigma_G0Wp, 4)
deleteat!(m_sigma_G0Wp, 4)
deleteat!(rs_sigma_G0Wp, 3)
deleteat!(m_sigma_G0Wp, 3)
# println(m_sigma_GWp)

println(m_sigma_G0Wpm)
# println(m_sigma_GWpm)

################
################
################

# Add points at rs = 0
for rslist in [rs_HDL, rs_sigma_G0W0, rs_sigma_G0Wp, rs_sigma_G0Wpm, rs_tree_level_G0W0, rs_tree_level_G0Wp, rs_sigma_GW0]
    if rslist[1] != 0.0
        pushfirst!(rslist, 0.0)
    end
end
for mefflist in [meff_HDL, m_tree_level_G0W0, m_tree_level_G0Wp, m_sigma_G0W0, m_sigma_G0Wp, m_sigma_G0Wpm, meff_HDL_plot, m_sigma_GW0]
    if mefflist[1] != 1.0
        pushfirst!(mefflist, 1.0)
    end
end

# Plot data
colors = [
    cdict["orange"],
    cdict["blue"],
    cdict["magenta"],
    cdict["cyan"],
    cdict["red"],
    cdict["teal"],
    "black",
]

# High-density limit
# plot_mvsrs(rs_HDL_plot, meff_HDL_plot, colors[1], "High-density limit", "--"; zorder=1000)

# Tree-level RPA
handle1 = plot_mvsrs(rs_tree_level_G0W0, m_tree_level_G0W0, "black", "Tree-level \$F^+_1\$, \$G_0 W_0\$", "--"; rs_HDL=rs_HDL, meff_HDL=meff_HDL)

# Tree-level RPA+FL
handle2 = plot_mvsrs(rs_tree_level_G0Wp, m_tree_level_G0Wp, "black", "Tree-level \$F^+_1\$, \$G_0 W_+\$", ":"; rs_HDL=rs_HDL, meff_HDL=meff_HDL)

# Self-energy G W_0
handle3 = plot_mvsrs(rs_sigma_GW0, m_sigma_GW0, colors[1], "\$G W_0\$", "-"; rs_HDL=rs_HDL, meff_HDL=meff_HDL)

# Self-energy G W_+ (Moroni)
# plot_mvsrs(rs_sigma_GWp, m_sigma_GWp, colors[2], "\$G W_+\$", "-"; rs_HDL=rs_HDL, meff_HDL=meff_HDL)

# Self-energy G W_± (Simion & Giuliani)
# plot_mvsrs(rs_sigma_GWpm, m_sigma_GWpm, colors[3], "\$G W_\\pm\$", "-"; rs_HDL=rs_HDL, meff_HDL=meff_HDL)

# Self-energy G_0 W_0
handle4 = plot_mvsrs(rs_sigma_G0W0, m_sigma_G0W0, colors[1], "\$G_0 W_0\$", "--"; rs_HDL=rs_HDL, meff_HDL=meff_HDL)

# Self-energy G_0 W_+ (Moroni)
handle5 = plot_mvsrs(rs_sigma_G0Wp, m_sigma_G0Wp, colors[2], "\$G_0 W_+\$", "--"; rs_HDL=rs_HDL, meff_HDL=meff_HDL)

# Self-energy G_0 W_± (Simion & Giuliani)
handle6 = plot_mvsrs(rs_sigma_G0Wpm, m_sigma_G0Wpm, colors[3], "\$G_0 W_\\pm\$", "--"; rs_HDL=rs_HDL, meff_HDL=meff_HDL)
# ax2.scatter(rspmlist, m_sigma_G0Wpm; color=colors[3], marker="v", zorder=600)

# Simion & Giuliani self-energy G_0 W_0, G_0 W_+, and G_0 W_± results
rs_RPA = [0, 1, 2, 3, 4, 5, 6]
m_G0W0 = [1.0, 0.970, 0.992, 1.016, 1.039, 1.059, 1.078]
m_Gp = [1.0, 0.952, 0.951, 0.956, 0.962, 0.968, 0.973]
m_Gpm = [1.0, 0.957, 0.966, 0.983, 1.005, 1.028, 1.055]
# plot_mvsrs(rs_RPA, m_G0W0, "black", "\$\\Sigma_{G_0 W_0}\$", ":")
# plot_mvsrs(rs_RPA, m_Gp, "black", "\$\\Sigma_{G_0 W_+}\$", "-.")
# plot_mvsrs(rs_RPA, m_G0W0, colors[5], "\$\\Sigma_{G_0 W_0}\$ (Simion, 2008)", "--"; rs_HDL=rs_HDL, meff_HDL=meff_HDL)
# plot_mvsrs(rs_RPA, m_Gp, colors[6], "\$\\Sigma_{G_0 W_+}\$ (Simion, 2008)", "--"; rs_HDL=rs_HDL, meff_HDL=meff_HDL)
# plot_mvsrs(rs_RPA, m_Gpm, "black", "\$\\Sigma_{G_0 W_\\pm}\$", "--"; rs_HDL=rs_HDL, meff_HDL=meff_HDL)
handle7 = ax2.scatter(rs_RPA, m_G0W0; color=colors[4], marker="^", label="\$G_0 W_0\$ (Ref. [16])", zorder=700)
handle8 = ax2.scatter(rs_RPA, m_Gp; color=colors[5], marker="s", label="\$G_0 W_+\$ (Ref. [16])", zorder=800)
handle9 = ax2.scatter(rs_RPA, m_Gpm; color=colors[6], marker="D", label="\$G_0 W_\\pm\$ (Ref. [16])", zorder=900)

# RPT Monte Carlo results for full m* / m
m_VDMC = [1.0, 0.95893, 0.9514, 0.9516, 0.9597, 0.9692, 0.9771, 0.9842]
m_VDMC_err = [0, 0.00067, 0.0016, 0.0018, 0.0016, 0.0026, 0.0028, 0.0029]
rs_VDMC = [0, 0.5, 1, 2, 3, 4, 5, 6]
handle10, = ax2.errorbar(rs_VDMC, m_VDMC, m_VDMC_err; fmt="o", capthick=1, capsize=4, ms=5, color="black", label="RPT (This work)", zorder=1000)
plot_mvsrs(rs_VDMC, m_VDMC, "black", nothing; rs_HDL=rs_HDL, meff_HDL=meff_HDL)

l1_handles = [handle3, handle4, handle5, handle6]
l2_handles = [handle7, handle8, handle9, handle10, handle1, handle2]
# l2_handles = [handle1, handle2, handle7, handle8, handle9, handle10]
top_legend = plt.legend(handles=l1_handles, loc="upper left", fontsize=12)
bottom_legend = plt.legend(handles=l2_handles, loc="lower left", fontsize=12)
ax2.add_artist(top_legend)
ax2.add_artist(bottom_legend)

# legend(loc="best", fontsize=12)
# legend(loc="best", ncol=2, columnspacing=0.45, fontsize=10)
# legend(loc="best", ncol=2, columnspacing=0.45)
# legend(loc=(0.09, 0.625))
# legend(loc=(0.1, 0.5))
ylabel("\$m^*/m\$")
xlabel("\$r_s\$")
# ylim(0.94, 1.02)
tight_layout()
savefig("figures/meff_3DUEG_comparisons_gw.pdf")

rs = CSV.read("results/finalized_meff_results/3d/rpa/rs.out", DataFrame)[:, 1]
z_g0w0 = CSV.read("results/finalized_meff_results/3d/rpa/z_g0w0.out", DataFrame)[:, 1]
z_gw0 = CSV.read("results/finalized_meff_results/3d/rpa/z_gw0.out", DataFrame)[:, 1]

# Plot Z factors for G_0 W_0 and G W_0
fig3 = figure(figsize=(6, 6))
ax3 = fig3.add_subplot(111)
ax3.plot(rs, z_g0w0, label="\$G_0 W_0\$")
ax3.plot(rs, z_gw0, label="\$G W_0\$")
ax3.set_xlabel("\$r_s\$")
ax3.set_ylabel("\$Z\$")
ax3.legend()
tight_layout()
savefig("figures/z_3DUEG_comparisons_gw.pdf")
