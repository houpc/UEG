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

function spline(x, y, e; xmin=0.0, xmax=x[end])
    # generate knots with spline without constraints
    w = 1.0 ./ e
    spl = interp.UnivariateSpline(x, y; w=w, k=3)
    __x = collect(LinRange(xmin, xmax, 1000))
    yfit = spl(__x)
    return __x, yfit
end

function get_Fs(rs)
    return get_Fs_PW(rs)
    # if rs < 1.0 || rs > 5.0
    #     return get_Fs_DMC(rs)
    # else
    #     return get_Fs_PW(rs)
    # end
end

"""
Get the symmetric l=0 Fermi-liquid parameter F⁰ₛ from DMC data of 
Moroni, Ceperley & Senatore (1995) [Phys. Rev. Lett. 75, 689].
"""
function get_Fs_DMC(rs)
    return error("Not yet implemented!")
end

"""
Get the symmetric l=0 Fermi-liquid parameter F⁰ₛ via interpolation of the 
compressibility ratio data of Perdew & Wang (1992) [Phys. Rev. B 45, 13244].
"""
@inline function get_Fs_PW(rs)
    # if rs < 1.0 || rs > 5.0
    #     @warn "The Perdew-Wang interpolation for Fs may " *
    #           "be inaccurate outside the metallic regime!"
    # end
    kappa0_over_kappa = 1.0025 - 0.1721rs - 0.0036rs^2
    # F⁰ₛ = κ₀/κ - 1
    return kappa0_over_kappa - 1.0
end

function get_meff_from_sigma(para::Parameter.Para; int_type=:rpa)
    # Make sure we are using parameters for the bare UEG theory
    @assert para.Λs == para.Λa == 0.0

    # Small params
    # kF = para.kF
    # Euv, rtol = 1000 * para.EF, 1e-11

    # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
    rs = round(para.rs; sigdigits=13)
    Fs = get_Fs(rs)
    if int_type == :ko_const
        println("Fermi liquid parameter at rs = $(rs): Fs = $Fs")
    end

    # Get RPA+FL self-energy
    sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(
        para;
        # Euv=Euv,
        # rtol=rtol,
        int_type=int_type,
        Fs=int_type == :ko_const ? -Fs : nothing,  # NOTE: NEFT uses opposite sign convention for F!
    )

    # Get effective mass ratio
    meff, kamp = SelfEnergy.massratio(para, sigma_tau_dynamic, sigma_tau_instant)
    @assert kamp ≈ para.kF
    return meff
end

# exchange interaction (Ws + Wa \sigma\sigma)_ex to a direct interaction Ws'+Wa' \sigma\sigma 
# # exchange S/A interaction projected to the spin-symmetric and antisymmetric parts
function exchange2direct(Wse, Wae)
    Ws = (Wse + 3 * Wae) / 2
    Wa = (Wse - Wae) / 2
    return Ws, Wa
end

function main()
    # System parameters
    dim = 3
    mass2 = 1e-8
    # beta = 40.0
    beta = 1000.0

    # rslist = [1.0, 5.0]
    # rslist = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # rslist = collect(LinRange(0.01, 6.0, 201))
    # rslist = collect(LinRange(0.01, 10.0, 51))

    bigrslist = collect(LinRange(0.01, 10.0, 41))
    rslist = [0.01; collect(range(0.25, 10; step=0.25))]
    rspmlist = copy(rslist)
    # rspmlist = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0]

    # bigrslist = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0]
    # rslist = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0]

    # rslist = [0.01, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # rslist = collect(LinRange(0.01, 6.0, 51))
    # rslist = collect(LinRange(0.01, 10.0, 11))

    make_rs_plot = rslist != [1.0, 5.0]
    make_angular_plot = rslist == [1.0, 5.0]

    # rs_Fsm1 = 5.24881  # Fs(rs ≈ 5.24881) = -1
    # rslist = [rs_Fsm1]
    # make_rs_plot = false
    # make_angular_plot = false

    int_type = :ko_const
    # int_type = :ko_moroni

    # int_type = :ko_takada
    # int_type = :ko_takada_plus

    if int_type == :ko_const
        kotypestring = "const. \$f_+\$"
    elseif int_type == :ko_takada
        kotypestring = "Takada \$f_+\$ and \$f_-\$"
    elseif int_type == :ko_takada_plus
        kotypestring = "Takada \$f_+\$"
    elseif int_type == :ko_moroni
        kotypestring = "Moroni \$f_+\$"
    end

    color = [
        [cdict["orange"], cdict["magenta"], cdict["red"]],
        [cdict["blue"], cdict["cyan"], cdict["teal"]],
    ]

    # Plot angular dependence and average of RPA(+FL) integrands
    if make_angular_plot
        fig = figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        rs1_handles = []
        rs5_handles = []
    end
    if make_rs_plot
        F1_rpa_vs_rs = []
        F1_rpa_fl_vs_rs = []
        m_sigma_G0W0 = []
        m_sigma_G0Wp = []
        m_sigma_G0Wpm = []
    end
    for (i, rs) in enumerate(bigrslist)
        param = Parameter.rydbergUnit(1.0 / beta, rs)
        kF, β = param.kF, param.β

        # Small params
        Euv, rtol = 100 * param.EF, 1e-11
        # Euv, rtol = 1000 * param.EF, 1e-11

        nu_grid = CompositeGrid.LogDensedGrid(
            :gauss,
            [-1.0, 1.0],
            [-1.0, 0.0, 1.0],
            20,
            1e-6,
            12,
        )
        nus = nu_grid.grid
        thetas = acos.(nus)

        # |k - k'| = kF sqrt(2(1 - ν))
        k_m_kps = @. kF * sqrt(2 * (1 - nus))

        # Get Landau parameter F⁰ₛ from Perdew & Wang compressibility fit
        Fs = get_Fs(rs)
        if int_type == :ko_const
            println("rs = $rs, Fs = $Fs, fs = $(Fs / param.NF)")
        end

        # Either use constant Fs from P&W, q-dependent Takada ansatz, or Corradini fit to Moroni DMC data
        if int_type == :ko_const
            landaufunc = Interaction.landauParameterConst
        elseif int_type == :ko_takada
            landaufunc = Interaction.landauParameterTakada
        elseif int_type == :ko_takada_plus
            landaufunc = Interaction.landauParameterTakadaPlus
        elseif int_type == :ko_moroni
            landaufunc = Interaction.landauParameterMoroni
        end

        # Get W0_{+}(p, iωₙ=0) * Vinv (spin-symmetric static exchange part of RPA interaction divided by V)
        W0_dyn, W0_inst = Interaction.RPAwrapped(
            Euv,
            rtol,
            k_m_kps,
            param;
            regular=false,
        )
        # W0_inst = 1 / V, W0_dyn = Wtilde, so W = (1 / W0_inst + W0_dyn) = V + Wtilde
        W0_plus_static = @. (1 / W0_inst[1, 1, :] + W0_dyn[1, 1, :])

        # Get R_{+}(p, iωₙ=0) * Vinv (spin-symmetric static exchange part of KO interaction divided by V)
        R_dyn, R_inst = Interaction.KOwrapped(
            Euv,
            rtol,
            k_m_kps,
            param;
            regular=false,
            int_type=int_type,
            landaufunc=landaufunc,
            Fs=int_type == :ko_const ? Fs : nothing,  # NOTE: NEFT uses opposite sign convention for F!
        )
        # R_inst = 1 / V, R_dyn = Rtilde, so R = (1 / R_inst + R_dyn) = V + Rtilde
        R_plus_static = @. (1 / R_inst[1, 1, :] + R_dyn[1, 1, :])

        @assert maximum(imag(R_plus_static)) ≤ 1e-10
        @assert maximum(imag(W0_plus_static)) ≤ 1e-10

        # The RPA and RPA+FL integrands are ν W0_{+}(p, iωₙ=0) and ν R_{+}(p, iωₙ=0), respectively
        F1p_rpa_integrand = @. -param.NF * nus * real(W0_plus_static)
        F1p_rpa_fl_integrand = @. -param.NF * nus * real(R_plus_static)

        # Perform angular integrations ν ∈ [-1, 1]
        F1p_rpa = 0.5 * CompositeGrids.Interp.integrate1D(F1p_rpa_integrand / 2.0, nu_grid)
        F1p_rpa_fl = 0.5 * CompositeGrids.Interp.integrate1D(F1p_rpa_fl_integrand / 2.0, nu_grid)

        # Use ElectronLiquid.jl to compute the same quantities
        p_rpa = ParaMC(rs=rs, beta=beta, Fs=-0.0, order=1, mass2=mass2)
        p_rpa_fl = ParaMC(rs=rs, beta=beta, Fs=-Fs, order=1, mass2=mass2)
        # NOTE: NEFT uses opposite sign convention for F!
        F1p_rpa_ueg, _ = -1 .* Ver4.projected_exchange_interaction(1, p_rpa, Ver4.exchange_interaction)
        F1p_rpa_fl_ueg, _ = -1 .* Ver4.projected_exchange_interaction(1, p_rpa_fl, Ver4.exchange_interaction)

        println("\nrs = $rs:" * "\nF^{(RPA)+}_1 = $F1p_rpa" * "\nF^{(RPA+FL)+}_1 = $F1p_rpa_fl")
        println("\nElectronLiquid:" * "\nF^{(RPA)+}_1 = $F1p_rpa_ueg" * "\nF^{(RPA+FL)+}_1 = $F1p_rpa_fl_ueg")

        if make_angular_plot
            # Plot angular dependence and average of RPA integrand
            plot(
                thetas,
                F1p_rpa .* one.(thetas);
                color=color[i][1],
                linestyle="--",
            )
            handle_rpa, = plot(
                thetas,
                F1p_rpa_integrand;
                color=color[i][1],
                linestyle="-",
                label="RPA",
            )
            # Plot angular dependence and average of RPA integrand
            plot(
                thetas,
                F1p_rpa_fl .* one.(thetas);
                color=color[i][2],
                linestyle="--",
            )
            handle_rpa_fl, = plot(
                thetas,
                F1p_rpa_fl_integrand;
                color=color[i][2],
                linestyle="-",
                label="\$G_{+}\$",
                # label="\$G_{+},\\; $kotypestring\$",
            )
            handles = [handle_rpa, handle_rpa_fl]
            if rs == 1
                append!(rs1_handles, handles)
            elseif rs == 5
                append!(rs5_handles, handles)
            end
        end
        if make_rs_plot
            push!(F1_rpa_vs_rs, F1p_rpa)
            push!(F1_rpa_fl_vs_rs, F1p_rpa_fl)
        end
    end
    for (i, rs) in enumerate(rspmlist)
        param = Parameter.rydbergUnit(1.0 / beta, rs)
        m_sigma_G0Wpm_this_rs = get_meff_from_sigma(param; int_type=:ko_simion_giuliani)
        # m_sigma_G0Wpm_this_rs = get_meff_from_sigma(param; int_type=:ko_takada)

        if make_rs_plot
            push!(m_sigma_G0Wpm, m_sigma_G0Wpm_this_rs)
        end
    end
    for (i, rs) in enumerate(rslist)
        param = Parameter.rydbergUnit(1.0 / beta, rs)
        m_sigma_G0W0_this_rs = get_meff_from_sigma(param; int_type=:rpa)
        m_sigma_G0Wp_this_rs = get_meff_from_sigma(param; int_type=:ko_moroni)
        # m_sigma_G0Wpm_this_rs = get_meff_from_sigma(param; int_type=:ko_takada)

        # m_sigma_G0Wp_this_rs = get_meff_from_sigma(param; int_type=:ko_const)

        # m_sigma_G0Wp_this_rs = get_meff_from_sigma(param; int_type=:ko_takada_plus)
        # m_sigma_G0Wpm_this_rs = get_meff_from_sigma(param; int_type=:ko_takada)

        if make_rs_plot
            push!(m_sigma_G0W0, m_sigma_G0W0_this_rs)
            push!(m_sigma_G0Wp, m_sigma_G0Wp_this_rs)
            # push!(m_sigma_G0Wpm, m_sigma_G0Wpm_this_rs)
        end
    end
    if make_angular_plot
        xticks([0, π / 4, π / 2, 3π / 4, π], ["\$0\$", "\$\\pi/4\$", "\$\\pi/2\$", "\$3\\pi/4\$", "\$\\pi\$"])
        xlim(0, π)
        xlabel("\$\\theta\$")
        ylabel("\$P_1(\\cos\\theta) F^+_{1,1}(\\theta)\$")

        # Assemble legends
        top_legend = plt.legend(handles=rs1_handles, loc=(0.5, 0.325), fontsize=14, title="\$r_s = 1\$", ncol=2, columnspacing=0.9)
        bottom_legend = plt.legend(handles=rs5_handles, loc=(0.5, 0.09), fontsize=14, title="\$r_s = 5\$", ncol=2, columnspacing=0.9)
        ax.add_artist(top_legend)
        ax.add_artist(bottom_legend)
        tight_layout()
        savefig("Fp1_rpa_and_rpa_fl_angular_dependence_$(int_type).pdf")
    end
    if make_rs_plot
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

        # Add points at rs = 0
        pushfirst!(bigrslist, 0.0)
        pushfirst!(rslist, 0.0)
        pushfirst!(rspmlist, 0.0)
        pushfirst!(rs_HDL, 0.0)
        pushfirst!(meff_HDL, 1.0)
        pushfirst!(rs_HDL_plot, 0.0)
        pushfirst!(meff_HDL_plot, 1.0)
        pushfirst!(F1_rpa_vs_rs, 0.0)
        pushfirst!(F1_rpa_fl_vs_rs, 0.0)
        pushfirst!(m_sigma_G0W0, 1.0)
        pushfirst!(m_sigma_G0Wp, 1.0)
        pushfirst!(m_sigma_G0Wpm, 1.0)

        # Save data
        np.savez("m_sigma_G0W0", rslist=rslist, mefflist=m_sigma_G0W0)
        # np.savez("m_sigma_G0Wp", rslist=rslist, mefflist=m_sigma_G0Wp)
        # np.savez("m_sigma_G0Wpm", rslist=rspmlist, mefflist=m_sigma_G0Wpm)
        np.savez("m_tree_level_G0W0", rslist=bigrslist, mefflist=(1 .+ F1_rpa_vs_rs))
        np.savez("m_tree_level_G0Wp", rslist=bigrslist, mefflist=(1 .+ F1_rpa_fl_vs_rs))

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
        plot_mvsrs(bigrslist, 1 .+ F1_rpa_vs_rs, "black", "Tree-level \$F^+_1\$, \$G_0 W_0\$", "--"; rs_HDL=rs_HDL, meff_HDL=meff_HDL)

        # Tree-level RPA+FL
        plot_mvsrs(bigrslist, 1 .+ F1_rpa_fl_vs_rs, "black", "Tree-level \$F^+_1\$, \$G_0 W_+\$", ":"; rs_HDL=rs_HDL, meff_HDL=meff_HDL)

        println()
        println(bigrslist)
        println()
        println(rslist)

        # Self-energy G_0 W_0
        plot_mvsrs(rslist, m_sigma_G0W0, colors[1], "\$G_0 W_0\$"; rs_HDL=rs_HDL, meff_HDL=meff_HDL)

        # Self-energy G_0 W_+ (Moroni)
        plot_mvsrs(rslist, m_sigma_G0Wp, colors[2], "\$G_0 W_+\$"; rs_HDL=rs_HDL, meff_HDL=meff_HDL)

        # Self-energy G_0 W_± (Simion & Giuliani)
        plot_mvsrs(rspmlist, m_sigma_G0Wpm, colors[3], "\$G_0 W_\\pm\$"; rs_HDL=rs_HDL, meff_HDL=meff_HDL)
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
        ax2.scatter(rs_RPA, m_G0W0; color=colors[4], marker="^", label="\$G_0 W_0\$ (Ref. [16])", zorder=700)
        ax2.scatter(rs_RPA, m_Gp; color=colors[5], marker="s", label="\$G_0 W_+\$ (Ref. [16])", zorder=800)
        ax2.scatter(rs_RPA, m_Gpm; color=colors[6], marker="D", label="\$G_0 W_\\pm\$ (Ref. [16])", zorder=900)

        # RPT Monte Carlo results for full m* / m
        m_VDMC = [1.0, 0.95893, 0.9514, 0.9516, 0.9597, 0.9692, 0.9771, 0.9842]
        m_VDMC_err = [0, 0.00067, 0.0016, 0.0018, 0.0016, 0.0026, 0.0028, 0.0029]
        rs_VDMC = [0, 0.5, 1, 2, 3, 4, 5, 6]
        ax2.errorbar(rs_VDMC, m_VDMC, m_VDMC_err; fmt="o", capthick=1, capsize=4, ms=5, color="black", label="RPT (This work)", zorder=1000)
        plot_mvsrs(rs_VDMC, m_VDMC, "black", nothing; rs_HDL=rs_HDL, meff_HDL=meff_HDL)

        legend(loc="best", fontsize=12)
        # legend(loc="best", ncol=2, columnspacing=0.45, fontsize=10)
        # legend(loc="best", ncol=2, columnspacing=0.45)
        # legend(loc=(0.09, 0.625))
        # legend(loc=(0.1, 0.5))
        ylabel("\$m^*/m\$")
        xlabel("\$r_s\$")
        # ylim(0.94, 1.02)
        tight_layout()
        savefig("effective_mass_comparisons.pdf")
        # savefig("meff3d_tree_level.pdf")
        # savefig("Fp1_rpa_and_rpa_fl_vs_rs_$(int_type).pdf")
    end
end

main()
