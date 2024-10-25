using PyCall
using PyPlot
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

function get_sigma_tau(para::Parameter.Para)
    Σ_dyn, Σ_inst = SelfEnergy.G0W0(para; int_type=:ko_simion_giuliani)
    kgrid = Σ_dyn.mesh[2]
    return Σ_dyn, Σ_inst, kgrid
end

function get_sigma_iw0(para::Parameter.Para)
    Σ_dyn, Σ_inst, kgrid = get_sigma_tau(para)
    Σ_freq = dlr_to_imfreq(to_dlr(Σ_dyn), [0, 1])
    @assert kgrid == Σ_freq.mesh[2]
    sigma_iw0 = Σ_freq[1, :] + Σ_inst[1, :]  # add back the instantaneous part
    return sigma_iw0, kgrid
end

function get_meff_from_sigma(para::Parameter.Para)
    # Make sure we are using parameters for the bare UEG theory
    @assert para.Λs == para.Λa == 0.0
    # Get RPA+FL self-energy
    Σ_dyn, Σ_inst = SelfEnergy.G0W0(para; int_type=:ko_simion_giuliani)
    # Get effective mass ratio
    meff, kamp = SelfEnergy.massratio(para, Σ_dyn, Σ_inst)
    @assert kamp ≈ para.kF
    return meff
end

function plot_sigma_iw0(beta, rslist; max_kkF_plot=6.0, dir=@__DIR__)
    Σ_iw0_k0_list = []
    for rs in rslist
        param = Parameter.rydbergUnit(1.0 / beta, rs)
        Σ_iw0, kgrid = get_sigma_iw0(param)
        kkFgrid = kgrid / param.kF
        kmask = kkFgrid .≤ max_kkF_plot
        kkF_plot = kkFgrid[kmask]
        push!(Σ_iw0_k0_list, Σ_iw0[1])
        if rs == 2.0
            # Plot Σ(iw = 0) vs k at rs=2
            fig1 = figure(figsize=(6, 6))
            ax1 = fig1.add_subplot(111)
            ax1.plot(kkF_plot, real(Σ_iw0)[kmask]; label="\$\\text{Re}\\Sigma(k, i\\omega_0)\$", color=cdict["orange"])
            ax1.plot(kkF_plot, imag(Σ_iw0)[kmask]; label="\$\\text{Im}\\Sigma(k, i\\omega_0)\$", color=cdict["blue"])
            legend(loc="best")
            ylabel("\$\\Sigma(k, i\\omega_0)\$")
            xlabel("\$k / k_F\$")
            tight_layout()
            savefig(joinpath(dir, "Σ_iw0_rs2_simion_giuliani.pdf"))
        end
    end
    # Plot Σ(iw = 0) vs rs at k=0
    fig2 = figure(figsize=(6, 6))
    ax2 = fig2.add_subplot(111)
    x, y = rslist, real(Σ_iw0_k0_list)
    e = 1e-5 * ones(length(x))
    __x, yfit = spline(x, y, e)
    ax2.scatter(rslist, real(Σ_iw0_k0_list); color=cdict["orange"])
    ax2.plot(__x, yfit; label="\$\\text{Re}\\Sigma(k = 0, i\\omega_0)\$", color=cdict["orange"])
    x, y = rslist, imag(Σ_iw0_k0_list)
    e = 1e-5 * ones(length(x))
    __x, yfit = spline(x, y, e)
    ax2.scatter(rslist, imag(Σ_iw0_k0_list); marker="s", color=cdict["blue"])
    ax2.plot(__x, yfit; label="\$\\text{Im}\\Sigma(k = 0, i\\omega_0)\$", color=cdict["blue"])
    legend(loc="best")
    ylabel("\$\\Sigma(k = 0, i\\omega_0)\$")
    xlabel("\$r_s\$")
    tight_layout()
    savefig(joinpath(dir, "Σ_iw0_k0_simion_giuliani.pdf"))
end

function plot_sigma_t0(beta, rslist; max_kkF_plot=6.0, dir=@__DIR__)
    Σ_dyn_k0_list = []
    Σ_inst_k0_list = []
    for rs in rslist
        param = Parameter.rydbergUnit(1.0 / beta, rs)
        Σ_dyn, Σ_inst, kgrid = get_sigma_tau(param)
        kkFgrid = kgrid / param.kF
        kmask = kkFgrid .≤ max_kkF_plot
        kkF_plot = kkFgrid[kmask]
        push!(Σ_dyn_k0_list, Σ_dyn[1, 1])
        push!(Σ_inst_k0_list, Σ_inst[1, 1])
        if rs == 2.0
            # Plot Σ(τ = 0) vs k at rs=2
            fig1 = figure(figsize=(6, 6))
            ax1 = fig1.add_subplot(111)
            ax1.plot(kkF_plot, real(Σ_dyn[1, :])[kmask]; label="\$\\text{Re}\\Sigma_{\\text{dyn}}(k, \\tau = 0)\$", color=cdict["orange"])
            ax1.plot(kkF_plot, real(Σ_inst[1, :])[kmask]; label="\$\\text{Re}\\Sigma_{\\text{inst}}(k, \\tau = 0)\$", color=cdict["blue"])
            legend(loc="best")
            ylabel("\$\\text{Re}\\Sigma(k, \\tau = 0)\$")
            xlabel("\$k / k_F\$")
            tight_layout()
            savefig(joinpath(dir, "ReΣ_t0_rs2_simion_giuliani.pdf"))
        end
    end
    # Plot Σ(τ = 0) vs rs at k=0
    fig2 = figure(figsize=(6, 6))
    ax2 = fig2.add_subplot(111)
    x, y = rslist, real(Σ_dyn_k0_list)
    e = 1e-5 * ones(length(x))
    __x, yfit = spline(x, y, e)
    ax2.scatter(rslist, real(Σ_dyn_k0_list); color=cdict["orange"])
    ax2.plot(__x, yfit; label="\$\\text{Re}\\Sigma_{\\text{dyn}}(k = 0, \\tau = 0)\$", color=cdict["orange"])
    x, y = rslist, real(Σ_inst_k0_list)
    e = 1e-5 * ones(length(x))
    __x, yfit = spline(x, y, e)
    ax2.scatter(rslist, real(Σ_inst_k0_list); marker="s", color=cdict["blue"])
    ax2.plot(__x, yfit; label="\$\\text{Re}\\Sigma_{\\text{inst}}(k = 0, \\tau = 0)\$", color=cdict["blue"])
    legend(loc="best")
    ylabel("\$\\text{Re}\\Sigma(k = 0, \\tau = 0)\$")
    xlabel("\$r_s\$")
    tight_layout()
    savefig(joinpath(dir, "ReΣ_t0_k0_simion_giuliani.pdf"))
end

function exchange2direct(Wse, Wae)
    Ws = (Wse + 3 * Wae) / 2
    Wa = (Wse - Wae) / 2
    return Ws, Wa
end

function plot_landaufunc(beta, rslist; q_kF_plot=collect(LinRange(1e-5, 8, 501)), dir=@__DIR__)
    # Plot F_s and F_a vs q at rs = 2, 4, 6
    fig1 = figure(figsize=(6, 6))
    fig2 = figure(figsize=(6, 6))
    fig3 = figure(figsize=(6, 6))
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    ax3 = fig3.add_subplot(111)
    ic = 1
    colors = [[cdict["orange"], cdict["blue"]], [cdict["magenta"], cdict["cyan"]], [cdict["red"], cdict["teal"]]]
    # Store q = 0 values of F_s and F_a for each rs
    Fs_q0_rslist = []
    Fa_q0_rslist = []
    for rs in rslist
        param = Parameter.rydbergUnit(1.0 / beta, rs)
        # Calculate the Coulomb interaction for plot momenta q
        Vq = []
        for q in q_kF_plot * param.kF
            Vs, Va = Interaction.coulomb(q, param)
            @assert Va == 0.0  # The Coulomb interaction couples spin-symmetrically
            push!(Vq, Vs)
        end
        if rs in [1.0, 2.0, 4.0]
            Fs_qlist = []
            Fa_qlist = []
            Fs_qlist_v2 = []
            for q_kF in q_kF_plot
                Fs, Fa = Interaction.landauParameterSimionGiuliani(q_kF * param.kF, 0, param)
                Fs_v2, _ = Interaction.landauParameterMoroni(q_kF * param.kF, 0, param)
                push!(Fs_qlist, Fs)
                push!(Fa_qlist, Fa)
                push!(Fs_qlist_v2, Fs_v2)
            end
            # Plot F_i
            ax1.plot(q_kF_plot, Fs_qlist; label="\$F^+(q)\$ (\$r_s = $(Int(round(rs)))\$)", color=colors[ic][1])
            ax1.plot(q_kF_plot, Fa_qlist; label="\$F^-(q)\$ (\$r_s = $(Int(round(rs)))\$)", color=colors[ic][2])
            # Same as above, but plot G± = F± / V
            Gs_qlist = Fs_qlist ./ Vq
            Ga_qlist = Fa_qlist ./ Vq
            Gs_qlist_v2 = Fs_qlist_v2 ./ Vq
            ax2.plot(q_kF_plot, Gs_qlist; label="\$r_s = $(Int(round(rs)))\$", color=color[ic+1])
            # ax2.plot(q_kF_plot, Gs_qlist; label="\$r_s = $(Int(round(rs)))\$ (landauParameterSimionGiuliani)", color=colors[ic][1])
            # ax2.plot(q_kF_plot, Gs_qlist_v2; label="\$r_s = $(Int(round(rs)))\$ (landauParameterMoroni)", linestyle="--", color=colors[ic][2])
            ax3.plot(q_kF_plot, Ga_qlist; label="\$r_s = $(Int(round(rs)))\$", color=color[ic+1])
            # Increment color index
            ic += 1
        end
        Fs_q0, Fa_q0 = Interaction.landauParameterSimionGiuliani(1e-5, 0, param)
        push!(Fs_q0_rslist, Fs_q0)
        push!(Fa_q0_rslist, Fa_q0)
    end
    # Finish Fig. 1
    ax1.legend(loc="best")
    ax1.set_ylabel("\$F^\\pm(q)\$")
    ax1.set_xlabel("\$q / k_F\$")
    fig1.tight_layout()
    fig1.savefig(joinpath(dir, "Fs_and_Fa_vs_q_simion_giuliani.pdf"))
    # Finish Fig. 2
    ax2.legend(loc="best")
    ax2.set_ylabel("\$G^+(q)\$")
    ax2.set_xlabel("\$q / k_F\$")
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 4)
    fig2.tight_layout()
    fig2.savefig(joinpath(dir, "Gs_vs_q_simion_giuliani.pdf"))
    # Finish Fig. 3
    ax3.legend(loc="best")
    ax3.set_ylabel("\$G^-(q)\$")
    ax3.set_xlabel("\$q / k_F\$")
    ax3.set_xlim(0, 4)
    ax3.set_ylim(0, 1)
    fig3.tight_layout()
    fig3.savefig(joinpath(dir, "Ga_vs_q_simion_giuliani.pdf"))
    # Plot Fs and Fa vs rs at q = 0
    fig4 = figure(figsize=(6, 6))
    ax4 = fig4.add_subplot(111)
    println(rslist)
    println(Fs_q0_rslist)
    println(Fa_q0_rslist)
    ax4.plot(rslist, Fs_q0_rslist; label="\$F^+(r_s, q = 0)\$", color=cdict["orange"])
    ax4.plot(rslist, Fa_q0_rslist; label="\$F^-(r_s, q = 0)\$", color=cdict["blue"])
    legend(loc="best")
    ylabel("\$F^\\pm(r_s, q = 0)\$")
    xlabel("\$r_s\$")
    tight_layout()
    savefig(joinpath(dir, "Fs_and_Fa_q0_vs_rs_simion_giuliani.pdf"))
end

function plot_meff(beta, rslist; dir=@__DIR__)
    meff_list = []
    for rs in rslist
        param = Parameter.rydbergUnit(1.0 / beta, rs)
        meff = get_meff_from_sigma(param)
        push!(meff_list, meff)
    end
    # Plot meff vs rs
    fig1 = figure(figsize=(6, 6))
    ax1 = fig1.add_subplot(111)
    x, y = rslist, meff_list
    e = 1e-5 * ones(length(x))
    __x, yfit = spline(x, y, e)
    ax1.scatter(rslist, meff_list; color="black")
    ax1.plot(__x, yfit; label="\$G_0 W_\\pm\$", color="black")
    legend(loc="best")
    ylabel("\$m^* / m\$")
    xlabel("\$r_s\$")
    tight_layout()
    savefig(joinpath(dir, "meff_simion_giuliani.pdf"))
end

function main()
    # System parameters
    beta = 1000.0
    rslist = sort(unique([0.01; collect(range(0.5, 10.0, step=0.5))]))
    big_rslist = sort(unique([[1.0, 2.0, 4.0]; collect(LinRange(0.01, 10.0, 41))]))

    # Output directory
    mkpath("test_simion_giuliani")
    dir = joinpath(@__DIR__, "test_simion_giuliani")

    # Generate plots
    plot_landaufunc(beta, big_rslist; dir=dir)
    plot_meff(beta, rslist; dir=dir)

    # # Extra plots
    # plot_sigma_t0(beta, rslist; dir=dir)
    # plot_sigma_iw0(beta, rslist; dir=dir)
end

main()
