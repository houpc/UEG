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

abstract type AbstractFitCoeffs end

# Perdew-Wang fit coefficients for α_c(rs), α^{RPA}_c(rs), E_c(rs), and E^{RPA}_c(rs).
# See Table I of J. P. Perdew and Y. Wang, Phys. Rev. B 45, 13244 (1992) (doi.org/10.1103/PhysRevB.45.13244).
struct PerdewWangFitCoeffs <: AbstractFitCoeffs
    p::Float64
    Ac::Float64
    α1::Float64
    β1::Float64
    β2::Float64
    β3::Float64
    β4::Float64
end

# Perdew-Wang interpolating function for α_c(rs), α^{RPA}_c(rs), E_c(rs), and E^{RPA}_c(rs)
@inline function perdew_wang_interpolant(fitcoeffs::PerdewWangFitCoeffs)
    @unpack p, Ac, α1, β1, β2, β3, β4 = fitcoeffs
    interpolant(rs) = -4Ac * (1 + α1 * rs) * log(1 + 1 / (2Ac * (β1 * sqrt(rs) + β2 * rs + β3 * rs^(3 / 2.0) + β4 * rs^(p + 1.0))))
    return interpolant
end

# Perdew-Wang fit to E^{RPA}_c(rs) for ζ = 0 (paramagnetic phase)
@inline const E_corr_rpa_z0 = perdew_wang_interpolant(
    PerdewWangFitCoeffs(0.75, 0.031091, 0.082477, 5.1486, 1.6483, 0.23647, 0.20614)
)

# Perdew-Wang fit to E^{RPA}_c(rs) for ζ = 1 (ferromagnetic phase)
@inline const E_corr_rpa_z1 = perdew_wang_interpolant(
    PerdewWangFitCoeffs(0.75, 0.015545, 0.035374, 6.4869, 1.3083, 0.1518, 0.082349)
)

# Perdew-Wang fit to -α^{RPA}_c(rs)
@inline const negative_spin_stiffness_rpa = perdew_wang_interpolant(
    PerdewWangFitCoeffs(1.0, 0.016887, 0.028829, 10.357, 3.6231, 0.4799, 0.12279)
)

# Perdew-Wang fit to E_c(rs) for ζ = 0 (paramagnetic phase)
@inline const E_corr_z0 = perdew_wang_interpolant(
    PerdewWangFitCoeffs(1.0, 0.031091, 0.2137, 7.5957, 3.5876, 1.6382, 0.49294)
)

# Perdew-Wang fit to E_c(rs) for ζ = 1 (ferromagnetic phase)
@inline const E_corr_z1 = perdew_wang_interpolant(
    PerdewWangFitCoeffs(1.0, 0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
)

# Perdew-Wang fit to -α_c(rs)
@inline const negative_spin_stiffness = perdew_wang_interpolant(
    PerdewWangFitCoeffs(1.0, 0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
)

@inline function spin_susceptibility_enhancement(rs)
    alpha_ueg = (4 / 9π)^(1 / 3)
    # return 1 - (alpha_ueg * rs / π) + 3(alpha_ueg * rs)^2 * spin_stiffness(rs)
    # return 1 - (alpha_ueg * rs / π) - 3(alpha_ueg * rs)^2 * negative_spin_stiffness(rs)
    return 1 - (alpha_ueg * rs / π) - 3(alpha_ueg * rs)^2 * negative_spin_stiffness(rs) / 2
end

@inline function negative_spin_stiffness_vosko(rs)
    # Linear interpolation of full negative spin stiffness using full and RPA correlation energies
    slope = ((E_corr_z1(rs) - E_corr_z0(rs)) / (E_corr_rpa_z1(rs) - E_corr_rpa_z0(rs)))
    return slope * negative_spin_stiffness_rpa(rs)
end

@inline function spin_susceptibility_enhancement_vosko(rs)
    alpha_ueg = (4 / 9π)^(1 / 3)
    # return 1 - (alpha_ueg * rs / π) + 3(alpha_ueg * rs)^2 * spin_stiffness(rs)
    return 1 - (alpha_ueg * rs / π) - 3(alpha_ueg * rs)^2 * negative_spin_stiffness_vosko(rs)
end

function plot_spin_susceptibility_enhancement(; dir=@__DIR__)
    rs = collect(LinRange(0, 10, 501))
    fig = figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot(rs, spin_susceptibility_enhancement.(rs); label="Perdew \\& Wang", color=cdict["orange"])
    # ax.plot(rs, spin_susceptibility_enhancement_vosko.(rs); label="Perdew \\& Wang (v2)", color=cdict["blue"], linestyle="--")

    # ax.plot(rs, 1.0 ./ spin_susceptibility_enhancement.(rs); label="Perdew \\& Wang", color=cdict["orange"])
    # ax.plot(rs, 1.0 ./ spin_susceptibility_enhancement_vosko.(rs); label="Vosko et al.", color=cdict["blue"], linestyle="--")
    ax.set_xlabel("\$r_s\$")
    ax.set_ylabel("\$\\chi_0 / \\chi\$")
    # ax.set_ylabel("\$\\chi / \\chi_0\$")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(joinpath(dir, "spin_susceptibility_enhancement.pdf"))
end

function plot_negative_spin_stiffness(; dir=@__DIR__)
    rs = collect(LinRange(0, 15, 501))
    fig = figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot(rs, negative_spin_stiffness.(rs); label="Perdew \\& Wang", color=cdict["orange"])
    # ax.plot(rs, negative_spin_stiffness_vosko.(rs); label="Perdew \\& Wang (v2)", color=cdict["blue"], linestyle="--")

    # Reported values
    rs_vosko = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0, 15.0]
    data_vosko_mRy = [97.98, 77.10, 57.86, 47.62, 40.92, 36.09, 32.39, 28.18, 23.29, 17.45]
    data_vosko_Ry = data_vosko_mRy / 1000

    # TODO: Note the extra factor of 1/2: does this mean we need to convert from Hartree back to Rydberg for Perdew-Wang fits?
    ax.scatter(rs_vosko, -data_vosko_Ry; color=cdict["blue"], marker="o")
    # ax.scatter(rs_vosko, -0.5 * data_vosko_Ry; color=cdict["blue"], marker="o")
    # spline fit
    x = rs_vosko
    y = -data_vosko_Ry
    # y = -0.5 * data_vosko_Ry
    e = 1e-5 * ones(length(x))
    __x, yfit = spline(x, y, e; xmin=0.0, xmax=15.0)
    ax.plot(__x, yfit; color=cdict["blue"], linestyle="--", label="Vosko et al.")

    ax.set_xlabel("\$r_s\$")
    ax.set_ylabel("\$-\\alpha_c(r_s)\$ (Ry)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(joinpath(dir, "negative_spin_stiffness.pdf"))
end

function main()
    # Output directory
    mkpath("test_simion_giuliani")
    dir = joinpath(@__DIR__, "test_simion_giuliani")

    # Generate plots
    plot_spin_susceptibility_enhancement(dir=dir)
    plot_negative_spin_stiffness(dir=dir)
end

main()
