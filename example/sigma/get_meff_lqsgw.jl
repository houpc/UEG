using BenchmarkTools
using CompositeGrids
using ElectronGas
using ElectronLiquid
using GreenFunc
using JLD2
using Lehmann
using Parameters
using PyCall

@pyimport numpy as np   # for saving/loading numpy data

@inline function get_Fs(rs)
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
@inline function get_Fs_DMC(rs)
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

"""
Get the antisymmetric l=0 Fermi-liquid parameter F⁰ₐ via interpolation of the 
susceptibility ratio data (c.f. Kukkonen & Chen, 2021)
"""
@inline function get_Fa(rs)
    chi0_over_chi = 0.9821 - 0.1232rs + 0.0091rs^2
    # F⁰ₐ = χ₀/χ - 1
    return chi0_over_chi - 1.0
end

function main()
    # System parameters
    dim = 3
    rs = 1.0
    δK = 5e-6
    beta = 40.0
    # beta = 100.0;
    # beta = 1000.0;
    param = Parameter.rydbergUnit(1.0 / beta, rs, dim)

    # GW parameters
    max_steps = 1
    # max_steps = 100
    atol = 1e-7
    alpha = 0.3
    save = true

    # Calculate local field factor cases?
    calc_pm = false
    if dim != 3
        @assert calc_pm = false "Local field factor calculations are only available in 3D!"
    end

    @unpack β, kF, EF = param

    Euv = 100 * EF
    rtol = 1e-14
    maxK = 10 * kF
    minK = 1e-8 * kF

    # Minimally sufficient grid
    Nk = 14
    order = 10

    # Nk = 16;
    # order = 12;

    # Nk = 18;
    # order = 14;

    # Nk = 20;
    # order = 16;

    # Output directory
    mkpath("results/finalized_meff_results/$(dim)d")
    dir = joinpath(@__DIR__, "results/finalized_meff_results/$(dim)d")

    rslist = [1.0]

    # # rslist = [0.001; collect(LinRange(0.0, 1.1, 111))[2:end]]  # for accurate 2D HDL
    # # rslist = [0.005; collect(LinRange(0.0, 5.0, 101))[2:end]]  # for 2D
    # # rslist = [0.01; collect(LinRange(0.0, 10.0, 101))[2:end]]  # for 3D

    # rslist = [0.001; collect(LinRange(0.0, 0.25, 5))[2:end]]
    # # rslist = [0.001; collect(range(0.0, 1.1, step=0.05))[2:end]]  # for accurate 2D HDL
    # # rslist = [0.01; collect(range(0.0, 10.0, step=0.5))[2:end]]  # for 3D
    # # rslist = [1.0, 3.0]

    # NOTE: int_type ∈ [:ko_const, :ko_takada_plus, :ko_takada, :ko_moroni, :ko_simion_giuliani] 
    # NOTE: KO interaction using G+ and/or G- is currently only available in 3D
    # int_type_Gp = :ko_const_p
    # int_type_Gpm = :ko_const_pm
    int_type_Gp = :ko_moroni
    int_type_Gpm = :ko_simion_giuliani
    @assert int_type_Gp ∈ [:ko_const_p, :ko_takada_plus, :ko_moroni]
    @assert int_type_Gpm ∈ [:ko_const_pm, :ko_takada, :ko_simion_giuliani]

    if int_type_Gpm == :ko_const_pm
        ko_dirstr = "const"
    elseif int_type_Gpm == :ko_takada
        ko_dirstr = "takada"
    elseif int_type_Gpm == :ko_simion_giuliani
        ko_dirstr = "simion_giuliani"
    end

    # LQSGW effective masses
    m_Σ_LQSGW0 = []
    if calc_pm
        m_Σ_LQSGWp = []
        m_Σ_LQSGWpm = []
    end
    for rs in rslist
        param = Parameter.rydbergUnit(1.0 / beta, rs, dim)

        # Make sure we are using parameters for the bare UEG theory
        @assert param.Λs == param.Λa == 0.0

        # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
        _rs = round(param.rs; sigdigits=13)
        Fs = -1.0 * get_Fs(_rs)  # NOTE: NEFT uses opposite sign convention for F!
        Fa = -1.0 * get_Fa(_rs)  # NOTE: NEFT uses opposite sign convention for F!

        print("Calculating LQSGW m*/m values for rs = $(_rs)...")
        m_Σ_LQSGW0_this_rs = LQSGW.get_meff_from_Σ_LQSGW(
            param;
            int_type=:rpa,
            Fs=Fs,
            Fa=Fa,
            δK=δK,
            max_steps=max_steps,
            atol=atol,
            alpha=alpha,
            verbose=true,
            save=true,
            savedir=joinpath(dir, "rpa"),
        )
        if calc_pm
            m_Σ_LQSGWp_this_rs = LQSGW.get_meff_from_Σ_LQSGW(
                param;
                int_type=int_type_Gp,
                Fs=Fs,
                Fa=Fa,
                δK=δK,
                max_steps=max_steps,
                atol=atol,
                alpha=alpha,
                verbose=true,
                save=true,
                savedir=joinpath(dir, "$ko_dirstr"),
            )
            m_Σ_LQSGWpm_this_rs = LQSGW.get_meff_from_Σ_LQSGW(
                param;
                int_type=int_type_Gpm,
                Fs=Fs,
                Fa=Fa,
                δK=δK,
                max_steps=max_steps,
                atol=atol,
                alpha=alpha,
                verbose=true,
                save=true,
                savedir=joinpath(dir, "$ko_dirstr"),
            )
        end
        println("done.")
        push!(m_Σ_LQSGW0, m_Σ_LQSGW0_this_rs)
        if calc_pm
            push!(m_Σ_LQSGWp, m_Σ_LQSGWp_this_rs)
            push!(m_Σ_LQSGWpm, m_Σ_LQSGWpm_this_rs)
        end
    end
    # Add points at rs = 0
    pushfirst!(rslist, 0.0)
    pushfirst!(m_Σ_LQSGW0, 1.0)
    if calc_pm
        pushfirst!(m_Σ_LQSGWp, 1.0)
        pushfirst!(m_Σ_LQSGWpm, 1.0)
    end
    # Save data
    np.savez(joinpath(dir, "rpa/meff_$(dim)d_sigma_LQSGW0.npz"), rslist=rslist, mefflist=m_Σ_LQSGW0)
    if calc_pm
        np.savez(joinpath(dir, "$(ko_dirstr)/meff_$(dim)d_sigma_LQSGWp.npz"), rslist=rslist, mefflist=m_Σ_LQSGWp)
        np.savez(joinpath(dir, "$(ko_dirstr)/meff_$(dim)d_sigma_LQSGWpm.npz"), rslist=rslist, mefflist=m_Σ_LQSGWpm)
    end
end

main()
