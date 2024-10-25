using CompositeGrids
using ElectronGas
using ElectronLiquid
using JLD2
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

function GW0(G_prev, param, Euv, rtol, Nk, maxK, minK, order, int_type, kgrid::Union{AbstractGrid,AbstractVector,Nothing}=nothing; kwargs...)
    dim = param.dim
    if dim == 2
        if isnothing(kgrid)
            kernel = SelfEnergy.LegendreInteraction.DCKernel_2d(param;
                Euv=Euv, rtol=rtol, Nk=Nk, maxK=maxK, minK=minK, order=order, int_type=int_type, spin_state=:sigma, kwargs...)
        else
            if (kgrid isa AbstractVector)
                kgrid = SimpleG.Arbitrary{eltype(kgrid)}(kgrid)
            end
            kernel = SelfEnergy.LegendreInteraction.DCKernel_2d(param;
                Euv=Euv, rtol=rtol, Nk=Nk, maxK=maxK, minK=minK, order=order, int_type=int_type, spin_state=:sigma, kgrid=kgrid, kwargs...)
        end
        Σ, Σ_ins = SelfEnergy.calcΣ_2d(G_prev, kernel)
    elseif dim == 3
        if isnothing(kgrid)
            kernel = SelfEnergy.LegendreInteraction.DCKernel0(param;
                Euv=Euv, rtol=rtol, Nk=Nk, maxK=maxK, minK=minK, order=order, int_type=int_type, spin_state=:sigma, kwargs...)
        else
            if (kgrid isa AbstractVector)
                kgrid = SimpleG.Arbitrary{eltype(kgrid)}(kgrid)
            end
            kernel = SelfEnergy.LegendreInteraction.DCKernel0(param;
                Euv=Euv, rtol=rtol, Nk=Nk, maxK=maxK, minK=minK, order=order, int_type=int_type, spin_state=:sigma, kgrid=kgrid, kwargs...)
        end
        Σ, Σ_ins = SelfEnergy.calcΣ_3d(G_prev, kernel)
    else
        error("No support for G0W0 in $dim dimension!")
    end

    return Σ, Σ_ins
end

function GW0(G_prev, param, kgrid::Union{AbstractGrid,AbstractVector,Nothing}=nothing; Euv=100 * param.EF, atol=1e-14, Nk=12, maxK=6 * param.kF, minK=1e-8 * param.kF, order=8, int_type=:rpa, kwargs...)
    return GW0(G_prev, param, Euv, atol, Nk, maxK, minK, order, int_type, kgrid; kwargs...)
end

function get_meff_from_Σ_GW0(param::Parameter.Para; int_type=:rpa, δK=5e-6, max_steps=10, atol=1e-7, alpha=0.3, save_sigma=false)
    # Get RPA+FL self-energy
    Σ_tau_dynamic, Σ_tau_instant = get_Σ_GW0(param, int_type, max_steps, atol, alpha, δK, save_sigma)

    # Get effective mass ratio
    meff, kamp = SelfEnergy.massratio(param, Σ_tau_dynamic, Σ_tau_instant, δK)
    @assert kamp ≈ param.kF
    return meff
end

# Helper function for linear interpolation with mixing parameter α
function LERP(M_start, M_end, alpha)
    return (1 - alpha) * M_start + alpha * M_end
end

function get_Σ_GW0(param::Parameter.Para, int_type, max_steps, atol, alpha, δK, save_sigma)
    # Make sigma output directory if needed
    if save_sigma
        mkpath("results/sigma_GW_$(param.dim)d")
        dir = joinpath(@__DIR__, "results/sigma_GW_$(param.dim)d")
    end

    # Make sure we are using parameters for the bare UEG theory
    @assert param.Λs == param.Λa == 0.0

    # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
    rs = round(param.rs; sigdigits=13)
    Fs = get_Fs(rs)
    Fa = get_Fa(rs)
    if int_type in [:ko_const_p, :ko_const_pm]
        _int_type = :ko_const
        if int_type == :ko_const_pm
            println("Fermi liquid parameters at rs = $(rs): Fs = $Fs, Fa = $Fa")
        end
    else
        _int_type = int_type
    end

    # DLR cutoffs
    kF = param.kF
    Euv, rtol_dlr = 1000 * param.EF, 1e-11
    maxK, minK = 20kF, 1e-8kF
    Nk, order = 12, 8

    # Helper function to get the GW self-energy Σ[G, W] for a given int_type (W)
    # NOTE: A large momentum grid is required for G and Σ at intermediate steps
    function Σ_GW0(G)
        return GW0(
            G,
            param;
            # kgrid;
            Euv=Euv,
            rtol=rtol_dlr,
            Nk=Nk,
            maxK=maxK,
            minK=minK,
            order=order,
            int_type=_int_type,
            Fs=_int_type == :ko_const ? -Fs : -0.0,   # NOTE: NEFT uses opposite sign convention for F!
            Fa=int_type == :ko_const_pm ? -Fa : -0.0,  # NOTE: NEFT uses opposite sign convention for F!
        )
    end

    # G0 for the UEG; a large kgrid is required for the self-consistency loop
    kgrid = CompositeGrid.LogDensedGrid(:cheb, [0.0, maxK], [0.0, param.kF], Nk, minK, order)
    G0 = SelfEnergy.G0wrapped(Euv, rtol_dlr, kgrid, param)

    # Initial G0W self-energy: Σ[G0, W]
    Σ, Σ_ins = Σ_GW0(G0)
    G_prev = G0
    δμ_prev = SelfEnergy.chemicalpotential(param, Σ, Σ_ins)
    meff_prev, kamp = SelfEnergy.massratio(param, Σ, Σ_ins, δK)
    zfactor_prev = SelfEnergy.zfactor(param, Σ; ngrid=[-1, 0])[1]
    @assert kamp ≈ param.kF

    # Write initial (G0W0) self-energy to JLD2 file, overwriting if it already exists
    if save_sigma
        jldopen(joinpath(dir, "sigma_$(int_type)_rs$(rs).jl"), "w") do file
            file["Σ_0"] = Σ
            file["Σ_ins_0"] = Σ_ins
        end
    end

    # Self-consistency loop for G
    println("\nBegin self-consistency loop...\n")
    i_step = 0
    while i_step < max_steps
        δμ = SelfEnergy.chemicalpotential(param, Σ, Σ_ins)
        meff = SelfEnergy.massratio(param, Σ, Σ_ins, δK)[1]
        zfactor = SelfEnergy.zfactor(param, Σ; ngrid=[-1, 0])[1]

        print("""\n
        Step $(i_step + 1):
        • m*/m        = \t$(meff)
        • Z           = \t$(zfactor)
        • δμ          = \t$(δμ)
        """)

        # Use Dyson's equation to find G = 1 / (G0⁻¹ - Σ)
        # NOTE: Σ and Σ_ins are converted to the Matsubara representation internally
        G = SelfEnergy.Gwrapped(Σ, Σ_ins, param)

        # Get Σ[G_mix, W], where the input Green's function is a linear interpolation
        # of the previous and current G: G_mix = (1 - α) * G_prev + α * G
        G_mix = LERP(G_prev, G, alpha)
        Σ, Σ_ins = Σ_GW0(G_mix)

        # Append self-energy at this step to JLD2 file
        if save_sigma
            jldopen(joinpath(dir, "sigma_$(int_type)_rs$(rs).jl"), "a+") do file
                file["Σ_$(i_step + 1)"] = Σ
                file["Σ_ins_$(i_step + 1)"] = Σ_ins
            end
        end

        # Test for convergence of quasiparticle properties
        if i_step > 0
            dmeff = abs(meff - meff_prev)
            dzfactor = abs(zfactor - zfactor_prev)
            ddeltamu = abs(δμ - δμ_prev)
            print("""
            • |Δ(m* / m)| = \t$(dmeff)
            • |Δ(Z)|      = \t$(dzfactor)
            • |Δ(δμ)|     = \t$(ddeltamu)
            """)
            if all([dmeff, dzfactor, ddeltamu] .< atol)
                println("\nConverged to atol = $atol after $i_step steps!")
                break
            end
        end

        # Prepare for next iteration
        i_step += 1
        # G_prev = G
        G_prev = G_mix
        δμ_prev = δμ
        meff_prev = meff
        zfactor_prev = zfactor
    end
    if i_step == max_steps
        println("\nWARNING: Convergence to atol = $atol not reached after $max_steps steps!")
    end
    return Σ, Σ_ins
end

function get_meff_from_Σ_G0W0(param::Parameter.Para; δK=5e-6, int_type=:rpa)
    # Make sure we are using parameters for the bare UEG theory
    @assert param.Λs == param.Λa == 0.0

    # DLR cutoffs
    kF = param.kF
    Euv, rtol_dlr = 1000 * param.EF, 1e-14
    # Euv, rtol_dlr = 100 * param.EF, 1e-11
    maxK, minK = 6kF, 1e-8kF
    Nk, order = 12, 8

    # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
    rs = round(param.rs; sigdigits=13)
    Fs = get_Fs(rs)
    Fa = get_Fa(rs)
    if int_type in [:ko_const_p, :ko_const_pm]
        _int_type = :ko_const
        if int_type == :ko_const_pm
            println("Fermi liquid parameters at rs = $(rs): Fs = $Fs, Fa = $Fa")
        end
    else
        _int_type = int_type
    end

    # Get RPA+FL self-energy
    Σ_tau_dynamic, Σ_tau_instant = SelfEnergy.G0W0(
        param;
        # kgrid;
        Euv=Euv,
        rtol=rtol_dlr,
        Nk=Nk,
        maxK=maxK,
        minK=minK,
        order=order,
        int_type=_int_type,
        Fs=_int_type == :ko_const ? -Fs : -0.0,   # NOTE: NEFT uses opposite sign convention for F!
        Fa=int_type == :ko_const_pm ? -Fa : -0.0,  # NOTE: NEFT uses opposite sign convention for F!
    )

    # Get effective mass ratio
    meff, kamp = SelfEnergy.massratio(param, Σ_tau_dynamic, Σ_tau_instant, δK)
    zfactor = SelfEnergy.zfactor_new(param, Σ_tau_dynamic)[1]
    @assert kamp ≈ param.kF
    return meff, zfactor
end

function main()
    # System parameters
    dim = 3
    # beta = 100.0
    beta = 1000.0
    δK = 5e-6
    # GW parameters
    max_steps = 100
    atol = 1e-7
    alpha = 0.3

    calc_g0w = true
    calc_gw = false
    save_sigma = false

    # rslist = [0.001; collect(LinRange(0.0, 1.1, 111))[2:end]]  # for accurate 2D HDL
    # rslist = [0.005; collect(LinRange(0.0, 5.0, 101))[2:end]]  # for 2D
    rslist = [0.01; collect(LinRange(0.0, 10.0, 101))[2:end]]  # for 3D

    # rslist = [0.001; collect(LinRange(0.0, 0.25, 5))[2:end]]
    # rslist = [0.001; collect(range(0.0, 1.1, step=0.05))[2:end]]  # for accurate 2D HDL
    # rslist = [0.01; collect(range(0.0, 10.0, step=0.5))[2:end]]  # for 3D
    # rslist = [1.0, 3.0]

    # NOTE: int_type ∈ [:ko_const, :ko_takada_plus, :ko_takada, :ko_moroni, :ko_simion_giuliani] 
    # NOTE: KO interaction using G+ and/or G- is currently only available in 3D
    int_type_Gp = :ko_const_p
    int_type_Gpm = :ko_const_pm
    # int_type_Gp = :ko_moroni
    # int_type_Gpm = :ko_simion_giuliani
    @assert int_type_Gp ∈ [:ko_const_p, :ko_takada_plus, :ko_moroni]
    @assert int_type_Gpm ∈ [:ko_const_pm, :ko_takada, :ko_simion_giuliani]

    if int_type_Gpm == :ko_const_pm
        ko_dirstr = "const"
    elseif int_type_Gpm == :ko_takada
        ko_dirstr = "takada"
    elseif int_type_Gpm == :ko_simion_giuliani
        ko_dirstr = "simion_giuliani"
    end

    # Output directory
    dir = joinpath(@__DIR__, "results/forlqsgw/$(dim)d")
    mkpath(joinpath(dir, "rpa"))
    mkpath(joinpath(dir, ko_dirstr))

    # G0W effective masses
    if calc_g0w
        m_Σ_G0W0 = []
        z_Σ_G0W0 = []
        if dim == 3
            m_Σ_G0Wp = []
            m_Σ_G0Wpm = []
            z_Σ_G0Wp = []
            z_Σ_G0Wpm = []
        end
        for rs in rslist
            param = Parameter.rydbergUnit(1.0 / beta, rs, dim)

            print("Calculating m*/m[G0W] values for rs = $rs...")
            m_Σ_G0W0_this_rs, z_Σ_G0W0_this_rs = get_meff_from_Σ_G0W0(param; δK=δK, int_type=:rpa)
            if dim == 3
                m_Σ_G0Wp_this_rs, z_Σ_G0Wp_this_rs = get_meff_from_Σ_G0W0(param; δK=δK, int_type=int_type_Gp)
                m_Σ_G0Wpm_this_rs, z_Σ_G0Wpm_this_rs = get_meff_from_Σ_G0W0(param; δK=δK, int_type=int_type_Gpm)
            end
            println("done.")
            println("(G0W0) m*/m = $m_Σ_G0W0_this_rs, z = $z_Σ_G0W0_this_rs")
            if dim == 3
                println("(G0Wp) m*/m = $m_Σ_G0Wp_this_rs, z = $z_Σ_G0Wp_this_rs")
                println("(G0Wpm) m*/m = $m_Σ_G0Wpm_this_rs, z = $z_Σ_G0Wpm_this_rs")
            end

            push!(m_Σ_G0W0, m_Σ_G0W0_this_rs)
            push!(z_Σ_G0W0, z_Σ_G0W0_this_rs)
            if dim == 3
                push!(m_Σ_G0Wp, m_Σ_G0Wp_this_rs)
                push!(m_Σ_G0Wpm, m_Σ_G0Wpm_this_rs)
                push!(z_Σ_G0Wp, z_Σ_G0Wp_this_rs)
                push!(z_Σ_G0Wpm, z_Σ_G0Wpm_this_rs)
            end
        end
        # Add points at rs = 0
        pushfirst!(rslist, 0.0)
        pushfirst!(m_Σ_G0W0, 1.0)
        pushfirst!(z_Σ_G0W0, 1.0)
        if dim == 3
            pushfirst!(m_Σ_G0Wp, 1.0)
            pushfirst!(m_Σ_G0Wpm, 1.0)
            pushfirst!(z_Σ_G0Wp, 1.0)
            pushfirst!(z_Σ_G0Wpm, 1.0)
        end
        # Save data
        np.savez(joinpath(dir, "rpa/meff_$(dim)d_sigma_G0W0.npz"), rslist=rslist, mefflist=m_Σ_G0W0, zlist=z_Σ_G0W0)
        if dim == 3
            np.savez(joinpath(dir, "$(ko_dirstr)/meff_$(dim)d_sigma_G0Wp.npz"), rslist=rslist, mefflist=m_Σ_G0Wp, zlist=z_Σ_G0Wp)
            np.savez(joinpath(dir, "$(ko_dirstr)/meff_$(dim)d_sigma_G0Wpm.npz"), rslist=rslist, mefflist=m_Σ_G0Wpm, zlist=z_Σ_G0Wpm)
        end
    end

    # GW effective masses
    if calc_gw
        # m_Σ_GW0 = []
        if dim == 3
            m_Σ_GWp = []
            m_Σ_GWpm = []
        end
        _rslist = calc_g0w ? rslist[2:end] : rslist
        for rs in _rslist
            param = Parameter.rydbergUnit(1.0 / beta, rs, dim)

            print("Calculating m*/m[GW] values for rs = $rs...")
            # m_Σ_GW0_this_rs = get_meff_from_Σ_GW0(param; int_type=:rpa, δK=δK,
            #     max_steps=max_steps, atol=atol, alpha=alpha, save_sigma=save_sigma)
            if dim == 3
                m_Σ_GWp_this_rs = get_meff_from_Σ_GW0(param; int_type=int_type_Gp, δK=δK,
                    max_steps=max_steps, atol=atol, alpha=alpha)
                m_Σ_GWpm_this_rs = get_meff_from_Σ_GW0(param; int_type=int_type_Gpm, δK=δK,
                    max_steps=max_steps, atol=atol, alpha=alpha)
            end
            println("done.")

            # push!(m_Σ_GW0, m_Σ_GW0_this_rs)
            if dim == 3
                push!(m_Σ_GWp, m_Σ_GWp_this_rs)
                push!(m_Σ_GWpm, m_Σ_GWpm_this_rs)
            end
        end
        # Add points at rs = 0
        if calc_g0w == false
            pushfirst!(rslist, 0.0)
        end
        # pushfirst!(m_Σ_GW0, 1.0)
        if dim == 3
            pushfirst!(m_Σ_GWp, 1.0)
            pushfirst!(m_Σ_GWpm, 1.0)
        end
        # Save data
        # np.savez(joinpath(dir, "rpa/meff_$(dim)d_sigma_GW0.npz"), rslist=rslist, mefflist=m_Σ_GW0)
        if dim == 3
            np.savez(joinpath(dir, "$(ko_dirstr)/meff_$(dim)d_sigma_GWp.npz"), rslist=rslist, mefflist=m_Σ_GWp)
            np.savez(joinpath(dir, "$(ko_dirstr)/meff_$(dim)d_sigma_GWpm.npz"), rslist=rslist, mefflist=m_Σ_GWpm)
        end
    end
end

main()
