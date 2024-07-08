using ElectronLiquid, UniElectronGas
using JLD2, DelimitedFiles
using Measurements

dim = 3
spin = 2
# rs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# rs = [1.0,]
# rs = [2.0,]
# rs = [3.0,]
rs = [4.0,]
# rs = [1.0, 2.0, 3.0, 4.0]
# Fs = -[0.223, 0.380, 0.516, 0.639, 0.752]
# Fs = -[0.223,]
# rs = [1.0,]
# mass2 = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
# mass2 = [1.0, 2.0, 3.0, 4.0, 5.0]
# mass2 = [9.0, 13.0, 19.0, 27.0]
# mass2 = [1e-3,]
# mass2 = [3.5,]
# mass2 = [2.0,]
# mass2 = [1.5,]
mass2 = [1.0,]
# mass2 = [2.44355,]
# mass2 = [1.22177,]
# mass2 = [0.814516,]
# mass2 = [0.610887,]
# mass2 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
# mass2 = [3.5,]
# Fs = [-0.0,]
Fs = -0.0 .* rs
beta = [50,]
# beta = [25.0, 50.0, 100.0]
# order = [5, 6,]
order = [6,]
Nl = 1
# isDynamic = true
isDynamic = false
isFock = false
ω_c = 0.1

# mstar = [0.96537 ± 0.00011 0.95292 ± 0.00014 0.94942 ± 0.00016 0.94914 ± 0.0002 0.94954 ± 0.00034]
# dmstar = -[(i == 0 ? 1.0 : mstar[i]) - mstar[i+1] for i in 0:length(mstar)-1]
# dmstar = dmstar .* 0.0
# 1.0
# dmstar = [0.965154 ± 3.5e-5 - 1.0, -0.012734 ± 2.6e-5, -0.003684 ± 2.6e-5, -1.7e-5 ± 3.3e-5, 0.000808 ± 5.4e-5]
# 2.0
# dmstar = [0.971212 ± 3.6e-5 - 1.0, -0.013191 ± 3.4e-5, -0.006643 ± 3.8e-5, -0.00133 ± 5.3e-5, 0.00119 ± 0.0001]
# 3.0
# dmstar = [0.978121 ± 3.4e-5 - 1.0, -0.01011 ± 3.3e-5, -0.007079 ± 3.8e-5, -0.002236 ± 5.7e-5, 0.00094 ± 0.00012]
# 4.0
dmstar = [0.977115 ± 3.9e-5 - 1.0, -0.006129 ± 3.9e-5, -0.005259 ± 5.1e-5, 0.000533 ± 9.1e-5, 0.00322 ± 0.00024]
mstar = [1.0 + sum(dmstar[1:i]) for i in 1:length(dmstar)]

const parafilename = "para_wn_1minus0.csv"
# const filename = "data_ver4PP_profile.jld2"
# const filename = "data_ver4PP_new.jld2"
# const filename = "data_ver4PP_parqAD.jld2"
# const filename = "data_ver4PP_parqAD_rs2.jld2"
# const filename = "data_ver4PP_parqAD_rs3.jld2"
const filename = "data_ver4PP_parqAD_rs4.jld2"
# const filename = "data_ver4PP_parqAD_rs1_correctn.jld2"
# const filename = "data_ver4PP_parqAD_rs1_v2.jld2"
# const filename = "data_ver4PP_parqAD_sugon.jld2"
# const filename = "data_ver4PP_parqAD_newsamp.jld2"
# const filename = "data_ver4PP_parqAD_newsamp_l10.jld2"
# const filename = "data_ver4PP_beta200.jld2"
# const filename = "data_ver4PP.jld2"
# const savefilename1 = "guu_$(dim)d.dat"
# const savefilename2 = "gud_$(dim)d.dat"
# const savefilename1 = "gsko_$(dim)d.dat"
# const savefilename2 = "gako_$(dim)d.dat"
# const savefilename1 = "gsrpa_$(dim)d.dat"
# const savefilename2 = "garpa_$(dim)d.dat"
# const savefilename1 = "gsyuk3_$(dim)d.dat"
# const savefilename2 = "gayuk3_$(dim)d.dat"
# const savefilename1 = "gsingyuk3_$(dim)d_01.dat"
# const savefilename2 = "gtripyuk3_$(dim)d_01.dat"
const savefilename1 = "gsingyuk3_$(dim)correctednomass_01.dat"
const savefilename2 = "gtripyuk3_$(dim)correctednomass_01.dat"
# const savefilename1 = "gsingrpa_$(dim)d.dat"
# const savefilename2 = "gtriprpa_$(dim)d.dat"

function shift_u(u, wc1, wc2)
    # shift u from wc1 to wc2
    return u ./ (1 .+ u .* log(wc1 / wc2))#, uerr ./ (1 .+ u .* log(wc1 / wc2)) .^ 2
end

function Πs(para; ω_c=0.1)
    return log(0.882 * ω_c * para.beta)
    # return log(ω_c * para.beta)
end

function Un(Γlist4, Π, n)
    Γlist = Γlist4 ./ 4
    for i in 1:length(Γlist)
        result = Γlist4[i] / 4
        for j in 1:i-1
            result += Γlist4[j] / 4 * dmstar[i-j]
        end
        Γlist[i] = result
    end
    result = 0.0
    if n == 1
        result += Γlist[1]
    elseif n == 2
        result += Γlist[1]
        result += Γlist[2] - Γlist[1]^2 * Π
    elseif n == 3
        result += Γlist[1]
        result += Γlist[2] - Γlist[1]^2 * Π
        result += Γlist[3] - 2 * Γlist[1] * Π * Γlist[2] + Γlist[1]^3 * Π^2
    elseif n == 4
        result += Γlist[1]
        result += Γlist[2] - Γlist[1]^2 * Π
        result += Γlist[3] - 2 * Γlist[1] * Π * Γlist[2] + Γlist[1]^3 * Π^2
        result += Γlist[4] - 2 * Γlist[1] * Π * Γlist[3] - Γlist[2]^2 * Π + 3 * Γlist[1]^2 * Π^2 * Γlist[2] - Γlist[1]^4 * Π^3
    elseif n == 5
        result += Γlist[1]
        result += Γlist[2] - Γlist[1]^2 * Π
        result += Γlist[3] - 2 * Γlist[1] * Π * Γlist[2] + Γlist[1]^3 * Π^2
        result += Γlist[4] - 2 * Γlist[1] * Π * Γlist[3] - Γlist[2]^2 * Π + 3 * Γlist[1]^2 * Π^2 * Γlist[2] - Γlist[1]^4 * Π^3
        result += (Γlist[5] - 2 * (Γlist[1] * Π * Γlist[4] + Γlist[2] * Π * Γlist[3])
                   +
                   3 * (Γlist[1]^2 * Π^2 * Γlist[3] + Γlist[2]^2 * Π^2 * Γlist[1])
                   -
                   4 * Γlist[1]^3 * Π^3 * Γlist[2] + Γlist[1]^5 * Π^4)
    elseif n == 6
        result += Γlist[1]
        result += Γlist[2] - Γlist[1]^2 * Π
        result += Γlist[3] - 2 * Γlist[1] * Π * Γlist[2] + Γlist[1]^3 * Π^2
        result += Γlist[4] - 2 * Γlist[1] * Π * Γlist[3] - Γlist[2]^2 * Π + 3 * Γlist[1]^2 * Π^2 * Γlist[2] - Γlist[1]^4 * Π^3
        result += (Γlist[5] - 2 * (Γlist[1] * Π * Γlist[4] + Γlist[2] * Π * Γlist[3])
                   +
                   3 * (Γlist[1]^2 * Π^2 * Γlist[3] + Γlist[2]^2 * Π^2 * Γlist[1])
                   -
                   4 * Γlist[1]^3 * Π^3 * Γlist[2] + Γlist[1]^5 * Π^4)
        result += (Γlist[6] - 2 * (Γlist[1] * Π * Γlist[5] + Γlist[2] * Π * Γlist[4]) - Γlist[3] * Π * Γlist[3]
                   +
                   (3 * Γlist[1]^2 * Π^2 * Γlist[4] + 6 * Γlist[1] * Π * Γlist[2] * Π * Γlist[3] + Γlist[2] * Π * Γlist[2] * Π * Γlist[2])
                   -
                   4 * Γlist[1]^3 * Π^3 * Γlist[3] - 6 * Γlist[1]^2 * Π^3 * Γlist[2]^2
                   +
                   5 * Γlist[1]^4 * Π^4 * Γlist[2]
                   -
                   Γlist[1]^6 * Π^5)

    end
    return -result
end

function Γ2U(Γlist, para; ω_c=0.1)
    println(para.EF, para.beta, para.β)
    # Γlist .= -Γlist
    # Γlist = [(-1)^i * Γlist[i] for i in 1:length(Γlist)]
    Π = Πs(para; ω_c=ω_c)
    return [Un(Γlist, Π, n) for n in 1:length(Γlist)]
end

if abspath(PROGRAM_FILE) == @__FILE__
    isSave = false
    if length(ARGS) >= 1 && (ARGS[1] == "s" || ARGS[1] == "-s" || ARGS[1] == "--save" || ARGS[1] == " save")
        # the second parameter may be set to save the derived parameters
        isSave = true
    end

    f = jldopen(filename, "r")
    results_s, results_a = Any[], Any[]
    # for (_rs, _mass2, _F, _beta, _order) in Iterators.product(rs, mass2, Fs, beta, order)
    for (irs, _mass2, _beta, _order) in Iterators.product([i for i in 1:length(rs)], mass2, beta, order)
        _F = Fs[irs]
        _rs = rs[irs]
        para = ParaMC(rs=_rs, beta=_beta, Fs=_F, order=_order, mass2=_mass2, isDynamic=isDynamic, isFock=isFock, dim=dim, spin=spin)
        kF = para.kF
        for key in keys(f)
            loadpara = ParaMC(key)
            if UEG.paraid(loadpara) == UEG.paraid(para)
                println(UEG.paraid(para))
                data_Fs, data_Fa = UniElectronGas.getVer4PHl(para, filename)
                # data_uu, data_ud = (data_Fs + data_Fa), (data_Fs - data_Fa)
                res_s = Any[_rs, _beta, _mass2, _order]
                for il in 1:Nl
                    # push!(results_s, append!(Any[_rs, _beta, _mass2, _order, il-1], Γ2U([real(data_uu[o][il, 1]) for o in 1:_order], para; ω_c=ω_c)))
                    # push!(results_a, append!(Any[_rs, _beta, _mass2, _order, il-1], Γ2U([real(data_ud[o][il, 1]) for o in 1:_order], para; ω_c=ω_c)))
                    if _order == 4
                        push!(results_s, append!(Any[_rs, _beta, _mass2, _order+1, il-1], Γ2U([real(data_Fs[o][il, 1] - 3 * data_Fa[o][il, 1]) for o in 1:_order], para; ω_c=ω_c), [0.0 * real(data_Fa[1][1, 1])]))
                        push!(results_a, append!(Any[_rs, _beta, _mass2, _order+1, il-1], Γ2U([real(data_Fs[o][il, 1] + 3 * data_Fa[o][il, 1]) for o in 1:_order], para; ω_c=ω_c), [0.0 * real(data_Fa[1][1, 1])]))
                    else
                        push!(results_s, append!(Any[_rs, _beta, _mass2, _order, il-1], Γ2U([real(data_Fs[o][il, 1] - 3 * data_Fa[o][il, 1]) for o in 1:_order], para; ω_c=ω_c)))
                        push!(results_a, append!(Any[_rs, _beta, _mass2, _order, il-1], Γ2U([real(data_Fs[o][il, 1] + 3 * data_Fa[o][il, 1]) for o in 1:_order], para; ω_c=ω_c)))
                    end
                end
            end
        end
    end

    if isSave
        open(savefilename1, "a+") do io
            writedlm(io, results_s)
        end
        open(savefilename2, "a+") do io
            writedlm(io, results_a)
        end
    end
end