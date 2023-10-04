using ElectronLiquid, UniElectronGas
using JLD2, DelimitedFiles

dim = 3
spin = 2
# rs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# rs = [1.0, 2.0, 3.0, 4.0, 5.0]
# Fs = -[0.223, 0.380, 0.516, 0.639, 0.752]
# Fs = -[0.223,]
rs = [1.0,]
# mass2 = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
# mass2 = [1.0, 2.0, 3.0, 4.0, 5.0]
# mass2 = [1e-3,]
# mass2 = [4.0,]
# mass2 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# mass2 = [2.5, 3.5]
Fs = [-0.0]
# Fs = -0.0 .* rs
beta = [100.0]
order = [5,]
Nl = 1
# isDynamic = true
isDynamic = false
isFock = false
ω_c = 0.1

const parafilename = "para_wn_1minus0.csv"
const filename = "data_ver4PP.jld2"
# const savefilename1 = "guu_$(dim)d.dat"
# const savefilename2 = "gud_$(dim)d.dat"
# const savefilename1 = "gsko_$(dim)d.dat"
# const savefilename2 = "gako_$(dim)d.dat"
const savefilename1 = "gsrpa_$(dim)d.dat"
const savefilename2 = "garpa_$(dim)d.dat"
# const savefilename1 = "gsyuk3_$(dim)d.dat"
# const savefilename2 = "gayuk3_$(dim)d.dat"

function Πs(para; ω_c=0.1)
    return 1 / (2π^2) * para.kF * log(0.882 * ω_c * para.EF * para.β)
end

function Un(Γlist, Π, n)
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
        result += Γlist[4] - 2 * Γlist[1] * Π * Γlist[3] + 3 * Γlist[1]^2 * Π^2 * Γlist[2] - Γlist[1]^4 * Π^3
    elseif n == 5
        result += Γlist[1]
        result += Γlist[2] - Γlist[1]^2 * Π
        result += Γlist[3] - 2 * Γlist[1] * Π * Γlist[2] + Γlist[1]^3 * Π^2
        result += Γlist[4] - 2 * Γlist[1] * Π * Γlist[3] + 3 * Γlist[1]^2 * Π^2 * Γlist[2] - Γlist[1]^4 * Π^3
        result += (Γlist[5] - 2 * (Γlist[1] * Π * Γlist[4] + Γlist[2] * Π * Γlist[3])
                   +
                   3 * (Γlist[1]^2 * Π^2 * Γlist[3] + Γlist[2]^2 * Π^2 * Γlist[1])
                   -
                   4 * Γlist[1]^3 * Π^3 * Γlist[2] + Γlist[1]^5 * Π^4)
    end
    return result
end

function Γ2U(Γlist, para; ω_c=0.1)
    Γlist .= -Γlist
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
                    push!(results_s, append!(Any[_rs, _beta, _mass2, _order, il-1], Γ2U([real(data_Fs[o][il, 1]) for o in 1:_order], para; ω_c=ω_c)))
                    push!(results_a, append!(Any[_rs, _beta, _mass2, _order, il-1], Γ2U([real(data_Fa[o][il, 1]) for o in 1:_order], para; ω_c=ω_c)))
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