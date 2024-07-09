using ElectronLiquid, UniElectronGas
using JLD2, DelimitedFiles
using Measurements

dim = 3
spin = 2
# rs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# rs = [1.0, 2.0, 3.0]
rs = [1.0,]
# mass2 = [0.5,]
# mass2 = [1e-3,]
mass2 = [3.5,]
# mass2 = [2.0,]
# mass2 = [6.0, 8.0, 10.0, 12.0, 14.0]
# mass2 = [10.5, 11.0]
# mass2 = [3.0,]
Fs = -0.0 .* rs
# Fs = -[0.223, 0.380, 0.516, 0.639, 0.752]
# Fs = -[0.223,]
beta = [25.0,]
order = [4,]

isDynamic = false
# isDynamic = true
isFock = false

const parafilename = "para_wn_1minus0.csv"
const filename = "data_ver3.jld2"
# const filename = "data_ver3q0.jld2"
# const savefilename1 = "v3s_$(dim)d.dat"
# const savefilename2 = "v3a_$(dim)d.dat"
# const savefilename1 = "vaa3s_$(dim)d.dat"
# const savefilename2 = "vaa3a_$(dim)d.dat"
# const savefilename1 = "vq03uu_$(dim)d.dat"
# const savefilename2 = "vq03ud_$(dim)d.dat"
const savefilename = "ver3_$(dim)d.dat"

if abspath(PROGRAM_FILE) == @__FILE__
    isSave = false
    if length(ARGS) >= 1 && (ARGS[1] == "s" || ARGS[1] == "-s" || ARGS[1] == "--save" || ARGS[1] == " save")
        # the second parameter may be set to save the derived parameters
        isSave = true
    end

    f = jldopen(filename, "r")
    results_s, results_a = Any[], Any[]
    for (irs, _mass2, _beta, _order) in Iterators.product([i for i in 1:length(rs)], mass2, beta, order)
        _F = Fs[irs]
        _rs = rs[irs]
        para = ParaMC(rs=_rs, beta=_beta, Fs=_F, order=_order, mass2=_mass2, isDynamic=isDynamic, isFock=isFock, dim=dim, spin=spin)
        kF = para.kF
        for key in keys(f)
            loadpara = ParaMC(key)
            if UEG.paraid(loadpara) == UEG.paraid(para)
                println(UEG.paraid(para))
                data_Fs, data_Fa = UniElectronGas.getVer3(para, filename)
                data_uu, data_ud = (data_Fs + data_Fa), (data_Fs - data_Fa)
                res_s = Any[_rs, _beta, _mass2, _order]
                # for iθ in 1:Na
                #     push!(results_s, append!(Any[_rs, _beta, _mass2, _order, anglegrid[iθ]], [(real(data_Fs[o][1, iθ])) for o in 1:_order]))
                #     push!(results_a, append!(Any[_rs, _beta, _mass2, _order, anglegrid[iθ]], [(real(data_Fa[o][1, iθ])) for o in 1:_order]))
                # end
                Nq = size(data_uu[1], 3)
                for iq in 1:Nq
                    uu = [sum(real(data_uu[o][1, 1, iq, 1]) for oi in 1:o) for o in 1:_order]
                    ud = [sum(real(data_ud[o][1, 1, iq, 1]) for oi in 1:o) for o in 1:_order]
                    # push!(results_s, append!(Any[_rs, _beta, _mass2, _order, iq], [sum(real(data_uu[o][ik, 1]) for oi in 1:o) for o in 1:_order]))
                    # push!(results_a, append!(Any[_rs, _beta, _mass2, _order, iq], [sum(real(data_ud[o][ik, 1]) for oi in 1:o) for o in 1:_order]))
                    push!(results_s, append!(Any[_rs, _beta, _mass2, _order, iq], uu .+ ud))
                end
                # push!(results_s, append!(Any[_rs, _beta, _mass2, _order, 0], [(real(data_Fs[o][1, 1])) for o in 1:_order]))
                # push!(results_a, append!(Any[_rs, _beta, _mass2, _order, 0], [(real(data_Fa[o][1, 1])) for o in 1:_order]))
            end
        end
    end

    if isSave
        open(savefilename, "a+") do io
            writedlm(io, results_s)
        end
        # open(savefilename2, "a+") do io
        #     writedlm(io, results_a)
        # end
    end
end