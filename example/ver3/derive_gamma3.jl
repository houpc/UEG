using ElectronLiquid, UniElectronGas
using JLD2, DelimitedFiles
using Measurements

# # rs1lam3.5
# F = [-0.132246 ± 9.4e-5 -0.051096 ± 4.9e-5 -0.017924 ± 3.9e-5 -0.006103 ± 4.8e-5 -0.002216 ± 8.2e-5]
# dzinv = [0.0 ± 5.1e-11, 0.017269 ± 2.6e-5, 0.017072 ± 4.1e-5, 0.01457 ± 5.9e-5]
# # rs5lam1.0
# F = [-0.1895 ± 0.00016 -0.17743 ± 0.00014 -0.13061 ± 0.00014 -0.09931 ± 0.00018 -0.07573 ± 0.00038]
# # dzinv = [0.0 ± 1.3e-9, 0.021954 ± 3.4e-5, 0.022269 ± 6.5e-5, 0.02696 ± 0.00012, 0.02859 ± 0.00026]
# dzinv = [0.0 ± 2.5e-9, 0.022086 ± 4.2e-5, 0.022347 ± 9.8e-5, 0.02707 ± 0.00024, 0.02818 ± 0.00076]

dim = 3
spin = 2
# rs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# rs = [1.0, 2.0, 3.0]
# rs = [1.0,]
rs = [6.0,]
# rs = [3.0, 4.0, 5.0]
# mass2 = [0.5,]
# mass2 = [1e-3,]
# mass2 = [3.5,]
mass2 = [0.75,]
# mass2 = [5.0,]
# mass2 = [1.0, 1.5,]
# mass2 = [1.5, 2.5,]
# mass2 = [6.0, 8.0, 10.0, 12.0, 14.0]
# mass2 = [10.5, 11.0]
# mass2 = [3.0,]
Fs = -0.0 .* rs
# Fs = -[0.223, 0.380, 0.516, 0.639, 0.752]
# Fs = -[0.223,]
# beta = [25.0,]
beta = [40.0]
# beta = [50.0,]
order = [5,]
# inqs = [1, 2, 3]
inqs = [1,]

isDynamic = false
# isDynamic = true
isFock = false

const parafilename = "para_wn_1minus0.csv"
# const filename = "data_ver3_reweight_v4.jld2"
const filename = "data_ver3_reweight_denseangle.jld2"
# const filename = "data_ver3_reweight_full.jld2"
# const filename = "data_ver3_angle_v3.jld2"
# const filename = "data_ver3q0.jld2"
# const savefilename1 = "v3s_$(dim)d.dat"
# const savefilename2 = "v3a_$(dim)d.dat"
# const savefilename1 = "vaa3s_$(dim)d.dat"
# const savefilename2 = "vaa3a_$(dim)d.dat"
# const savefilename1 = "vq03uu_$(dim)d.dat"
# const savefilename2 = "vq03ud_$(dim)d.dat"
const savefilename = "ver3_$(dim)d.dat"
# const savefilename = "ver3_$(dim)d_full.dat"

if abspath(PROGRAM_FILE) == @__FILE__
    isSave = false
    if length(ARGS) >= 1 && (ARGS[1] == "s" || ARGS[1] == "-s" || ARGS[1] == "--save" || ARGS[1] == " save")
        # the second parameter may be set to save the derived parameters
        isSave = true
    end

    f = jldopen(filename, "r")
    for (irs, _mass2, _beta, _order) in Iterators.product([i for i in 1:length(rs)], mass2, beta, order)
        results_sum, results_ea = Any[], Any[]
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
                for inq in inqs
                    for iq in 1:Nq
                        uu = [sum(real(data_uu[oi][1, 1, iq, inq]) * (1)^(oi) for oi in 1:o) for o in 1:_order]
                        ud = [sum(real(data_ud[oi][1, 1, iq, inq]) * (1)^(oi) for oi in 1:o) for o in 1:_order]
                        # uu = [sum(real(data_uu[o][1, 1, iq, inq]) for oi in 1:o) for o in 1:_order]
                        # ud = [sum(real(data_ud[o][1, 1, iq, inq]) for oi in 1:o) for o in 1:_order]
                        # push!(results_s, append!(Any[_rs, _beta, _mass2, _order, iq], [sum(real(data_uu[o][ik, 1]) for oi in 1:o) for o in 1:_order]))
                        # push!(results_a, append!(Any[_rs, _beta, _mass2, _order, iq], [sum(real(data_ud[o][ik, 1]) for oi in 1:o) for o in 1:_order]))
                        push!(results_sum, append!(Any[_rs, _beta, _mass2, _order, iq], uu .+ ud))
                        uu = [sum(real(data_uu[oi][1, 1, iq, inq]) * (1)^(oi) for oi in o:o) for o in 1:_order]
                        ud = [sum(real(data_ud[oi][1, 1, iq, inq]) * (1)^(oi) for oi in o:o) for o in 1:_order]
                        push!(results_ea, append!(Any[_rs, _beta, _mass2, _order, iq], uu .+ ud))
                    end
                end
                # push!(results_s, append!(Any[_rs, _beta, _mass2, _order, 0], [(real(data_Fs[o][1, 1])) for o in 1:_order]))
                # push!(results_a, append!(Any[_rs, _beta, _mass2, _order, 0], [(real(data_Fa[o][1, 1])) for o in 1:_order]))
            end
        end
        println(results_sum)
        if isSave
            savefname = "./ver3_$(dim)d_rs$(_rs)beta$(_beta)lam$(_mass2)o$(_order).dat"
            open(savefname, "w") do io
                writedlm(io, results_sum)
            end
            savefname2 = "./ver3_$(dim)d_rs$(_rs)beta$(_beta)lam$(_mass2)o$(_order)_eachorder.dat"
            open(savefname2, "w") do io
                writedlm(io, results_ea)
            end
            # open(savefilename2, "a+") do io
            #     writedlm(io, results_a)
            # end
        end
    end
end
