using UniElectronGas, ElectronLiquid
using JLD2, DelimitedFiles

dim = 3
# dim = 2
# spin = 1
spin = 2
rs = [1.0,]
# rs = [5.0,]
# rs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# rs = [1.0, 2.0, 3.0, 4.0]
# rs = [1.0, 2.0, 3.0, 4.0, 5.0]
# Fs = -[0.223, 0.380, 0.516, 0.639, 0.752]
# Fs = -[0.223,]
# rs = [1.0, 2.0, 3.0]
# mass2 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
mass2 = [3.5,]
# mass2 = [1.0,]
# mass2 = [0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5]
# mass2 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
# mass2 = [9.0, 13.0, 19.0, 27.0]
# mass2 = [6.0, 8.0, 10.0, 12.0, 14.0]
# mass2 = [10.5, 11.0]
# mass2 = [1.0, 2.0, 3.0, 4.0, 5.0]
# mass2 = [2.44355,]
# mass2 = [1.22177,]
# mass2 = [0.814516,]
# mass2 = [0.610887,]
# mass2 = [0.5, 1.0, 1.5, 2.0, 2.2, 2.4, 2.5, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]
# mass2 = [1.6, 1.8, 2.0, 2.2, 2.4]
# mass2 = [1e-3,]
# mass2 = [4.0,]
Fs = -0.0 .* rs
# beta = [20.0, 25.0, 40.0, 80.0]
beta = [25.0,]
# beta = [25.0,]
# order = [3, 4, 5, 6]
order = [5,]
# isDynamic = true
isDynamic = false

const parafilename = "para_wn_1minus0.csv"
# const filename = "./data3d_Z_beta80.jld2"
# const filename = "./data3d_Z_test.jld2"
const filename = "./data_Z_test.jld2"
# const filename = "./data$(dim)d/data$(dim)d_Z.jld2"
# const filename = "./data$(dim)d/data$(dim)d_Z_rs$(rs[1]).jld2"
const savefilename = spin == 2 ? "zfactor_$(dim)d.dat" : "zfactor_$(dim)d_spin$spin.dat"

function process(para, datatuple, isSave)
    if isSave
        UniElectronGas.save_zmu(para, datatuple; parafile=parafilename, verbose=1)
    end
    z = UniElectronGas.getZfactor(para, parafile=parafilename)
    println("Zfactor: ", z)
    return z
end

if abspath(PROGRAM_FILE) == @__FILE__
    isSave = false
    if length(ARGS) >= 1 && (ARGS[1] == "s" || ARGS[1] == "-s" || ARGS[1] == "--save" || ARGS[1] == " save")
        # the second parameter may be set to save the derived parameters
        isSave = true
    end

    f = jldopen(filename, "r")
    results = Any[]
    # for (_rs, _mass2, _F, _beta, _order) in Iterators.product(rs, mass2, Fs, beta, order)
    for (irs, _mass2, _beta, _order) in Iterators.product([i for i in 1:length(rs)], mass2, beta, order)
        _F = Fs[irs]
        _rs = rs[irs]
        para = ParaMC(rs=_rs, beta=_beta, Fs=_F, order=_order, mass2=_mass2, isDynamic=isDynamic, dim=dim, spin=spin)
        kF = para.kF
        for key in keys(f)
            loadpara = ParaMC(key)
            if UEG.paraid(loadpara) == UEG.paraid(para)
                println(UEG.paraid(para))
                zfactor = process(para, f[key], isSave)
                push!(results, Any[_rs, _beta, _mass2, _order, zfactor...])
            end
        end
    end
    if isSave
        open(savefilename, "a+") do io
            writedlm(io, results)
        end
    end
end