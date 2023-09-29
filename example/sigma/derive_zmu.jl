using UniElectronGas, ElectronLiquid
using JLD2, DelimitedFiles

dim = 3
spin = 2

### rs = 1 ###
rs = [1.0]
order = [4,]
mass2 = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]
# order = [5,]
# mass2 = [1.5, 1.75, 2.0, 3.0, 3.5]

### rs = 2 ###
# rs = [2.0]
# order = [4,]
# mass2 = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]
# order = [5,]
# mass2 = [1.625, 1.75, 1.875, 2.0, 2.5]

### rs = 3 ###
# rs = [3.0]
# order = [4,]
# mass2 = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 1.75, 2.0]
# order = [5,]
# mass2 = [1.0, 1.125, 1.25, 1.5, 1.75, 2.0]

### rs = 4 ###
# rs = [4.0]
# order = [4,]
# mass2 = [0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 2.0]
# order = [5,]
# mass2 = [0.875, 1.0, 1.125, 1.25, 1.5]

### rs = 5 ###
# rs = [5.0]
# order = [4,]
# mass2 = [0.375, 0.5, 0.625, 0.75, 1.0, 1.125, 1.25, 1.5]
# order = [5,]
# mass2 = [0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25]

Fs = [-0.0,]
beta = [40.0]
isDynamic = false

# spinPolarPara = 0.0
spinPolarPara = 1.0

ispolarized = spinPolarPara != 0.0

if ispolarized
    const parafilename = "para_wn_1minus0_GV_spin_polarized.csv"
    const filename = "./data$(dim)d/data$(dim)d_Z_GV_spin_polarized.jld2"
    const savefilename = spin == 2 ? "zfactor_$(dim)d_GV_spin_polarized.dat" : "zfactor_$(dim)d_spin$(spin)_GV_spin_polarized.dat"
else
    const parafilename = "para_wn_1minus0.csv"
    const filename = "./data$(dim)d/data$(dim)d_Z.jld2"
    # const filename = "./data$(dim)d/data$(dim)d_Z_o5.jld2"
    const savefilename = spin == 2 ? "zfactor_$(dim)d.dat" : "zfactor_$(dim)d_spin$(spin).dat" 
end

function zfactor_renorm(dz, dzinv; isRenorm=true)
    if isRenorm
        sumzinv = accumulate(+, dzinv)
        return @. 1.0 / (1.0 + sumzinv)
    else
        sumz = accumulate(+, dz)
        return @. 1.0 + sumz
    end
end

function process(para, datatuple, isSave)
    dz, dzinv, dmu = UniElectronGas.get_dzmu(para, datatuple; parafile=parafilename, verbose=1, isSave)

    z = zfactor_renorm(dz, dzinv)
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
    for (_rs, _mass2, _F, _beta, _order) in Iterators.product(rs, mass2, Fs, beta, order)
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
