using UniElectronGas, ElectronLiquid
using JLD2, DelimitedFiles

include("../input.jl")

function zfactor_renorm(dz, dzinv; isRenorm=false)
    if isRenorm
        sumzinv = accumulate(+, dzinv)
        return @. 1.0 / (1.0 + sumzinv)
    else
        sumz = accumulate(+, dz)
        return @. 1.0 + sumz
    end
end

# function process(para, datatuple, datatuple1, isSave)
# dz, dzinv, dmu = UniElectronGas.get_dzmu(para, datatuple, datatuple1; parafile=parafilename, verbose=1, isSave)
function process(para, datatuple, isSave)
    dz, dzinv, dmu = UniElectronGas.get_dzmu(para, datatuple; parafile=parafilename, verbose=1, isSave=isSave)

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

    f = jldopen(sigma_z_filename, "r")
    results = Any[]
    for (_rs, _mass2, _F, _beta, _order) in Iterators.product(rs, mass2, Fs, beta, order)
        para = ParaMC(rs=_rs, beta=_beta, Fs=_F, order=_order, mass2=_mass2, isDynamic=isDynamic, dim=dim, spin=spin)
        kF = para.kF
        for key in keys(f)
            # println(key)
            # println(f[key])
            loadpara = ParaMC(key)
            if UEG.paraid(loadpara) == UEG.paraid(para)
                println("loading ... ", UEG.paraid(para))
                zfactor = process(para, f[key], isSave)
                # zfactor = process(para, f[key], f1[key], isSave)
                push!(results, Any[_rs, _beta, _mass2, _order, zfactor...])
            end
        end
    end
    # if isSave
    #     open(zfactor_filename, "a+") do io
    #         writedlm(io, results)
    #     end
    # end
end
