function getVer3(para, filename; parafile="para_wn_1minus0.csv", root_dir=@__DIR__)
    local _mu, _zinv
    try
        println("try loading order=$(para.order-1)")
        para1 = UEG.ParaMC(rs=para.rs, beta=para.beta, Fs=para.Fs, Fa=-0.0, order=para.order - 1, dim=para.dim,
            mass2=para.mass2, isDynamic=para.isDynamic, isFock=para.isFock)
        _mu, _zinv = CounterTerm.getSigma(para1, parafile=parafile, root_dir=root_dir)
        println("end loading order=$(para.order-1)")
    catch e
        println("error caught")
        # if isa(e, LoadError)
        println("sigma data for order=$(para.order-1) not found, trying higher order")
        try
            para1 = UEG.ParaMC(rs=para.rs, beta=para.beta, Fs=para.Fs, Fa=-0.0, order=para.order, dim=para.dim,
                mass2=para.mass2, isDynamic=para.isDynamic, isFock=para.isFock)
            _mu, _zinv = CounterTerm.getSigma(para1, parafile=parafile, root_dir=root_dir)
        catch e
            println("error caught")
            # if isa(e, LoadError)
            println("sigma data for order=$(para.order) not found, trying higher order")
            para1 = UEG.ParaMC(rs=para.rs, beta=para.beta, Fs=para.Fs, Fa=-0.0, order=para.order + 1, dim=para.dim,
                mass2=para.mass2, isDynamic=para.isDynamic, isFock=para.isFock)
            _mu, _zinv = CounterTerm.getSigma(para1, parafile=parafile, root_dir=root_dir)
        end
        println("end loading order=$(para.order)")
        # end
    end

    # _mu, _zinv = CounterTerm.getSigma(para, parafile=parafile, root_dir=root_dir)
    dzinv, dmu, dz = CounterTerm.sigmaCT(para.order - 1, _mu, _zinv)
    println("zinv", _zinv)
    println("dmu=", dmu)
    println("dz=", dz)
    println("dzinv=", dzinv)

    vuu, vud = ver3_renormalization(para, filename, dz, dmu)
    return (vuu + vud) / 2.0, (vuu - vud) / 2.0
end

function ver3_renormalization(para, filename, dz, dmu)
    # println("read Fs = $Fs from $filename")
    kF = para.kF
    f = jldopen(filename, "r")
    # z1 = zeros(Measurement{Float64}, length(Fs), length(Λgrid))

    vuu = Dict()
    vud = Dict()

    key = UEG.short(para)
    kin, nkin, qout, nqout, ver3 = f[key]

    for p in keys(ver3)
        println(p)
        if haskey(vuu, p) == false
            vuu[p] = MeshArray(kin, nkin, qout, nqout; dtype=Complex{Measurement{Float64}})
            vud[p] = MeshArray(kin, nkin, qout, nqout; dtype=Complex{Measurement{Float64}})
            # vuu[p] = MeshArray(1, anglegrid; dtype=Complex{Measurement{Float64}})
            # vud[p] = MeshArray(1, anglegrid; dtype=Complex{Measurement{Float64}})
        end
        vuu[p][:, :, :, :] = ver3[p][1, :, :, :, :]
        vud[p][:, :, :, :] = ver3[p][2, :, :, :, :]
    end

    vuu_renorm = [vuu[(1, 0, 0)],]
    # sample = collect(values(vuu))[1]
    # z = [zero(sample) for i in 1:order]
    append!(vuu_renorm, CounterTerm.chemicalpotential_renormalization(para.order - 1, vuu, dmu, offset=1))
    vuu_renorm = CounterTerm.z_renormalization(para.order, vuu_renorm, dz, 1) #left leg renormalization

    vud_renorm = [vud[(1, 0, 0)],]
    append!(vud_renorm, CounterTerm.chemicalpotential_renormalization(para.order - 1, vud, dmu, offset=1))
    vud_renorm = CounterTerm.z_renormalization(para.order, vud_renorm, dz, 1) #left leg renormalization

    # vuu = [vuu[(o, 0)] for o in 1:para.order]
    # vud = [vud[(o, 0)] for o in 1:para.order]
    return vuu_renorm, vud_renorm
end
