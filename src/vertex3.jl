function getVer3(para, filename; parafile="para_wn_1minus0.csv", root_dir=@__DIR__)
    # if para.order < 4
    #     para1 = UEG.ParaMC(rs=para.rs, beta=para.beta, Fs=para.Fs, Fa=-0.0, order=4, dim=para.dim,
    #         mass2=para.mass2, isDynamic=para.isDynamic, isFock=para.isFock)
    # end
    # _mu, _zinv = CounterTerm.getSigma(para1, parafile=parafile, root_dir=root_dir)
    _mu, _zinv = CounterTerm.getSigma(para, parafile=parafile, root_dir=root_dir)
    dzinv, dmu, dz = CounterTerm.sigmaCT(para.order, _mu, _zinv)

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
    kamp, kamp2, anglegrid, ver3 = f[key]

    for p in keys(ver3)
        println(p)
        if haskey(vuu, p) == false
            vuu[p] = MeshArray(1, anglegrid; dtype=Complex{Measurement{Float64}})
            vud[p] = MeshArray(1, anglegrid; dtype=Complex{Measurement{Float64}})
        end
        vuu[p][:, :] = ver3[p][1, :, :, 1, 1]
        vud[p][:, :] = ver3[p][2, :, :, 1, 1]
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
