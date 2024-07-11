using ElectronLiquid
using CompositeGrids

dim = 3
rs = [5.0,]
# rs = [1.0, 2.0, 3.0]
# rs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# rs = [1.0, 2.0, 3.0, 4.0, 5.0]
# Fs = -[0.223, 0.380, 0.516, 0.639, 0.752]
# Fs = -[0.223,]
# mass2 = [1.0, 2.0, 3.0]
# mass2 = [1e-3,]
mass2 = [0.5,]
# mass2 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# mass2 = [4.0, 5.0, 6.0]
# mass2 = [1.0, 2.0, 3.0, 4.0, 5.0]
# mass2 = [6.0, 8.0, 10.0, 12.0, 14.0]
# mass2 = [10.5, 11.0]
Fs = [-0.0,]
beta = [25.0,]
order = [4,]
# neval = 1e6
neval = 1e7
# isDynamic = true
isDynamic = false
isFock = false
diagGenerate = :GV
# diagGenerate = :Parquet

# mission = :Z
# mission = :K
mission = ARGS[1]
println("mission: ", mission)

# for (_rs, _mass2, _F, _beta, _order) in Iterators.product(rs, mass2, Fs, beta, order)
for (irs, _mass2, _beta, _order) in Iterators.product([i for i in 1:length(rs)], mass2, beta, order)
    _F = Fs[irs]
    _rs = rs[irs]
    para = ParaMC(rs=_rs, beta=_beta, Fs=_F, order=_order, mass2=_mass2, isDynamic=isDynamic, dim=dim, isFock=isFock)
    println(UEG.short(para))
    # para = ParaMC(rs=_rs, beta=_beta, Fs=_F, order=_order, mass2=_mass2, isDynamic=isDynamic, dim=dim, isFock=isFock, spin=1)
    kF = para.kF

    if mission == "Z"
        ######### calcualte Z factor ######################
        kgrid = [kF,]
        ngrid = [-1, 0]
    elseif mission == "K"
        ######### calculate K dependence #####################
        Nk, korder = 4, 4
        minK = 0.2kF
        # kgrid = CompositeGrid.LogDensedGrid(:uniform, [0.0, 2.2kF], [kF,], Nk, minK, korder).grid
        # kgrid = kF .+ [-0.1, -0.05, -0.03, -0.01, -0.005, -0.001, 0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1] * kF
        kgrid = kF .+ [-0.1, -0.05, 0, 0.05, 0.1] * kF
        ngrid = [0,]
    else
        error("unknown mission")
    end

    filename = "data_$(mission)_test.jld2"
    # filename = "data$(dim)_$(mission).jld2"

    sigma, result = Sigma.MC_dk(para; kgrid=kgrid, ngrid=ngrid,
        neval=neval, filename=filename)
    # diagtype=diagGenerate)

end
