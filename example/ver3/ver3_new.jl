using ElectronLiquid, FeynmanDiagram, JLD2
import FeynmanDiagram.FrontEnds: NoHartree, Proper
dim = 3
# rs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# rs = [1.0, 2.0, 3.0, 4.0, 5.0]
# rs = [5.0,]
rs = [1.0,]
# Fs = -[0.223, 0.380, 0.516, 0.639, 0.752]
# Fs = -[0.223,]
# mass2 = [1.0,]
mass2 = [3.5,]
# mass2 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# mass2 = [4.0, 5.0, 6.0]
# mass2 = [2.44355,]
# Fs = [-0.0,]
Fs = -0.0 .* rs
beta = [25.0]
order = [4,]
# neval = 1e9
neval = 1e7
# isDynamic = true
isDynamic = false
isFock = false

Nth = 4
# theta = [(i) / (Nth * π) for i in 0:Nth] # N+1 points
theta = [0.0,]
nkin = [0,]
# nqout = [0, 1]
nqout = [0,]

for (irs, _mass2, _beta, _order) in Iterators.product([i for i in 1:length(rs)], mass2, beta, order)
    _F = Fs[irs]
    _rs = rs[irs]
    para = ParaMC(rs=_rs, beta=_beta, Fs=_F, order=_order, mass2=_mass2, isDynamic=isDynamic, dim=dim, isFock=isFock)
    println(UEG.short(para))
    filename = "data_ver3.jld2"

    qout = [[para.kF * (1 - cos(θ)), -para.kF * sin(θ), 0.0] for θ in theta]

    partition = UEG.partition(_order)
    println(partition)
    neighbor = UEG.neighbor(partition)
    reweight_goal = Float64[]
    for (order, sOrder, vOrder) in partition
        order == 1 && sOrder > 0 && continue
        push!(reweight_goal, 2.0^(2order + sOrder + vOrder - 2))
    end
    push!(reweight_goal, 4.0)
    println(reweight_goal)

    transferLoop = [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    ver3, result = Ver3.MC_KW_angle(para;
        qout=qout, nkin=nkin, nqout=nqout,
        neval=neval, filename=filename, partition=partition,
        filter=[NoHartree, Proper], transferLoop=transferLoop)

end
