using ElectronLiquid
using JLD2

dim = 3
rs = [1.0]
mass2 = [2.5]
# mass2 = [0.5, 1.0, 2.0, 4.0]
# mass2 = [0.001]
Fs = [-0.0,]
beta = [1000.0]
order = [0,]
neval = 1e9
isDynamic = false
isFock = false
spinPolarPara = 0.0 # spin-polarization parameter (n_up - n_down) / (n_up + n_down) ∈ [0,1]

for (_rs, _mass2, _F, _beta, _order) in Iterators.product(rs, mass2, Fs, beta, order)
    para = UEG.ParaMC(rs=_rs, beta=_beta, Fs=_F, order=_order, mass2=_mass2, isDynamic=isDynamic, dim=dim, isFock=isFock)
    println(UEG.short(para))
    filename = "data_freeE_test.jld2"

    freeE, result = FreeEnergy.MC(para; neval=neval, filename=filename, spinPolarPara=spinPolarPara)
end