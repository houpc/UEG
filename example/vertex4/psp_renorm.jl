using ElectronLiquid, UniElectronGas
using JLD2, DelimitedFiles
using Measurements

const dmstar1 = [0.965154 ± 3.5e-5 - 1.0, -0.012734 ± 2.6e-5, -0.003684 ± 2.6e-5, -1.7e-5 ± 3.3e-5, 0.000808 ± 5.4e-5]
const dmstar2 = [0.971212 ± 3.6e-5 - 1.0, -0.013191 ± 3.4e-5, -0.006643 ± 3.8e-5, -0.00133 ± 5.3e-5, 0.00119 ± 0.0001]
const dmstar3 = [0.978121 ± 3.4e-5 - 1.0, -0.01011 ± 3.3e-5, -0.007079 ± 3.8e-5, -0.002236 ± 5.7e-5, 0.00094 ± 0.00012]
const dmstar4 = [0.977115 ± 3.9e-5 - 1.0, -0.006129 ± 3.9e-5, -0.005259 ± 5.1e-5, 0.000533 ± 9.1e-5, 0.00322 ± 0.00024]

function load_mstar(rs, lam; fname="/media/iintsjds/File/Research/JuliaLibs/UniElectronGas.jl/run/meff_3d_eachorder.dat")
    mdata = readdlm(fname)
    # println(size(mdata))
    # println(mdata)
    for i in 1:size(mdata)[1]
        # println(mdata[i, :])
        mdatalist = mdata[i, :]
        _rs = mdatalist[1]
        _beta = mdatalist[2]
        _lam = mdatalist[3]
        _order = mdatalist[4]
        if _rs == rs && _lam == lam
            dmstar = zeros(Measurement, _order + 1)
            for j in 1:_order+1
                if j == 1
                    dmstar[j] = measurement(mdatalist[1+3j+1], mdatalist[1+3j+3]) - 1.0
                else
                    dmstar[j] = measurement(mdatalist[1+3j+1], mdatalist[1+3j+3])
                end
            end
            return dmstar
        end
    end
    println("mstar not found for rs=$rs, lam=$lam, using bare mass instead!")
    bm = zeros(Measurement, 5)
    bm[1] = 1.0
    return bm
end

function mstar_info(rs, lam)
    # if rs == 1.0
    #     dmstar = dmstar1
    # elseif rs == 2.0
    #     dmstar = dmstar2
    # elseif rs == 2.0
    #     dmstar = dmstar2
    # elseif rs == 2.0
    #     dmstar = dmstar2
    # else
    #     dmstar = zeros(Float64, 5)
    # end
    dmstar = load_mstar(rs, lam)
    mstar = [1.0 + sum(dmstar[1:i]) for i in 1:length(dmstar)]
    return dmstar, mstar
end

function shift_u(u, wc1, wc2)
    # shift u from wc1 to wc2
    return u ./ (1 .+ u .* log(wc1 / wc2))#, uerr ./ (1 .+ u .* log(wc1 / wc2)) .^ 2
end

# Πs returns dimensionless projected pp propagator
function Πs(para; ω_c=0.1)
    # 0.882 comes from the continuous approximation of mat-freq summation
    return log(0.882 * ω_c * para.beta)
    # return log(ω_c * para.beta)
end

# compute U series from Γ
function Un(Γlist4, Π, n, dmstar)
    # divide by 4, 2 from spin and 2 from direct-exchange symmetry
    Γlist = Γlist4 ./ 4
    for i in 1:length(Γlist)
        result = Γlist4[i] / 4
        for j in 1:i-1
            # effective mass renorm, multiplied by m*/m
            result += Γlist4[j] / 4 * dmstar[i-j]
        end
        Γlist[i] = result
    end
    result = 0.0
    # subtract the log(T) component
    if n == 1
        result += Γlist[1]
    elseif n == 2
        result += Γlist[1]
        result += Γlist[2] - Γlist[1]^2 * Π
    elseif n == 3
        result += Γlist[1]
        result += Γlist[2] - Γlist[1]^2 * Π
        result += Γlist[3] - 2 * Γlist[1] * Π * Γlist[2] + Γlist[1]^3 * Π^2
    elseif n == 4
        result += Γlist[1]
        result += Γlist[2] - Γlist[1]^2 * Π
        result += Γlist[3] - 2 * Γlist[1] * Π * Γlist[2] + Γlist[1]^3 * Π^2
        result += Γlist[4] - 2 * Γlist[1] * Π * Γlist[3] - Γlist[2]^2 * Π + 3 * Γlist[1]^2 * Π^2 * Γlist[2] - Γlist[1]^4 * Π^3
    elseif n == 5
        result += Γlist[1]
        result += Γlist[2] - Γlist[1]^2 * Π
        result += Γlist[3] - 2 * Γlist[1] * Π * Γlist[2] + Γlist[1]^3 * Π^2
        result += Γlist[4] - 2 * Γlist[1] * Π * Γlist[3] - Γlist[2]^2 * Π + 3 * Γlist[1]^2 * Π^2 * Γlist[2] - Γlist[1]^4 * Π^3
        result += (Γlist[5] - 2 * (Γlist[1] * Π * Γlist[4] + Γlist[2] * Π * Γlist[3])
                   +
                   3 * (Γlist[1]^2 * Π^2 * Γlist[3] + Γlist[2]^2 * Π^2 * Γlist[1])
                   -
                   4 * Γlist[1]^3 * Π^3 * Γlist[2] + Γlist[1]^5 * Π^4)
    elseif n == 6
        result += Γlist[1]
        result += Γlist[2] - Γlist[1]^2 * Π
        result += Γlist[3] - 2 * Γlist[1] * Π * Γlist[2] + Γlist[1]^3 * Π^2
        result += Γlist[4] - 2 * Γlist[1] * Π * Γlist[3] - Γlist[2]^2 * Π + 3 * Γlist[1]^2 * Π^2 * Γlist[2] - Γlist[1]^4 * Π^3
        result += (Γlist[5] - 2 * (Γlist[1] * Π * Γlist[4] + Γlist[2] * Π * Γlist[3])
                   +
                   3 * (Γlist[1]^2 * Π^2 * Γlist[3] + Γlist[2]^2 * Π^2 * Γlist[1])
                   -
                   4 * Γlist[1]^3 * Π^3 * Γlist[2] + Γlist[1]^5 * Π^4)
        result += (Γlist[6] - 2 * (Γlist[1] * Π * Γlist[5] + Γlist[2] * Π * Γlist[4]) - Γlist[3] * Π * Γlist[3]
                   +
                   (3 * Γlist[1]^2 * Π^2 * Γlist[4] + 6 * Γlist[1] * Π * Γlist[2] * Π * Γlist[3] + Γlist[2] * Π * Γlist[2] * Π * Γlist[2])
                   -
                   4 * Γlist[1]^3 * Π^3 * Γlist[3] - 6 * Γlist[1]^2 * Π^3 * Γlist[2]^2
                   +
                   5 * Γlist[1]^4 * Π^4 * Γlist[2]
                   -
                   Γlist[1]^6 * Π^5)

    end
    return -result
end

function Γ2U(Γlist, dmstar, para; ω_c=0.1)
    println(para.EF, para.beta, para.β)
    # Γlist .= -Γlist
    # Γlist = [(-1)^i * Γlist[i] for i in 1:length(Γlist)]
    Π = Πs(para; ω_c=ω_c)
    return [Un(Γlist, Π, n, dmstar) for n in 1:length(Γlist)]
end

function load_gamma0(rs, mass2, beta, order;
    dim=3, spin=2,
    Fs=-0.0,
    isDynamic=false, isFock=false,
    ω_c=0.1,
    filename="data_ver4PP_parqAD.jld2",
    filelist=[filename,])

    dmstar, mstar = mstar_info(rs, mass2)
    resultlist = [zeros(Measurement, order) for f in filelist]
    # for fname in filelist
    for fi in 1:length(filelist)
        fname = filelist[fi]
        println("fname=$fname")
        f = jldopen(fname, "r")
        # println(keys(f))
        # results_s, results_a = Any[], Any[]
        result = zeros(Measurement, order)
        _F = Fs
        _rs = rs
        para = ParaMC(rs=_rs, beta=beta, Fs=_F, order=order, mass2=mass2, isDynamic=isDynamic, isFock=isFock, dim=dim, spin=spin)
        # println("NF=$(para.NF)")
        # println("$para")
        kF = para.kF
        println(keys(f))
        for key in keys(f)
            loadpara = ParaMC(key)
            if UEG.paraid(loadpara) == UEG.paraid(para)
                println(UEG.paraid(para))
                data_Fs, data_Fa = UniElectronGas.getVer4PHl(para, fname)
                # data_uu, data_ud = (data_Fs + data_Fa), (data_Fs - data_Fa)
                result .= Γ2U([real(data_Fs[o][1, 1] - 3 * data_Fa[o][1, 1]) for o in 1:order], dmstar, para; ω_c=ω_c)
                # return Γ2U([real(data_Fs[o][1, 1] + 3 * data_Fa[o][1, 1]) for o in 1:_order], para; ω_c=ω_c)
            end
        end
        for i in 2:length(result)
            result[i] = shift_u(result[i], ω_c, ω_c * mstar[i-1])
        end
        println("result=$result")
        # push!(resultlist, result)
        resultlist[fi] .= result
    end
    validresults = [r for r in resultlist if sum(r) != 0]
    weights = [1 / (Measurements.uncertainty(r[end]))^2 for r in validresults]
    if length(validresults) > 1
        println("multiple results!")
        println(validresults)
        println(weights)
    end
    finalresult = zeros(Measurement, order)
    for i in 1:length(validresults)
        finalresult .+= weights[i] .* validresults[i] ./ sum(weights)
    end
    return finalresult
end

if abspath(PROGRAM_FILE) == @__FILE__
    # m1 = load_mstar(1.0, 3.5)
    # println(m1)
    uc1 = load_gamma0(1.0, 3.5, 25, 6; ω_c=0.1, filename="/media/iintsjds/File/Research/JuliaLibs/UniElectronGas.jl/run/kunshan/data_ver4PP_parqAD_rs1.jld2")
    println(uc1)
end