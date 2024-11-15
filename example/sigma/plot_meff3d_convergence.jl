using UniElectronGas
using PyCall
using PyPlot
using DelimitedFiles
using CurveFit
using Measurements

@pyimport scienceplots  # `import scienceplots` is required as of version 2.0.0
@pyimport scipy.interpolate as interp
@pyimport matplotlib.gridspec as gridspec

include("../input.jl")

cdict = Dict([
    "blue" => "#0077BB",
    "cyan" => "#33BBEE",
    "teal" => "#009988",
    "orange" => "#EE7733",
    "red" => "#CC3311",
    "magenta" => "#EE3377",
    "grey" => "#BBBBBB",
])
style = PyPlot.matplotlib."style"
style.use(["science", "std-colors"])
const color = [
    "black",
    cdict["orange"],
    cdict["blue"],
    cdict["cyan"],
    cdict["magenta"],
    cdict["teal"],
    cdict["red"],
]
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 16
rcParams["mathtext.fontset"] = "cm"
rcParams["font.family"] = "Times New Roman"

modestr = ""
if spin != 2
    modestr *= "_spin$(spin)"
end
if ispolarized
    modestr *= "_polarized"
end
if isLayered2D
    modestr *= "_layered"
end

# Fixed lambda optima, rₛ ↦ λ*(rₛ), d = 3
const fixed_lambda_optima_3d = Dict(
    # 0.5 => 3.5,
    1.0 => 5.0,
    2.0 => 3.0,
    3.0 => 1.875,
    4.0 => 1.625,
    5.0 => 1.25,
    # 6.0 => 1.0,
)

# For combined plot, rₛ ↦ λ*(rₛ), d = 3
const fixed_lambda_comparisons_3d = Dict(
    1.0 => 6.5,
    2.0 => 3.5,
    3.0 => 2.25,
    # 4.0 => 1.75,
    4.0 => 2.0,
    5.0 => 1.75,
    6.0 => 1.0,
)
# Two lambda points to plot for convergence tests
# Where possible, we take: (1) the chosen optimum λ*(rₛ) and (2) the largest calculated λ
# NOTE: At rs = 5, λ* itself is the largest calculated λ, so we compare with smaller λ = 0.875
const lambdas_meff_convergence_plot_3d = Dict(
    # 0.5 => [3.5, 5.0],
    1.0 => [1.75, 3.5, 5.0, 6.5, 8.0],
    2.0 => [2.0, 2.5, 3.0, 3.5],
    3.0 => [1.5, 1.75, 1.875, 2.0, 2.25],
    4.0 => [1.375, 1.5, 1.625, 1.75, 2.0],
    # 4.0 => [1.25, 1.375, 1.5, 1.625, 1.75],
    5.0 => [1.125, 1.25, 1.375, 1.5, 1.75],
    # 6.0 => [0.75, 1.0, 1.25],
)

const optimal_lambda_indices_3d = Dict(
    1.0 => 2,
    2.0 => 3,
    3.0 => 3,
    4.0 => 3,
    5.0 => 3,
    6.0 => 1,
)

const lambdas_meff_lower_bounds_3d = Dict(
    # 0.5 => ...,
    # 1.0 => ...,
    # 2.0 => ...,
    3.0 => 1.5,
    # 4.0 => ...,
    # 5.0 => ...,
    # 6.0 => ...,
)

const lambdas_meff_upper_bounds_3d = Dict(
    # 0.5 => ...,
    # 1.0 => ...,
    # 2.0 => ...,
    3.0 => 2.25,
    # 4.0 => ...,
    # 5.0 => ...,
    # 6.0 => ...,
)

function get_local_minima(a)
    indices = []
    minima = []
    sgn_prev = sign(a[2] - a[1])
    for i in eachindex(a)[3:end]
        sgn_curr = sign(a[i] - a[i-1])
        if (sgn_curr != sgn_prev) && (sgn_prev < 0)
            push!(indices, i - 1)
            push!(minima, a[i-1])
        end
        sgn_prev = sgn_curr
    end
    return indices, minima
end

function get_local_extrema(a)
    indices = []
    extrema = []
    sgn_prev = sign(a[2] - a[1])
    for i in eachindex(a)[3:end]
        sgn_curr = sign(a[i] - a[i-1])
        if sgn_curr != sgn_prev
            push!(indices, i - 1)
            push!(extrema, a[i-1])
        end
        sgn_prev = sgn_curr
    end
    return indices, extrema
end

function spline(x, y, e; xmin=0.0, xmax=x[end])
    # generate knots with spline without constraints
    w = 1.0 ./ e
    spl = interp.UnivariateSpline(x, y; w=w, k=3)
    __x = collect(LinRange(xmin, xmax, 1000))
    yfit = spl(__x)
    return __x, yfit
end

function spline_with_bc(x, y, e; xmin=0.0, xmax=x[end])
    _x, _y = deepcopy(x), deepcopy(y)
    _w = 1.0 ./ e

    #enforce left boundary condition: zero derivative at 1/N → 0
    rescale = 10000
    pushfirst!(_x, 0.0)
    pushfirst!(_y, y[1] / _w[1])
    pushfirst!(_w, _w[1] * rescale)

    # generate knots with spline without constraints
    spl = interp.UnivariateSpline(_x, _y; w=_w, k=3)
    __x = collect(LinRange(xmin, xmax, 1000))
    yfit = spl(__x)
    return __x, yfit
end

# NOTE: assumes the following row format: 
#       | rs | beta | mass2 | order | mean_1 ± error_1 | ... | mean_N ± error_N |
function load_from_dlm(filename, mass2; rs=rs[1], beta=beta[1], verbose=false)
    data = readdlm(filename)
    num_data = size(data)[1]
    idx = 0
    currMaxOrder = 0
    for i in 1:num_data
        if data[i, 1] == rs && data[i, 3] == mass2
            idx = i
            if data[i, 4] > currMaxOrder > 0
                println("(lambda = $(mass2)) Promoting from order $(currMaxOrder) to $(data[i, 4])")
                currMaxOrder = data[i, 4]
            end
        end
    end
    @assert idx != 0 "Data for rs = $(rs), mass2 = $(mass2) not found in file $(filename)"
    _order = data[idx, 4]
    mean, error = [], [], []
    for o in 1:_order
        push!(mean, data[idx, 3o+2])
        push!(error, data[idx, 3o+4])
    end
    # # Add padding values if this mass2 run has N < maxOrder
    npad = max(0, order[1] - _order)
    append!(mean, repeat([missing], npad))
    append!(error, repeat([missing], npad))

    mean_total = mean
    error_total = error
    mass2_total = data[idx, 3]
    if verbose
        println(mean_total)
        println(error_total)
        println(mass2_total)
    end
    return mean_total, error_total, mass2_total
end
function load_from_dlm(filename; mass2=mass2, rs=rs[1], beta=beta[1], sortby="order", verbose=false)
    @assert sortby in ["order", "mass2"]

    data = readdlm(filename)
    num_data = size(data)[1]
    mean_total, error_total, mass2_total = [], [], []
    for _mass2 in mass2
        idx = 0
        currMaxOrder = 0
        for i in 1:num_data
            if data[i, 1] == rs && data[i, 3] == _mass2
                idx = i
                if data[i, 4] > currMaxOrder > 0
                    println("(lambda = $(_mass2)) Promoting from order $(currMaxOrder) to $(data[i, 4])")
                    currMaxOrder = data[i, 4]
                end
            end
        end
        idx == 0 && continue
        _order = data[idx, 4]
        mean, error = [], [], []
        for o in 1:_order
            push!(mean, data[idx, 3o+2])
            push!(error, data[idx, 3o+4])
        end
        # # Add padding values if this mass2 run has N < maxOrder
        npad = max(0, order[1] - _order)
        append!(mean, repeat([missing], npad))
        append!(error, repeat([missing], npad))
        # Add results for this mass2 to lists
        push!(mean_total, mean)
        push!(error_total, error)
        push!(mass2_total, data[idx, 3])
    end
    if verbose
        println(mean_total)
        println(error_total)
        println(mass2_total)
    end

    if sortby == "order"
        mt = hcat(mean_total...)
        et = hcat(error_total...)
        mean_order, error_order = [], []
        for o in 1:order[1]
            push!(mean_order, mt[o, :])
            push!(error_order, et[o, :])
        end
        return mean_order, error_order, mass2_total
    else # sortby == "mass2"
        return mean_total, error_total, mass2_total
    end
end

function plot_cancellation_convergence(meff_estimates; beta=beta[1])
    plot_rs = [1.0, 5.0]
    num_rs = length(plot_rs)
    @assert num_rs == 2 "plot_rs must be a 2-element array"
    plot_lambda = [fixed_lambda_optima_3d[rs] for rs in plot_rs]

    # Helper function to plot a single subplot
    function plot_at_rs(i, rs, lambda; ax0=nothing)
        if isnothing(ax0)
            ax = plt.subplot(num_rs, 1, i)
        else
            ax = plt.subplot(num_rs, 1, i, sharex=ax0)
        end
        # Z, D, and m/m* ≈ Z * D
        filenames = [
            dispersion_ratio_dk_filename,
            zfactor_filename,
            inverse_meff_dk_filename,
        ]
        colors = [
            cdict["red"],
            cdict["blue"],
            "black",
        ]
        label_locs = [(4.4, 1.075), (1.4, 0.975), (1.975, 1.025)]
        labels = [
            "\$D\$",
            "\$Z\$",
            "\$m / m^* = Z \\cdot D\$",
        ]
        for (j, (filename, color)) in enumerate(zip(filenames, colors))
            means, errors, lambda = load_from_dlm(filename, lambda; rs=rs)
            valid_means = collect(skipmissing(means))
            valid_errors = collect(skipmissing(errors))
            x = collect(eachindex(valid_means))
            yval = valid_means
            yerr = valid_errors
            println(lambda)
            println(x)
            println(yval)
            println(yerr)
            ax.errorbar(
                x,
                yval,
                yerr=yerr,
                color=color,
                capsize=4,
                fmt="o--",
                zorder=10 * j,
            )
            if i == 1
                println(labels[j], " ", label_locs[j])
                ax.annotate(labels[j], xy=label_locs[j], xycoords="data")
            end
            if j == 3
                # Error in final m/m* estimate is obtained from combined approaches
                # TODO: actually compute meff_inv_estimates, although the result will hardly change
                error_estimate = (1 / meff_estimates[rs]).err
                ax.axhspan(
                    yval[end] - error_estimate,
                    yval[end] + error_estimate;
                    color=cdict["grey"],
                )
                # # Rough estimate of total error using the last 3 orders
                # d1 = abs(yval[end] - yval[end-1])
                # d2 = abs(yval[end] - yval[end-2])
                # error_estimate = yerr[end] + max(d1, d2)
                # ax.axhspan(
                #     yval[end] - error_estimate,
                #     yval[end] + error_estimate;
                #     color=cdict["grey"],
                # )
                ax.axhline(yval[end]; linestyle="--", color="dimgrey")
                meff_estimate = measurement(yval[end], error_estimate)
                println("rs = $rs, λ = $lambda:\tm/m* ≈ $meff_estimate")
            end
        end
        # ax.set_ylim(0.85, 1.1)
        ax.set_xticks(collect(1:order[1]))
        ax.legend(; title="\$r_s = $(Int(rs))\$", loc="upper left")
        return ax
    end

    figure(figsize=(4, 4 * num_rs))

    # first subplot
    ax0 = plot_at_rs(1, plot_rs[1], plot_lambda[1])
    plt.setp(ax0.get_xticklabels(), visible=false)  # use shared x-axis

    # second subplot
    ax1 = plot_at_rs(2, plot_rs[2], plot_lambda[2]; ax0=ax0)
    ax1.set_xlabel("Perturbation order \$N\$")

    # remove last tick label for the second subplot
    yticks = ax1.yaxis.get_major_ticks()
    yticks[end].label1.set_visible(false)

    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=0.0)

    plt.savefig("meff$(dim)d_cancellations_vs_N.pdf")
end

function plot_regular_vs_inverse_convergence_comparisons(meff_estimates; beta=beta[1])
    rs = 1.0
    lambda = fixed_lambda_optima_3d[rs]
    fig, ax = plt.subplots(; figsize=(4, 4))

    # (regular) Z, D, and m/m* ≈ Z * D
    # (inverse) Z ≈ 1 / Z⁻¹, D ≈ 1 / D⁻¹, and m/m* ≈ 1 / (Z⁻¹ * D⁻¹)
    filenames = [
        inverse_meff_dk_filename,
        meff_dk_filename,
        dispersion_ratio_dk_filename,
        inverse_dispersion_ratio_dk_filename,
        zfactor_filename,
        zinv_filename,
    ]
    zorders = [
        4, 3, 2, 1, 6, 5,
    ]
    colors = [
        "black",
        cdict["grey"],
        cdict["red"],
        cdict["orange"],
        cdict["blue"],
        cdict["teal"],
    ]
    labels = [
        "\${m}/{m^*} = Z \\cdot D\$",
        "\${m}/{m^*} = {1}/({Z^{-1} \\cdot D^{-1}})\$",
        "\$D\$",
        "\$1 / D^{-1}\$",
        "\$Z\$",
        "\$1 / Z^{-1}\$",
    ]
    for (j, (filename, color, zorder)) in enumerate(zip(filenames, colors, zorders))
        invert = iseven(j)
        means, errors, lambda = load_from_dlm(filename, lambda; rs=rs)
        valid_means = collect(skipmissing(means))
        valid_errors = collect(skipmissing(errors))
        x = collect(eachindex(valid_means))
        if invert
            inv_ymeas = measurement.(valid_means, valid_errors)
            ymeas = 1 ./ inv_ymeas
            yval = Measurements.value.(ymeas)
            yerr = Measurements.uncertainty.(ymeas)
        else
            yval = valid_means
            yerr = valid_errors
        end
        println(lambda)
        println(x)
        println(yval)
        println(yerr)
        ax.errorbar(
            x,
            yval;
            yerr=yerr,
            color=color,
            capsize=4,
            fmt="o--",
            zorder=zorder,
            label=labels[j],
        )
        if j in [1, 2]
            # Error in final m/m* estimate is obtained from combined approaches
            # TODO: actually compute meff_inv_estimates when invert == false, although the result will hardly change
            error_estimate = (1 / meff_estimates[rs]).err
            # ax.axhspan(
            #     yval[end] - error_estimate,
            #     yval[end] + error_estimate;
            #     color=cdict["grey"],
            #     zorder=3,
            # )
            # ax.axhline(yval[end]; linestyle="--", color="dimgrey", zorder=4)
            meff_estimate = measurement(yval[end], error_estimate)
            typestr = invert ? "(inverse)" : "(regular)"
            println("$typestr rs = $rs, λ = $lambda:\tm/m* ≈ $meff_estimate, $yval")
        end
    end
    xy_loc = (1.2, 0.96)
    ax.set_ylim(0.94, 1.16)
    # xy_loc = (1.2, 0.915)
    # ax.set_ylim(0.89, 1.16)
    ax.annotate("\$r_s = $(Int(round(rs)))\$", xy=xy_loc, xycoords="data", fontsize=14)
    ax.set_xticks(collect(1:order[1]))
    ax.legend(; loc="upper left", fontsize=10)
    # ax.legend(; title="\$r_s = $(Int(rs))\$", loc="upper left", fontsize=10, ncol=2, columnspacing=0.5)
    ax.set_xlabel("Perturbation order \$N\$")
    plt.tight_layout()
    fig.savefig("meff$(dim)d_regular_and_inverse_comparisons_vs_N_rs=$(Int(round(rs))).pdf")
end

function plot_meff_order_convergence_minimal_sensitivity(;
    beta=beta[1],
    maxOrder=order[1],
    plot_rs=range(1.0, 6.0),
)
    pick_extrema = Dict(
        1.0 => [missing, 1, 2, 1, 4, 1],
        2.0 => [missing, 1, 2, 1, 2, 1],
    )
    pick_indices = Dict(
        1.0 => [missing, missing, missing, missing, missing, missing],
        2.0 => [missing, missing, missing, missing, missing, missing],
    )
    num_rs = length(plot_rs)
    figure(figsize=(4 * num_rs, 4))
    for (i, rs) in enumerate(plot_rs)
        subplot(1, num_rs, i)

        # The full set of mass2 values for this rs
        all_mass2 = sort(union(rs_to_lambdas[dim][4][rs], rs_to_lambdas[dim][5][rs], rs_to_lambdas[dim][6][rs]))
        meff_means, meff_errors, _mass2 = load_from_dlm(meff_dk_filename; mass2=all_mass2, rs=rs, sortby="order")

        # println(_mass2)
        # println(meff_means)

        lambdas_minimal_sensitivity = [Inf]
        means_minimal_sensitivity = [NaN]
        errs_minimal_sensitivity = [NaN]
        for o in 2:maxOrder
            valid_meff = skipmissing(meff_means[o])
            valid_errors = skipmissing(meff_errors[o])

            idx_valid_mass2 = collect(eachindex(valid_meff))
            x = _mass2[idx_valid_mass2]
            yval = collect(valid_meff)
            yerr = collect(valid_errors)

            println("\nN = $o")
            println(x)
            println(yval)

            # Find local extrema in meff
            indices, extrema = get_local_extrema(yval)
            println(indices)
            println(extrema)

            # Use a specific index at 6th order if the extrema are not known
            if ismissing(pick_extrema[rs][o])
                idx = pick_indices[rs][o]
                lambda = x[idx]
                push!(lambdas_minimal_sensitivity, lambda)
                push!(means_minimal_sensitivity, yval[idx])
                push!(errs_minimal_sensitivity, yerr[idx])
            else
                idx = indices[pick_extrema[rs][o]]
                lambda = x[idx]
                push!(lambdas_minimal_sensitivity, lambda)
                push!(means_minimal_sensitivity, yval[idx])
                push!(errs_minimal_sensitivity, yerr[idx])
            end
            # if o == maxOrder
            #     println("\n(N = $o)\nλ = $x\nm*/m = $(measurement.(yval, yerr))\n")
            # end
        end

        x = collect(eachindex(means_minimal_sensitivity))
        yval = means_minimal_sensitivity
        yerr = errs_minimal_sensitivity
        errorbar(
            x,
            yval,
            yerr=yerr,
            color=color[i+1],
            capsize=4,
            fmt="o--",
            zorder=10 * i,
        )
        for (_x, _y) in zip(x, yval)
            lambda = lambdas_minimal_sensitivity[_x]
            # annotate("\$\\lambda = $lambda\$",
            annotate(lambda,
                xy=(_x, _y), xycoords="data",
                xytext=(-12, 12),
                textcoords="offset points",
                horizontalalignment="center", verticalalignment="center",
            )
        end

        # Rough estimate of total error using the last 3 orders
        d1 = abs(yval[end] - yval[end-1])
        d2 = abs(yval[end] - yval[end-2])
        error_estimate = yerr[end] + max(d1, d2)
        meff_estimate = measurement(yval[end], error_estimate)
        axhspan(
            yval[end] - error_estimate,
            yval[end] + error_estimate;
            color=color[i+1],
            alpha=0.2,
        )
        axhline(yval[end]; linestyle="-", color=color[i+1], alpha=0.6)
        println("rs = $rs, λ = $(lambdas_minimal_sensitivity[end]):\tm*/m ≈ $meff_estimate")
        xticks(collect(1:order[1]))
        xlabel("Perturbation order \$N\$")
        ylabel("\$m^* / m\$")
        # legend(; title="\$r_s = $(rs)\$", loc="best")
        legend(; title="\$r_s = $(rs)\$", loc="best", fontsize=14)
    end
    # ylim(0.978, 1.002)
    tight_layout()
    savefig("meff$(dim)d_beta$(beta)$(modestr)_minimal_sensitivity_vs_N.pdf")
end

function plot_meff_order_convergence(;
    beta=beta[1],
    plot_rs=range(1.0, 6.0),
    plot_lambdas=[lambdas_meff_convergence_plot_3d[rs] for rs in plot_rs],
)
    num_rs = length(plot_rs)
    figure(figsize=(4 * num_rs, 4))
    for (i, (rs, lambdas)) in enumerate(zip(plot_rs, plot_lambdas))
        subplot(1, num_rs, i)
        # # Use lambda upper/lower bounds to define the error bar
        # have_lambda_bounds = haskey(lambdas_meff_lower_bounds_3d, rs) && haskey(lambdas_meff_upper_bounds_3d, rs)
        # if have_lambda_bounds
        #     lambda_lower = lambdas_meff_lower_bounds_3d[rs]
        #     lambda_upper = lambdas_meff_upper_bounds_3d[rs]
        #     meff_lower = minimum(load_from_dlm(meff_dk_filename, lambda_lower; rs=rs)[1])
        #     meff_upper = minimum(load_from_dlm(meff_dk_filename, lambda_upper; rs=rs)[1])
        #     max_error = maximum(abs.(meff_upper - meff_lower))
        #     axhspan(
        #         meff_lower - max_error,
        #         meff_upper + max_error;
        #         color=cdict["grey"],
        #         alpha=0.2,
        #     )
        # end
        for (j, lambda) in enumerate(lambdas)
            # meff_file = lambda == 1.75 ? meff_dk_filename : meff_dk_filename
            # means, errors, lambda = load_from_dlm(meff_dk_filename, lambda; rs=rs)
            # means, errors, lambda = load_from_dlm(meff_dk_filename, lambda; rs=rs)
            means, errors, lambda = load_from_dlm(meff_dk_filename, lambda; rs=rs)
            valid_means = collect(skipmissing(means))
            valid_errors = collect(skipmissing(errors))
            x = collect(eachindex(valid_means))
            yval = valid_means
            yerr = valid_errors
            # starstr = j == 1 ? "^*" : ""
            starstr = lambda == fixed_lambda_optima_3d[rs] ? "^*" : ""
            errorbar(
                x,
                yval,
                yerr=yerr,
                color=color[j+1],
                capsize=4,
                fmt="o--",
                # label="\$\\lambda = $lambda\$",
                label="\$\\lambda$starstr = $lambda\$",
                zorder=10 * j,
            )
            # # Rough estimate of total error using the last 3 orders
            # d1 = abs(yval[end] - yval[end-1])
            # d2 = abs(yval[end] - yval[end-2])
            # error_estimate = yerr[end] + max(d1, d2)
            # meff_estimate = measurement(yval[end], error_estimate)
            # if j == 1
            #     axhspan(
            #         yval[end] - error_estimate,
            #         yval[end] + error_estimate;
            #         color=color[j+1],
            #         alpha=0.2,
            #     )
            #     axhline(yval[end]; linestyle="-", color=color[j+1], alpha=0.6)
            # end
            if lambda == lambdas_meff_convergence_plot_3d[rs]
                println("rs = $rs, λ = $lambda:\tm*/m ≈ $meff_estimate")
            end
        end
        xticks(collect(1:order[1]))
        xlabel("Perturbation order \$N\$")
        if i == 1
            ylabel("\$m^* / m\$")
        end
        # legend(; title="\$r_s = $(rs)\$", loc="best")
        legend(; title="\$r_s = $(rs)\$", loc="best", fontsize=14)
    end
    # ylim(0.963, 1.003)
    # ylim(0.978, 1.002)
    tight_layout()
    savefig("meff$(dim)d_beta$(beta)$(modestr)_vs_N.pdf")
end

function plot_combined_order_convergence(;
    beta=beta[1],
    maxOrder=order[1],
    plot_rs=range(1.0, 6.0),
)
    meff_estimates = Dict()
    ylimits = Dict(
        1.0 => 0.938,
        2.0 => 0.938,
        3.0 => 0.944,
        4.0 => 0.956,
        5.0 => 0.966,
        6.0 => 0.972,
    )
    pick_extrema = Dict(
        1.0 => [missing, 1, 2, 1, 4, 1],
        2.0 => [missing, 1, 2, 1, 2, 1],
        3.0 => [missing, missing, 1, 1, 1, 1],
        4.0 => [missing, 1, 1, 1, 1, 1],
        5.0 => [missing, 1, 1, 1, 1, 1],
        6.0 => [missing, 1, 1, 1, 1, 1],
    )
    pick_indices = Dict(
        1.0 => [missing, missing, missing, missing, missing, missing],
        2.0 => [missing, missing, missing, missing, missing, missing],
        3.0 => [missing, 1, missing, missing, missing, missing],
        4.0 => [missing, missing, missing, missing, missing, missing],
        5.0 => [missing, missing, missing, missing, missing, missing],
        6.0 => [missing, missing, missing, missing, missing, missing],
    )
    num_rs = length(plot_rs)
    figure(figsize=(4 * num_rs, 4))
    for (i, rs) in enumerate(plot_rs)
        subplot(1, num_rs, i)
        # Plot large fixed lambda curve
        fixed_lambda = fixed_lambda_comparisons_3d[rs]
        means, errors, fixed_lambda = load_from_dlm(meff_dk_filename, fixed_lambda; rs=rs)
        valid_means = collect(skipmissing(means))
        valid_errors = collect(skipmissing(errors))
        x = collect(eachindex(valid_means))
        yval_hi = valid_means
        yerr_hi = valid_errors
        errorbar(
            x,
            yval_hi,
            yerr=yerr_hi,
            color=cdict["red"],
            capsize=4,
            fmt="o--",
            label="\$\\lambda = $fixed_lambda\$",
        )

        # Plot fixed lambda optimum curve
        optimal_lambda = fixed_lambda_optima_3d[rs]
        means, errors, optimal_lambda = load_from_dlm(meff_dk_filename, optimal_lambda; rs=rs)
        valid_means = collect(skipmissing(means))
        valid_errors = collect(skipmissing(errors))
        errorbar(
            eachindex(valid_means),
            valid_means,
            yerr=valid_errors,
            color=cdict["teal"],
            capsize=4,
            fmt="o--",
            label="\$\\lambda^* = $optimal_lambda\$",
        )

        # The full set of mass2 values for this rs
        all_mass2 = sort(union(rs_to_lambdas[dim][4][rs], rs_to_lambdas[dim][5][rs], rs_to_lambdas[dim][6][rs]))
        meff_means, meff_errors, _mass2 = load_from_dlm(meff_dk_filename; mass2=all_mass2, rs=rs, sortby="order")
        lambdas_minimal_sensitivity = [Inf]
        means_minimal_sensitivity = [NaN]
        errs_minimal_sensitivity = [NaN]
        for o in 2:maxOrder
            valid_meff = skipmissing(meff_means[o])
            valid_errors = skipmissing(meff_errors[o])
            idx_valid_mass2 = collect(eachindex(valid_meff))
            x = _mass2[idx_valid_mass2]
            yval = collect(valid_meff)
            yerr = collect(valid_errors)
            println("\nN = $o")
            println(x)
            println(yval)
            # Find local extrema in meff
            indices, extrema = get_local_extrema(yval)
            println(indices)
            println(extrema)
            # Use a specific index at 6th order if the extrema are not known
            if ismissing(pick_extrema[rs][o])
                idx = pick_indices[rs][o]
                lambda = x[idx]
                push!(lambdas_minimal_sensitivity, lambda)
                push!(means_minimal_sensitivity, yval[idx])
                push!(errs_minimal_sensitivity, yerr[idx])
            else
                idx = indices[pick_extrema[rs][o]]
                lambda = x[idx]
                push!(lambdas_minimal_sensitivity, lambda)
                push!(means_minimal_sensitivity, yval[idx])
                push!(errs_minimal_sensitivity, yerr[idx])
            end
        end

        # Plot minimal sensitivity curve
        x = collect(eachindex(means_minimal_sensitivity))
        yval_lo = means_minimal_sensitivity
        yerr = errs_minimal_sensitivity
        errorbar(
            x,
            yval_lo,
            yerr=yerr,
            color=cdict["blue"],
            capsize=4,
            fmt="o--",
            label="Min. sens.",
        )
        for (_x, _y) in zip(x, yval_lo)
            lambda = lambdas_minimal_sensitivity[_x]
            # annotate("\$\\lambda = $lambda\$",
            annotate(lambda,
                xy=(_x, _y), xycoords="data",
                xytext=(-4, -14),
                textcoords="offset points",
                horizontalalignment="center", verticalalignment="center",
                fontsize=14,
            )
        end

        # Extrapolate to infinite order using upper and lower bounds
        meff_min = yval_lo[end]
        meff_max = yval_hi[end]
        # meff_mean = (meff_min + meff_max) / 2
        meff_mean = meff_min
        meff_err = (meff_max - meff_min) + yerr[end] + yerr_hi[end]
        meff_estimate = measurement(meff_mean, meff_err)
        println("rs = $rs:\tm*/m ≈ $meff_estimate")

        # Add meff estimate at this rs to dictionary
        meff_estimates[rs] = meff_estimate

        # Rough estimate of total error using the last 2 orders from hi plus statistical error
        error_estimate_hi = yerr[end] + abs(yval_hi[end] - yval_hi[end-1])
        meff_estimate_hi = measurement(yval_hi[end], error_estimate_hi)
        println("rs = $rs, λ = $fixed_lambda:\tm*/m ≈ $meff_estimate_hi")

        axhspan(
            meff_mean - meff_err,
            meff_mean + meff_err;
            color=cdict["grey"],
            alpha=0.4,
        )
        axhline(meff_mean; linestyle="-", color=cdict["grey"])

        # Add labels
        ylim(ylimits[rs], nothing)
        xticks(collect(1:order[1]))
        xlabel("Perturbation order \$N\$")
        ylabel("\$m^* / m\$")
        legend(; title="\$r_s = $(rs)\$", loc="upper right", fontsize=12)
    end
    tight_layout()
    savefig("meff$(dim)d_beta$(beta)$(modestr)_minimal_sensitivity_vs_N.pdf")
    return meff_estimates
end

function plot_meff_lambda_convergence(maxOrder=order[1]; rs=rs[1], beta=beta[1])
    # The full set of mass2 values for this rs
    all_mass2 = sort(union(rs_to_lambdas[dim][4][rs], rs_to_lambdas[dim][5][rs], rs_to_lambdas[dim][6][rs]))
    meff_means, meff_errors, _mass2 = load_from_dlm(meff_dk_filename; mass2=all_mass2, sortby="order")

    println(meff_means)
    println(all_mass2)
    println(_mass2)

    figure(figsize=(6, 4))
    xmin_plot = Inf
    xmax_plot = -Inf
    xgrid = LinRange(0.0, 5.0, 100)
    for o in 1:maxOrder
        valid_meff = skipmissing(meff_means[o])
        valid_errors = skipmissing(meff_errors[o])
        if isempty(valid_meff)
            continue
        end

        idx_valid_mass2 = collect(eachindex(valid_meff))
        x = _mass2[idx_valid_mass2]
        yval = collect(valid_meff)
        yerr = collect(valid_errors)
        errorbar(
            x,
            yval,
            yerr=yerr,
            color=color[o],
            capsize=4,
            fmt="o",
            markerfacecolor="none",
            label="\$N = $o\$",
            zorder=10 * o,
        )
        xmin_plot = min(x[1], xmin_plot)
        xmax_plot = max(x[end], xmax_plot)
        if o < 5
            xfit, yfit = spline(x, yval, yerr)
            plot(xfit, yfit; color=color[o], linestyle="--")
        end
        if o == maxOrder
            println("\n(N = $o)\nλ = $x\nm*/m = $(measurement.(yval, yerr))\n")
        end
    end
    ncol = 1
    loc = "lower right"
    if dim == 3
        if rs < 1
            xpad = 0.2
        elseif rs < 3
            xpad = 0.1
        elseif rs < 6
            xpad = 0.05
        else
            xpad = 0.2
        end
        xlim(xmin_plot - xpad, xmax_plot + xpad)
        ylim(0.855, 1.0)
        # Plot fixed lambda optima for rs = 1, 2 at d = 3
        if dim == 3 && rs in keys(fixed_lambda_optima_3d)
            if ispolarized
                lambda_optimum = fixed_lambda_optima_3d_GV_spin_polarized[rs]
            else
                lambda_optimum = fixed_lambda_optima_3d[rs]
            end
            # axvline(lambda_optimum; linestyle="-", color="dimgray", zorder=-10)
        end
        if rs == 0.5
            columnspacing = 0.45
            ncol = 1
            xloc = 4.0
            yloc = 0.9825
            ylim(0.865, 1.005)
        elseif rs == 1.0
            loc = (0.155, 0.05)
            columnspacing = 0.9
            ncol = maxOrder > 5 ? 2 : 1
            if ispolarized
                xloc = 2.125
                yloc = 0.98
                ylim(0.83, 1.005)
                # if spinPolarPara == 1.0
                #     text(
                #         0.8,
                #         0.855,
                #         "\$n_\\downarrow = 0\$";
                #         fontsize=16
                #     )
                # else
                #     text(
                #         0.68,
                #         0.98,
                #         "\$\\frac{n_\\uparrow - n_\\downarrow}{n_\\uparrow + n_\\downarrow} = $spinPolarPara\$";
                #         fontsize=16
                #     )
                # end
            else
                # xloc = 2.125
                # yloc = 0.98
                # ylim(0.84, 1.005)
                xloc = 5.6
                yloc = 0.93
                ylim(0.86, 1.005)
                # ylim(0.945, 0.955)
                # text(
                #     1.0,
                #     0.855,
                #     "\$n_\\uparrow = n_\\downarrow\$";
                #     fontsize=16
                # )
            end
        elseif rs == 2.0
            loc = (0.245, 0.05)
            columnspacing = 0.9
            ncol = 2
            xloc = 0.725
            yloc = 0.9875
            ylim(0.865, 1.005)
        elseif rs == 3.0
            loc = (0.18, 0.05)
            columnspacing = 0.9
            ncol = 2
            xloc = 1.0
            yloc = 0.99
            ylim(0.92, 1.005)
        elseif rs == 4.0
            columnspacing = 0.9
            ncol = 2
            xloc = 1.375
            yloc = 0.9965
            ylim(0.94, 1.005)
        elseif rs == 5.0
            columnspacing = 1.8
            ncol = 2
            xloc = 0.55
            yloc = 1.01
            ylim(0.93, 1.025)
        elseif rs == 6.0
            columnspacing = 0.9
            ncol = 2
            xloc = 1.125
            yloc = 1.0
            xlim(0.3, 2.1)
            ylim(0.965, 1.005)
        end
        xmin, xmax = xlim()
        ymin, ymax = ylim()
        if rs < 1
            xstep = 1.0
        elseif rs < 4
            xstep = 0.5
        else
            xstep = 0.25
        end
        big_xticks = collect(range(0.0, 7.0, step=xstep))
        big_yticks = collect(range(0.8, 1.2, step=0.025))
    end
    text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta)\$";
        fontsize=16
    )
    legend(; loc=loc, ncol=ncol, columnspacing=columnspacing)
    xlabel("\$\\lambda\$ (Ry)")
    ylabel("\$m^* / m\$")
    savefig("meff$(dim)d_rs$(rs)_beta$(beta)$(modestr)_vs_lambda.pdf")
end

if abspath(PROGRAM_FILE) == @__FILE__
    meff_estimates = plot_combined_order_convergence(plot_rs=[1.0, 2.0, 3.0, 4.0, 5.0])
    plot_regular_vs_inverse_convergence_comparisons(meff_estimates)
    plot_cancellation_convergence(meff_estimates)
    # plot_meff_order_convergence(plot_rs=[1.0, 2.0, 3.0, 4.0, 5.0])
    plot_meff_order_convergence(plot_rs=[1.0])
    plot_meff_lambda_convergence()
    println("\nFinal estimates for m*/m:")
    for (rs, meff) in sort(collect(meff_estimates))
        println("rs = $rs:\tm/m* ≈ $meff")
    end
end
