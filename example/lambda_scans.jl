const LambdaDictType = Dict{Float64,Vector{Float64}}

# Finalized 3D lambda scans for maximum orders N = 4, 5, 6
const rs_to_lambdas_3d_N4 = LambdaDictType(
    0.5 => [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0],
    1.0 => [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0],
    2.0 => [0.5, 0.75, 1.0, 1.25, 1.5, 1.625, 1.75, 1.875, 2.0, 2.5, 3.0],
    3.0 => [0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 1.75, 2.0],
    4.0 => [0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 2.0],
    5.0 => [0.375, 0.5, 0.625, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.5],
    6.0 => [0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0],
)
const rs_to_lambdas_3d_N5 = LambdaDictType(
    0.5 => [3.0, 3.5, 4.0, 4.5, 5.0],
    1.0 => [1.5, 1.75, 2.0, 3.0, 3.5],
    2.0 => [1.625, 1.75, 1.875, 2.0, 2.125, 2.25, 2.5],
    3.0 => [1.0, 1.125, 1.25, 1.5, 1.75, 2.0],
    4.0 => [0.875, 1.0, 1.125, 1.25, 1.5, 2.0],
    5.0 => [0.8125, 0.875, 0.9375, 1.0, 1.125],
    6.0 => [0.625, 0.75, 0.875, 1.0, 1.125, 1.25],
)
const rs_to_lambdas_3d_N6 = LambdaDictType(
    1.0 => [1.75],
) # TODO: Add N = 6 3D lambda scans and datfile entries

# TODO: Add N = 4, 5, 6 2D lambda scans and datfiles
const rs_to_lambdas_2d_N4 = LambdaDictType(
    #    1.0 => [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
    1.0 => [0.2, 0.1, 0.05, 0.025, 0.0125, 0.0075, 0.0035],
    2.0 => [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.0075],
)
const rs_to_lambdas_2d_N5 = LambdaDictType()
const rs_to_lambdas_2d_N6 = LambdaDictType()

"""
Dict of dict of 3D lambda scans vs N and rs (usage: `rs_to_lambdas_3d[N][rs]`).
"""
const rs_to_lambdas_3d = Dict(
    4 => rs_to_lambdas_3d_N4,
    5 => rs_to_lambdas_3d_N5,
    6 => rs_to_lambdas_3d_N6,
)

"""
Dict of dict of 2D lambda scans vs N and rs (usage: `rs_to_lambdas_3d[N][rs]`).
"""
const rs_to_lambdas_2d = Dict(
    4 => rs_to_lambdas_2d_N4,
    5 => rs_to_lambdas_2d_N5,
    6 => rs_to_lambdas_2d_N6,
)

"""
Dict of dict of dict of lambda scans vs dim, N, and rs (usage: `rs_to_lambdas[dim][N][rs]`).
"""
const rs_to_lambdas = Dict(
    2 => rs_to_lambdas_2d,
    3 => rs_to_lambdas_3d,
)