#!/bin/bash
julia derive_zmu.jl --save &&
    julia derive_zinv.jl --save &&
    julia derive_chemical_potential.jl --save &&
    julia derive_inverse_dispersion_ratio.jl --save &&
    julia derive_dispersion_ratio.jl --save &&
    julia derive_meff.jl --save
