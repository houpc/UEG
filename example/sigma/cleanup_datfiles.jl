using UniElectronGas, ElectronLiquid
using JLD2, CSV, DataFrames, Measurements

include("../input.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    isSave = false
    if length(ARGS) >= 1 && (ARGS[1] == "s" || ARGS[1] == "-s" || ARGS[1] == "--save" || ARGS[1] == " save")
        # the second parameter may be set to save the derived parameters
        isSave = true
    end

    # Sort datfiles by rs, beta, and N and prune duplicate entries
    datfiles = [
        zfactor_filename,
        zinv_filename,
        chemical_potential_filename,
        inverse_dispersion_ratio_dk_filename,
        dispersion_ratio_dk_filename,
        meff_dk_filename,
        inverse_meff_dk_filename,
    ]
    for datfile in datfiles
        if isfile(datfile)
            df = DataFrame(CSV.File(datfile; header=false))
            # Remove exact duplicate entries
            unique!(df)
            println("Cleaned exact duplicates in file $datfile:\n$df\n")
            # Remove less accurate entries for duplicate parameters
            indices_minimum_error = []
            df[!, :Index] = 1:nrow(df)  # append a column of indices to the dataframe
            for group in groupby(df, [:Column1, :Column2, :Column3, :Column4])
                min_idx = Inf
                min_err = Inf
                for (i, row) in enumerate(eachrow(group))
                    errs = [measurement(v).err for v in row[5:(end-1)] if !ismissing(v)]
                    total_err = sum(errs)
                    if total_err < min_err
                        min_err = total_err
                        min_idx = row[end]  # row.Index
                    end
                end
                push!(indices_minimum_error, min_idx)
            end
            df = select!(DataFrame(eachrow(df)[indices_minimum_error]), Not([:Index]))
            sort!(df, [1, 2, 3, 4])
            println("Removed less accurate duplicate entries in file $datfile:\n$df\n")
            if isSave
                CSV.write(datfile, df; delim="\t", header=false)
            end
        end
    end
end