"""
The HDF5 form of the standard solution format: ExaModelsIO's `/solution` group, so
`ExaModelsIO.read_solution` reads it, plus `/blocks/<kind>/<name>/<field>`.
"""
module CNLPModelsHDF5Ext

using CNLPModels, HDF5

function CNLPModels._write_standard(::Val{:h5}, path::AbstractString, r)
    h5open(path, "w") do f
        a = attributes(f)
        a["format"] = "cnlp-solution"; a["format_version"] = "1.0"
        a["library"] = r.library; a["model"] = r.model; a["minimize"] = Int(r.minimize)
        g = create_group(f, "solution")
        g["solution"] = r.x; g["multipliers"] = r.y; g["multipliers_L"] = r.zl; g["multipliers_U"] = r.zu
        g["objective"] = r.objective; g["status"] = r.status; g["iterations"] = r.iterations
        b = create_group(f, "blocks")
        for (kind, name, fields) in r.blocks
            k = haskey(b, kind) ? b[kind] : create_group(b, kind)
            h = create_group(k, name)
            for (field, arr) in fields
                h[field] = arr
            end
        end
    end
    return path
end

end # module
