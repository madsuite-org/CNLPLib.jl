"""
Device models for libraries that export the cnlp device tier (`P_device`).

`CNLPModel(lib, args...; device = :cuda)` returns a [`CuCNLPModel`](@ref): the same
instance, with `CuVector` metadata and buffers, evaluated through the library's
`P_*_d` callbacks on device pointers. It is an `AbstractNLPModel{Float64,
CuVector{Float64}}`, the model type GPU solvers such as cuMadNLP consume.
"""
module CNLPModelsCUDAExt

using CNLPModels, NLPModels, CUDACore
using CUDACore: CuVector, StridedCuVector
using CNLPModels: CNLPModel, _check
import Libdl

struct CuCNLPModel{VT <: CuVector{Float64}} <: AbstractNLPModel{Float64, VT}
    meta::NLPModelMeta{Float64, VT}
    counters::Counters
    host::CNLPModel                  # the same instance, for structure, layout and parameters
    obj_p::Ptr{Cvoid}
    grad_p::Ptr{Cvoid}
    cons_p::Ptr{Cvoid}
    jac_p::Ptr{Cvoid}
    hess_p::Ptr{Cvoid}
    zeroy::VT                        # multipliers for the objective-only Hessian
    # the structures on the device and value buffers, for the products below
    jrows::CuVector{Int}
    jcols::CuVector{Int}
    jbuf::VT
    hrows::CuVector{Int}
    hcols::CuVector{Int}
    hbuf::VT
end

# A contiguous CuArray (or contiguous view of one) as the raw device pointer the ABI takes.
_ptr(a::StridedCuVector{Float64}) = reinterpret(Ptr{Cdouble}, UInt(pointer(a)))

function CNLPModels._device_model(::Val{:cuda}, m::CNLPModel)
    sym(s) = Symbol(m.prefix, "_", s)
    dp = Libdl.dlsym(m.lib.handle, sym(:device); throw_error = false)
    dp === nothing && throw(ArgumentError(
        "`$(m.prefix)` has no device tier: the library does not export `$(sym(:device))` " *
        "(compile it with `compile_library(...; device = :cuda)`)"))
    space = ccall(dp, Cint, (Cint,), m.id)
    space == 1 || throw(ArgumentError("`$(sym(:device))` reports memory space $space, not CUDA (1)"))
    hm = m.meta
    meta = NLPModelMeta(
        hm.nvar; ncon = hm.ncon, nnzj = hm.nnzj, nnzh = hm.nnzh,
        x0 = CuVector(hm.x0), lvar = CuVector(hm.lvar), uvar = CuVector(hm.uvar),
        y0 = CuVector(hm.y0), lcon = CuVector(hm.lcon), ucon = CuVector(hm.ucon),
        minimize = hm.minimize, name = hm.name,
    )
    fp(s) = Libdl.dlsym(m.lib.handle, sym(s))
    VT = typeof(meta.x0)
    return CuCNLPModel(meta, Counters(), m, fp(:obj_d), fp(:grad_d), fp(:cons_d), fp(:jac_d), fp(:hess_d),
                       CUDACore.zeros(Float64, hm.ncon),
                       CuVector(m.jrows), CuVector(m.jcols), VT(undef, hm.nnzj),
                       CuVector(m.hrows), CuVector(m.hcols), VT(undef, hm.nnzh))
end

# The library launches on the legacy default stream and synchronizes its context
# before returning; CUDA.jl works on task-local streams. Finishing this task's
# stream first is what orders the library's reads after our writes.
@inline _handoff() = CUDACore.synchronize()

function NLPModels.obj(m::CuCNLPModel, x::StridedCuVector{Float64})
    @lencheck m.meta.nvar x
    increment!(m, :neval_obj)
    _handoff()
    out = Ref{Cdouble}(0.0)
    _check(ccall(m.obj_p, Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}), m.host.id, _ptr(x), out), :obj_d)
    return out[]
end

function NLPModels.grad!(m::CuCNLPModel, x::StridedCuVector{Float64}, g::StridedCuVector{Float64})
    @lencheck m.meta.nvar x g
    increment!(m, :neval_grad)
    _handoff()
    _check(ccall(m.grad_p, Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}), m.host.id, _ptr(x), _ptr(g)), :grad_d)
    return g
end

for (f, counter) in ((:cons!, :neval_cons), (:cons_nln!, :neval_cons_nln))
    @eval function NLPModels.$f(m::CuCNLPModel, x::StridedCuVector{Float64}, c::StridedCuVector{Float64})
        @lencheck m.meta.nvar x
        @lencheck m.meta.ncon c
        increment!(m, $(QuoteNode(counter)))
        m.meta.ncon == 0 && return c
        _handoff()
        _check(ccall(m.cons_p, Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}), m.host.id, _ptr(x), _ptr(c)), :cons_d)
        return c
    end
end

function NLPModels.jac_coord!(m::CuCNLPModel, x::StridedCuVector{Float64}, vals::StridedCuVector{Float64})
    @lencheck m.meta.nvar x
    @lencheck m.meta.nnzj vals
    increment!(m, :neval_jac)
    m.meta.nnzj == 0 && return vals
    _handoff()
    _check(ccall(m.jac_p, Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}), m.host.id, _ptr(x), _ptr(vals)), :jac_d)
    return vals
end

function NLPModels.hess_coord!(
    m::CuCNLPModel, x::StridedCuVector{Float64}, y::StridedCuVector{Float64},
    vals::StridedCuVector{Float64}; obj_weight::Real = 1.0,
)
    @lencheck m.meta.nvar x
    @lencheck m.meta.ncon y
    @lencheck m.meta.nnzh vals
    increment!(m, :neval_hess)
    m.meta.nnzh == 0 && return vals
    _handoff()
    yp = m.meta.ncon == 0 ? _ptr(m.zeroy) : _ptr(y)
    _check(ccall(m.hess_p, Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Ptr{Cdouble}),
                 m.host.id, _ptr(x), yp, Cdouble(obj_weight), _ptr(vals)), :hess_d)
    return vals
end

NLPModels.hess_coord!(m::CuCNLPModel, x::StridedCuVector{Float64}, vals::StridedCuVector{Float64};
                      obj_weight::Real = 1.0) =
    NLPModels.hess_coord!(m, x, m.zeroy, vals; obj_weight)

# Structures are host state, computed once (cached on the host model); this copies them over.
function NLPModels.jac_structure!(m::CuCNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    @lencheck m.meta.nnzj rows cols
    copyto!(rows, convert(Vector{eltype(rows)}, m.host.jrows))
    copyto!(cols, convert(Vector{eltype(cols)}, m.host.jcols))
    return rows, cols
end
function NLPModels.hess_structure!(m::CuCNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    @lencheck m.meta.nnzh rows cols
    copyto!(rows, convert(Vector{eltype(rows)}, m.host.hrows))
    copyto!(cols, convert(Vector{eltype(cols)}, m.host.hcols))
    return rows, cols
end

# ── products, assembled from the COO values the ABI provides ──────────────────────────
# One callback into the cached value buffer, then a COO matvec on the device: what
# CNLPModels does on the host, for solvers that ask for products (MadNLP does, in some
# phases). A library with true device products would be a future ABI addition.

function _coo!(y, rows, cols, vals, x, n)
    k = (CUDACore.blockIdx().x - Int32(1)) * CUDACore.blockDim().x + CUDACore.threadIdx().x
    if k <= n
        @inbounds CUDACore.@atomic y[rows[k]] += vals[k] * x[cols[k]]
    end
    return
end
# the Hessian is stored as its lower triangle; off-diagonal entries act both ways
function _sym!(y, rows, cols, vals, x, n)
    k = (CUDACore.blockIdx().x - Int32(1)) * CUDACore.blockDim().x + CUDACore.threadIdx().x
    if k <= n
        @inbounds begin
            i, j, h = rows[k], cols[k], vals[k]
            CUDACore.@atomic y[i] += h * x[j]
            i == j || (CUDACore.@atomic y[j] += h * x[i])
        end
    end
    return
end
function _matvec!(f, y, rows, cols, vals, x)
    fill!(y, 0.0)
    n = length(vals)
    n == 0 && return y
    CUDACore.@cuda threads = 256 blocks = cld(n, 256) f(y, rows, cols, vals, x, n)
    return y
end

for (fn, counter) in ((:jprod_nln!, :neval_jprod), (:jprod!, :neval_jprod))
    @eval function NLPModels.$fn(m::CuCNLPModel, x::StridedCuVector{Float64}, v::StridedCuVector{Float64},
                                 Jv::StridedCuVector{Float64})
        @lencheck m.meta.nvar x v
        @lencheck m.meta.ncon Jv
        increment!(m, $(QuoteNode(counter)))
        NLPModels.jac_coord!(m, x, m.jbuf)
        return _matvec!(_coo!, Jv, m.jrows, m.jcols, m.jbuf, v)
    end
end
for (fn, counter) in ((:jtprod_nln!, :neval_jtprod), (:jtprod!, :neval_jtprod))
    @eval function NLPModels.$fn(m::CuCNLPModel, x::StridedCuVector{Float64}, v::StridedCuVector{Float64},
                                 Jtv::StridedCuVector{Float64})
        @lencheck m.meta.nvar x Jtv
        @lencheck m.meta.ncon v
        increment!(m, $(QuoteNode(counter)))
        NLPModels.jac_coord!(m, x, m.jbuf)
        return _matvec!(_coo!, Jtv, m.jcols, m.jrows, m.jbuf, v)
    end
end
function NLPModels.hprod!(m::CuCNLPModel, x::StridedCuVector{Float64}, y::StridedCuVector{Float64},
                          v::StridedCuVector{Float64}, Hv::StridedCuVector{Float64}; obj_weight::Real = 1.0)
    @lencheck m.meta.nvar x v Hv
    @lencheck m.meta.ncon y
    increment!(m, :neval_hprod)
    NLPModels.hess_coord!(m, x, y, m.hbuf; obj_weight)
    return _matvec!(_sym!, Hv, m.hrows, m.hcols, m.hbuf, v)
end
NLPModels.hprod!(m::CuCNLPModel, x::StridedCuVector{Float64}, v::StridedCuVector{Float64},
                 Hv::StridedCuVector{Float64}; obj_weight::Real = 1.0) =
    NLPModels.hprod!(m, x, m.zeroy, v, Hv; obj_weight)

end # module
