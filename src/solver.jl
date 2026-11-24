# The solver side of cnlplib (`cnlp_solver.h`): a problem handed to a solver as sizes,
# bounds and evaluation callbacks, and the result the solver writes back. A Julia solver
# compiled as a cnlp solver library takes a `Ptr{CNLPProblem}`, solves `ProblemModel(p)`
# with any NLPModels solver, and reports through `set_result!`; `problem(m)` goes the other
# way, handing a `CNLPModel` to a cnlp solver library.

"""
    CNLPProblem

The layout of `cnlp_problem` (`cnlp_solver.h`): sizes, bounds, a starting point, optional
starting multipliers, and the evaluation callbacks, each called with `ctx`.
"""
struct CNLPProblem
    ctx::Ptr{Cvoid}
    nvar::Int64
    ncon::Int64
    nnzj::Int64
    nnzh::Int64
    minimize::Int64
    x0::Ptr{Float64}
    lvar::Ptr{Float64}
    uvar::Ptr{Float64}
    lcon::Ptr{Float64}
    ucon::Ptr{Float64}
    y0::Ptr{Float64}
    zl0::Ptr{Float64}
    zu0::Ptr{Float64}
    obj::Ptr{Cvoid}
    grad::Ptr{Cvoid}
    cons::Ptr{Cvoid}
    jac_structure::Ptr{Cvoid}
    jac::Ptr{Cvoid}
    hess_structure::Ptr{Cvoid}
    hess::Ptr{Cvoid}
end

"""
    CNLPResult

The layout of `cnlp_result` (`cnlp_solver.h`): the caller's arrays for the solution
and multipliers, which the solver fills, and the objective, status, iteration count and
the solver's own status text.
"""
struct CNLPResult
    x::Ptr{Float64}
    y::Ptr{Float64}
    zl::Ptr{Float64}
    zu::Ptr{Float64}
    objective::Float64
    status::Int64
    iterations::Int64
    solver_status::NTuple{128, UInt8}
end

# Solver-neutral statuses (`cnlp_solver.h`).
const CNLP_SOLVED = 0
const CNLP_ACCEPTABLE = 1
const CNLP_INFEASIBLE = 2
const CNLP_UNBOUNDED = 3
const CNLP_ITERATION_LIMIT = 4
const CNLP_TIME_LIMIT = 5
const CNLP_ERROR = 6

"The version of the solver interface implemented here (`CNLP_SOLVER_VERSION`: 0.1)."
const CNLP_SOLVER_VERSION = 1

"""
    CallbackError(name, status)

A problem's callback `name` returned the nonzero `status`.
"""
struct CallbackError <: Exception
    name::Symbol
    status::Int64
end
Base.showerror(io::IO, e::CallbackError) = print(io, "the problem's ", e.name, " callback returned ", e.status)
_check_cb(st::Int64, name::Symbol) = st == 0 || throw(CallbackError(name, st))

"""
    ProblemModel(problem::CNLPProblem)
    ProblemModel(problem::Ptr{CNLPProblem})

An `AbstractNLPModel{Float64, Vector{Float64}}` over a problem given as callbacks, as a
cnlp solver receives it. The bounds, starting point and starting multipliers are copied;
evaluations call the problem's callbacks, and a failing callback throws
[`CallbackError`](@ref). `jprod`/`jtprod` are computed from the Jacobian's coordinates.
"""
struct ProblemModel <: AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
    p::CNLPProblem
    # the Jacobian in coordinates, for the products (filled on first use)
    jrows::Vector{Int64}
    jcols::Vector{Int64}
    jvals::Vector{Float64}
end

ProblemModel(p::Ptr{CNLPProblem}) = ProblemModel(unsafe_load(p))

function ProblemModel(p::CNLPProblem)
    n, m = Int(p.nvar), Int(p.ncon)
    vec(ptr, k) = k == 0 || ptr == C_NULL ? zeros(k) : copy(unsafe_wrap(Array, ptr, k))
    meta = NLPModelMeta(
        n; ncon = m, nnzj = Int(p.nnzj), nnzh = Int(p.nnzh),
        x0 = vec(p.x0, n), lvar = vec(p.lvar, n), uvar = vec(p.uvar, n),
        y0 = vec(p.y0, m), lcon = vec(p.lcon, m), ucon = vec(p.ucon, m), minimize = p.minimize != 0,
    )
    return ProblemModel(meta, Counters(), p, Int64[], Int64[], Float64[])
end

"""
    start_multipliers(m::ProblemModel) -> (zl, zu) or nothing

The problem's starting bound multipliers (`zl0`, `zu0`), when it gives them.
"""
function start_multipliers(m::ProblemModel)
    (m.p.zl0 == C_NULL || m.p.zu0 == C_NULL) && return nothing
    n = m.meta.nvar
    return copy(unsafe_wrap(Array, m.p.zl0, n)), copy(unsafe_wrap(Array, m.p.zu0, n))
end

function NLPModels.obj(m::ProblemModel, x::AbstractVector{Float64})
    increment!(m, :neval_obj)
    f = Ref(0.0)
    _check_cb(ccall(m.p.obj, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), m.p.ctx, x, f), :obj)
    return f[]
end

function NLPModels.grad!(m::ProblemModel, x::AbstractVector{Float64}, g::AbstractVector{Float64})
    increment!(m, :neval_grad)
    _check_cb(ccall(m.p.grad, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), m.p.ctx, x, g), :grad)
    return g
end

function NLPModels.cons_nln!(m::ProblemModel, x::AbstractVector{Float64}, c::AbstractVector{Float64})
    increment!(m, :neval_cons_nln)
    m.meta.ncon == 0 && return c
    _check_cb(ccall(m.p.cons, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), m.p.ctx, x, c), :cons)
    return c
end

function _structure(m::ProblemModel, f::Ptr{Cvoid}, nnz::Int, rows, cols, name::Symbol)
    r = Vector{Int64}(undef, nnz); c = Vector{Int64}(undef, nnz)
    nnz == 0 || _check_cb(ccall(f, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64}), m.p.ctx, r, c), name)
    copyto!(rows, r); copyto!(cols, c)
    return rows, cols
end
NLPModels.jac_nln_structure!(m::ProblemModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) =
    _structure(m, m.p.jac_structure, m.meta.nnzj, rows, cols, :jac_structure)
NLPModels.hess_structure!(m::ProblemModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) =
    _structure(m, m.p.hess_structure, m.meta.nnzh, rows, cols, :hess_structure)

function NLPModels.jac_nln_coord!(m::ProblemModel, x::AbstractVector{Float64}, vals::AbstractVector{Float64})
    increment!(m, :neval_jac_nln)
    m.meta.nnzj == 0 && return vals
    _check_cb(ccall(m.p.jac, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), m.p.ctx, x, vals), :jac)
    return vals
end

function NLPModels.hess_coord!(m::ProblemModel, x::AbstractVector{Float64}, y::AbstractVector{Float64},
                               vals::AbstractVector{Float64}; obj_weight::Real = 1.0)
    increment!(m, :neval_hess)
    m.meta.nnzh == 0 && return vals
    _check_cb(ccall(m.p.hess, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Float64, Ptr{Float64}),
                 m.p.ctx, x, y, Float64(obj_weight), vals), :hess)
    return vals
end
NLPModels.hess_coord!(m::ProblemModel, x::AbstractVector{Float64}, vals::AbstractVector{Float64};
                      obj_weight::Real = 1.0) = NLPModels.hess_coord!(m, x, zeros(m.meta.ncon), vals; obj_weight)

# J v and J' v from the Jacobian's coordinates (the interface has no products)
function _jacobian!(m::ProblemModel, x::AbstractVector{Float64})
    if length(m.jrows) != m.meta.nnzj
        resize!(m.jrows, m.meta.nnzj); resize!(m.jcols, m.meta.nnzj); resize!(m.jvals, m.meta.nnzj)
        NLPModels.jac_nln_structure!(m, m.jrows, m.jcols)
    end
    NLPModels.jac_nln_coord!(m, x, m.jvals)
    return m
end

function NLPModels.jprod_nln!(m::ProblemModel, x::AbstractVector{Float64}, v::AbstractVector{Float64},
                              Jv::AbstractVector{Float64})
    increment!(m, :neval_jprod_nln)
    _jacobian!(m, x)
    fill!(Jv, 0.0)
    @inbounds for k in eachindex(m.jvals)
        Jv[m.jrows[k]] += m.jvals[k] * v[m.jcols[k]]
    end
    return Jv
end

function NLPModels.jtprod_nln!(m::ProblemModel, x::AbstractVector{Float64}, v::AbstractVector{Float64},
                               Jtv::AbstractVector{Float64})
    increment!(m, :neval_jtprod_nln)
    _jacobian!(m, x)
    fill!(Jtv, 0.0)
    @inbounds for k in eachindex(m.jvals)
        Jtv[m.jcols[k]] += m.jvals[k] * v[m.jrows[k]]
    end
    return Jtv
end

"""
    solver_options(noptions, keys, values) -> Vector{Pair{String, String}}

The options a cnlp solver receives (`key=value` strings), read from the C arrays.
"""
solver_options(n::Integer, keys::Ptr{Ptr{UInt8}}, vals::Ptr{Ptr{UInt8}}) =
    Pair{String, String}[unsafe_string(unsafe_load(keys, k)) => unsafe_string(unsafe_load(vals, k)) for k in 1:n]

_text128(s::AbstractString) = (b = codeunits(s); ntuple(i -> i < 128 && i <= length(b) ? b[i] : 0x00, 128))

"""
    set_result!(result::Ptr{CNLPResult}, m::ProblemModel, x, y, zl, zu, objective, status,
                iterations, solver_status)

Write a solve's outcome to the caller's `cnlp_result`. The multipliers are given in the
NLPModels convention, `∇f + Jᵀy − zL + zU = 0` in either sense (as MadNLP and Ipopt.jl
return them), and are written in the cnlp convention, `s∇f + Jᵀy − zL + zU = 0` with
`s = −1` when maximizing, `zL, zU ≥ 0`. `objective` is `f(x)` in the model's sense.
"""
function set_result!(result::Ptr{CNLPResult}, m::ProblemModel, x::AbstractVector, y::AbstractVector,
                     zl::AbstractVector, zu::AbstractVector, objective::Real, status::Integer,
                     iterations::Integer, solver_status::AbstractString)
    r = unsafe_load(result)
    n, k = m.meta.nvar, m.meta.ncon
    s = m.meta.minimize ? 1.0 : -1.0
    for i in 1:n
        unsafe_store!(r.x, x[i], i)
        unsafe_store!(r.zl, s * zl[i], i)
        unsafe_store!(r.zu, s * zu[i], i)
    end
    for i in 1:k
        unsafe_store!(r.y, s * y[i], i)
    end
    unsafe_store!(result, CNLPResult(r.x, r.y, r.zl, r.zu, Float64(objective), Int64(status),
                                     Int64(iterations), _text128(solver_status)))
    return result
end

"""
    set_failure!(result::Ptr{CNLPResult}, message)

Mark a solve that could not run (`CNLP_ERROR`, no iterations, objective NaN), with the
reason in `solver_status`; the arrays are left as they are.
"""
function set_failure!(result::Ptr{CNLPResult}, message::AbstractString)
    r = unsafe_load(result)
    unsafe_store!(result, CNLPResult(r.x, r.y, r.zl, r.zu, NaN, Int64(CNLP_ERROR), Int64(-1), _text128(message)))
    return result
end

# ── a CNLPModel handed to a cnlp solver ──────────────────────────────────────────────

"""
    ModelProblem

A `CNLPModel` as a `cnlp_problem` ([`problem`](@ref)): converts to `Ptr{CNLPProblem}` for a
`ccall`; `p[]` is the `CNLPProblem`. The callbacks find the model through this object, so
keep it alive while a solver uses the problem.
"""
mutable struct ModelProblem
    model::CNLPModel{Vector{Float64}}
    ref::Base.RefValue{CNLPProblem}
    ModelProblem(m::CNLPModel{Vector{Float64}}) = new(m)
end
Base.unsafe_convert(::Type{Ptr{CNLPProblem}}, h::ModelProblem) = Base.unsafe_convert(Ptr{CNLPProblem}, h.ref)
Base.getindex(h::ModelProblem) = h.ref[]

_model(c::Ptr{Cvoid}) = (unsafe_pointer_to_objref(c)::ModelProblem).model
_wrap(p::Ptr{T}, n) where {T} = unsafe_wrap(Array, p, n)
function _guard(f)
    try
        f()
        return Int64(0)
    catch
        return Int64(1)
    end
end
_cb_obj(c::Ptr{Cvoid}, x::Ptr{Float64}, f::Ptr{Float64}) =
    _guard(() -> (m = _model(c); unsafe_store!(f, NLPModels.obj(m, _wrap(x, m.meta.nvar)))))
_cb_grad(c::Ptr{Cvoid}, x::Ptr{Float64}, g::Ptr{Float64}) =
    _guard(() -> (m = _model(c); NLPModels.grad!(m, _wrap(x, m.meta.nvar), _wrap(g, m.meta.nvar))))
_cb_cons(c::Ptr{Cvoid}, x::Ptr{Float64}, v::Ptr{Float64}) =
    _guard(() -> (m = _model(c); NLPModels.cons!(m, _wrap(x, m.meta.nvar), _wrap(v, m.meta.ncon))))
_cb_js(c::Ptr{Cvoid}, r::Ptr{Int64}, k::Ptr{Int64}) =
    _guard(() -> (m = _model(c); NLPModels.jac_structure!(m, _wrap(r, m.meta.nnzj), _wrap(k, m.meta.nnzj))))
_cb_jac(c::Ptr{Cvoid}, x::Ptr{Float64}, v::Ptr{Float64}) =
    _guard(() -> (m = _model(c); NLPModels.jac_coord!(m, _wrap(x, m.meta.nvar), _wrap(v, m.meta.nnzj))))
_cb_hs(c::Ptr{Cvoid}, r::Ptr{Int64}, k::Ptr{Int64}) =
    _guard(() -> (m = _model(c); NLPModels.hess_structure!(m, _wrap(r, m.meta.nnzh), _wrap(k, m.meta.nnzh))))
_cb_hess(c::Ptr{Cvoid}, x::Ptr{Float64}, y::Ptr{Float64}, w::Float64, v::Ptr{Float64}) =
    _guard(() -> (m = _model(c); NLPModels.hess_coord!(m, _wrap(x, m.meta.nvar), _wrap(y, m.meta.ncon),
                                                       _wrap(v, m.meta.nnzh); obj_weight = w)))

"""
    problem(m::CNLPModel{Vector{Float64}}) -> ModelProblem

`m` as a `cnlp_problem`, to hand to a cnlp solver library (`cnlp_solve`). The callbacks
evaluate `m` on the host. Use the problem under `GC.@preserve`:

    p = problem(m)
    GC.@preserve p ccall(solve, Int64, (Ptr{CNLPProblem}, ...), p, ...)
"""
function problem(m::CNLPModel{Vector{Float64}})
    meta = m.meta
    h = ModelProblem(m)
    h.ref = Ref(CNLPProblem(
        pointer_from_objref(h), meta.nvar, meta.ncon, meta.nnzj, meta.nnzh, meta.minimize ? 1 : 0,
        pointer(meta.x0), pointer(meta.lvar), pointer(meta.uvar), pointer(meta.lcon), pointer(meta.ucon),
        C_NULL, C_NULL, C_NULL,
        @cfunction(_cb_obj, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
        @cfunction(_cb_grad, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
        @cfunction(_cb_cons, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
        @cfunction(_cb_js, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64})),
        @cfunction(_cb_jac, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
        @cfunction(_cb_hs, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64})),
        @cfunction(_cb_hess, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Float64, Ptr{Float64})),
    ))
    return h
end
