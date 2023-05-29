module QuadSolve
using Base.Threads
using SIMD
using MuladdMacro
using DataFrames
using CSV
export quadsolve, solve, solve!, linsolve

include("types.jl")

function linsolve(a::T, b::T) where T
    -b/a
end

begin
    local disc
    @inline signs()::Vec{2, Float64} = signs(Float64)
    @inline function signs(v::Union{T, Type{T}})::Vec{2, T} where {T}
        Vec{2, T}((one(v), -one(v)))
    end
    @inline @muladd function disc(a::T, b::T, c::T)::T where T 
        b^2 - 4*a*c
    end
    @inline @muladd function quadsolve(a::Complex{T}, b::Complex{T}, c::Complex{T})::NTuple{2, Complex{T}} where {T <: AbstractFloat}
        if iszero(a)
            if iszero(b)
                return (NaN, NaN)
            end
            x = linsolve(b, c)
            return (x, x)
        end
        d1 = sqrt(disc(a, b, c))
        a2 = inv(2a)
        ((-b+d1)*a2, (-b-d1)*a2)
    end
    @inline @muladd function quadsolve(a::T, b::T, c::T)::NTuple{2, T} where {T <: AbstractFloat}
        if iszero(a)
            if iszero(b)
                return (NaN, NaN)
            end
            x = linsolve(b, c)
            return (x, x)
        end
        d1 = disc(a, b, c)
        if d1 < 0
            return (NaN, NaN)
        end
        d1 = sqrt(d1)
        a2 = inv(2a)
        rv = (signs(T)*d1-b)*a2
        (rv[1], rv[2])
    end
    @inline function quadsolve(a::FComplex, b::FComplex, c::FComplex)
        quadsolve(promote(a, b, c)...)
    end
    @inline function quadsolve(a::Number, b::Number, c::Number)
        quadsolve(promote(0.0+a, 0.0+b, 0.0+c)...) 
    end
end

function _solve!(root1s::AbstractVector{T}, root2s::AbstractVector{T}, av::AbstractVector, bv::AbstractVector, cv::AbstractVector) where T
    Base.require_one_based_indexing(root1s, root2s, av, bv, cv)
    for i = 1:min(length(root1s), length(root2s), length(av), length(bv), length(cv))
        t = quadsolve(av[i], bv[i], cv[i])
        if isa(t, NTuple{2, T})
            root1s[i], root2s[i] = t
        else
            root1s[i] = root2s[i] = NaN
        end
    end
end

solve(df::DataFrame; T=promote_type(eltype(df.a), eltype(df.b), eltype(df.c))) = solve!(copy(df), T=T)


function solve!(df::DataFrame; T=promote_type(eltype(df.a), eltype(df.b), eltype(df.c)))
    s = size(df, 1)
    root1s, root2s = Vector{T}(undef, s), Vector{T}(undef, s)
    _solve!(root1s, root2s, df.a, df.b, df.c)
    insertcols!(df, :root1 => root1s, :root2 => root2s, copycols=false)
    df
end

randf(n, x=n/2) = DataFrame(a=randn(n)*x, b=randn(n)*x, c=randn(n)*x)

function main(args::Vector{String})
    ns = parse.(Int, args[1])
    for n = ns
        df = randf(n, Complex(n))
        println(solve!(df))
    end
end

end # module QuadSolve
