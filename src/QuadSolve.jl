module QuadSolve
using Printf
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
    @inline function quadsolve(a::Number, b::Number, c::Number)
        T = promote_type(typeof(a), typeof(b), typeof(c))
        if T <: FComplex
            quadsolve(T(a), T(b), T(c))
        elseif T <: Real
            let T = Float64
                quadsolve(Float64(a), Float64(b), Float64(c))
            end
        else
            let T = Complex{Float64}
                quadsolve(T(a), T(b), T(c))
            end
        end
    end
end

function _solve!(root1s::AbstractVector{T}, root2s::AbstractVector{T}, av::AbstractVector, bv::AbstractVector, cv::AbstractVector) where T
    Base.require_one_based_indexing(root1s, root2s, av, bv, cv)
    @assert reduce((r, a) -> r==a, [length(root1s), length(root2s), length(av), length(bv)], init=length(cv))
    for i = 1:length(root1s)
        root1s[i], root2s[i] = quadsolve(av[i], bv[i], cv[i])
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
randt(x) = (randn()*x, randn()*x, randn()*x)

function main(args::Vector{String})
    skipno = 0
    if length(args) >= 1
        skipno = parse(Int, args[1])
    end
    @printf("%s\t%s\t%s\t%s\t%s\n", "a", "b", "c", "r1", "r2")
    for (i, l) = enumerate(eachline(stdin))
        if i <= skipno 
            continue
        end
        try
            (a, b, c, _...) = parse.(Float64, split(l, r"\t+"))
            r1, r2 = quadsolve(a, b, c)
            @printf("%s\t%s\t%s\t%s\t%s\n", a, b, c, r1, r2)
        catch e
            println(stderr, e)
            continue
        end
    end
end

end # module QuadSolve
