module QuadSolve
using Printf
using Base.Threads
using SIMD
using MuladdMacro
using DataFrames
using CSV
export quadsolve, solve, solve!, linsolve

include("types.jl")

@inline function linsolve(a::T, b::T) where {T<:Number}
    -b/a
end

begin
    local disc
    @inline signs()::Vec{2, Float64} = signs(Float64)
    @inline function signs(v::Union{T, Type{T}})::Vec{2, T} where {T}
        Vec{2, T}((one(v), -one(v)))
    end
    @inline @muladd function disc(a::T, b::T, c::T)::T where {T}
        b^2 - 4*a*c
    end
    @inline @muladd function quadsolve(a::BigFloat, b::BigFloat, c::BigFloat)::NTuple{2, BigFloat}
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
        ((-b+d1)*a2, (-b-d1)*a2)
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
    @inline function quadsolve(a::A, b::B, c::C) where {A <: Number, B <: Number, C <: Number}
        T = promote_type(A, B, C)
        if T <: FComplex
            quadsolve(T(a), T(b), T(c))
        elseif T <: Union{BigInt, Rational{BigInt}}
            let T = BigFloat
                quadsolve(T(a), T(b), T(c))
            end
        elseif T <: Union{Complex{BigInt}, Complex{Rational{BigInt}}}
            let T = Complex{BigFloat}
                quadsolve(T(a), T(b), T(c))
            end
        elseif T <: Real
            let T = Float64
                quadsolve(T(a), T(b), T(c))
            end
        else
            let T = Complex{Float64}
                quadsolve(T(a), T(b), T(c))
            end
        end
    end
end

function _solve!(root1s::AbstractVector{T}, root2s::AbstractVector{T}, av::AbstractVector, bv::AbstractVector, cv::AbstractVector) where T
    for (i, j, a, b, c) = zip(eachindex(root1s), eachindex(root2s), av, bv, cv)
        root1s[i], root2s[i] = quadsolve(a, b, c)
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
    dlm = r"\t+"
    odlm = "\t"
    header = false
    skipno = 0
    outfile = stdout
    infile = stdin
    time = false
    T = Float64
    pad = 0
    i = 1
    while i ≤ length(args)
        arg = args[i]
        if arg == "-h"
            header = true
        elseif arg == "-c"
            T = Complex{Float64}
        elseif arg == "-n"
            i += 1
            arg = args[i]
            odlm = arg
        elseif arg == "-s"
            i += 1
            arg = args[i]
            skipno = parse(Int, arg)
        elseif arg == "-p"
            i += 1
            arg = args[i]
            pad = parse(Int, arg)
        elseif arg == "-d"
            i += 1
            arg = args[i]
            odlm = arg
            dlm = Regex(arg)
        elseif arg == "-i"
            i += 1
            arg = args[i]
            infile = arg
        elseif arg == "-o"
            i += 1
            arg = args[i]
            outfile = arg
        elseif arg == "-t"
            time = true
        end
        i += 1
    end
    if time
        @timev process(outfile, infile; skip=skipno, header=header, dlm=dlm, odlm=odlm, T=T, pad=pad)
    else
        process(outfile, infile; skip=skipno, header=header, dlm=dlm, odlm=odlm, T=T, pad=pad)
    end
end

function process(outfile::Union{IO, AbstractString}, infile::AbstractString; nargs...)
    open(infile, "r") do f
        process(outfile, f; nargs...)
    end
end
function process(outfile::AbstractString, infile::IO; nargs...)
    open(outfile, "w") do f
        process(f, infile; nargs...)
    end
end
function process(outfile::IO, infile::IO; skip=1, header=false, dlm=r"\t+", odlm="\t", T=Float64, interactive=true, pad=0)
    if header
        join(outfile, lpad.(("a", "b", "c", "r1", "r2"), pad), odlm)
        print(outfile, "\n")
    end
    for (i, l) = enumerate(eachline(infile))
        if interactive && l == "quit"
            break
        end
        if i <= skip
            continue
        end
        try
            (a, b, c, _...) = parse.(T, split(l, dlm))
            r1, r2 = quadsolve(a, b, c)
            join(outfile, lpad.((a, b, c, r1, r2), pad), odlm)
            print(outfile, "\n")
        catch e
            join(stderr, (i, e), "\t")
            print(stderr, "\n")
            continue
        end
    end
end

precompile(main, (Vector{String},))

end # module QuadSolve
