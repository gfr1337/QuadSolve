module QuadSolve
using SIMD
using MuladdMacro
using DataFrames
using CSV
export quadform

function linsolve(a::T, b::T) where T
    -b/a
end


begin
    local d, _quadform, FComplex
    FComplex = Union{AbstractFloat, Complex{<:AbstractFloat}}
    @inline signs()::Vec{2, Float64} = signs(Float64)
    @inline function signs(v::Union{T, Type{T}})::Vec{2, T} where {T}
        Vec{2, T}((one(v), -one(v)))
    end
    @inline @muladd function d(a::T, b::T, c::T)::T where T 
        b^2 - 4*a*c
    end
    @inline @muladd function quadform(a::Complex{T}, b::Complex{T}, c::Complex{T})::NTuple{2, Complex{T}} where {T <: AbstractFloat}
        if iszero(a)
            x = linsolve(b, c)
            return (x, x)
        end
        d1 = sqrt(d(a, b, c))
        a2 = inv(2a)
        ((-b+d1)*a2, (-b-d1)*a2)
    end
    @inline @muladd function quadform(a::T, b::T, c::T)::NTuple{2, T} where {T <: AbstractFloat}
        if iszero(a)
            x =linsolve(b, c)
            return (x, x)
        end
        d1 = d(a, b, c)
        if d1 < 0
            return (T(NaN), T(NaN))
        end
        d1 = sqrt(d1)
        a2 = inv(2a)
        rv = (signs(T)*d1-b)*a2
        (rv[1], rv[2])
    end
    @inline function quadform(a::FComplex, b::FComplex, c::FComplex)
        quadform(promote(a, b, c)...)
    end
    @inline function quadform(a::Number, b::Number, c::Number)
        quadform(promote(0.0+a, 0.0+b, 0.0+c)...) 
    end
end

function quadform!(root1s::Vector{T}, root2s::Vector{T}, a::AbstractVector, b::AbstractVector, c::AbstractVector) where T
    for i = eachindex(root1s)
        root1s[i], root2s[i] = quadform(a[i], b[i], c[i])
    end
end

randf(n, x=n/2) = DataFrame(a=randn(n)*x, b=randn(n)*x, c=randn(n)*x)

main(fname::String) = CSV.write(stdout, solve!(CSV.read(fname, DataFrame)))
solve(df::DataFrame; T=Float64) = solve!(copy(df), T=T)
function solve!(df::DataFrame; T=Float64)
    s = size(df, 1)
    root1s, root2s = Vector{T}(undef, s), Vector{T}(undef, s)
    quadform!(root1s, root2s, df.a, df.b, df.c)
    insertcols!(df, :root1 => root1s, :root2 => root2s, copycols=false)
    df
end

end # module QuadSolve
