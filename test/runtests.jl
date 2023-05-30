using Test
using QuadSolve

@testset verbose=true "quadform" begin
    for tt = NTuple{3, Int}[(4, 28, 49), (10, 20, -30)]
        @testset let q = quadsolve(tt...)
            @test typeof(q) == NTuple{2, Float64}
            @test !any(isnan.(q))
            @test all(@.(abs(tt[1]*q^2+tt[2]*q+tt[3]) < 1e-14))
        end
    end
    for tt = NTuple{3, Complex{Int}}[(4, 28, 49), (10, 20, 30)]
        @testset let q = quadsolve(tt...)
            @test typeof(q) == NTuple{2, Complex{Float64}}
            @test !any(isnan.(q))
            @test all(@.(abs(tt[1]*q^2+tt[2]*q+tt[3]) < 1e-14))
        end
    end
end
