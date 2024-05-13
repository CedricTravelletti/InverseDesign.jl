#= Test Geometry Optimization on an aluminium supercell.
=#
@testitem "Test energy derivative wrt position." setup=[TestCases] begin
    using InverseDesign
    using DFTK
    using ForwardDiff
    using FiniteDiff
    using LinearAlgebra
    
    system = construct_silicon()
    calculator = TestCases.calculator
    
    # Compute energy at equilibrium position.
    # Note that we have to make x0 mutable.
    x0 = Vector(flatten(DFTK.parse_system(system).positions))
    
    dfx0 = ForwardDiff.gradient(x -> energy_wrt_pos(calculator, system, x), x0)
    dfx0_finite = FiniteDiff.finite_difference_gradient(x -> energy_wrt_pos(calculator, system, x), x0)
    
    @test isapprox(norm(dfx0 - dfx0_finite), 0; atol=1e-3)
end

# @testitem "Test direct band gap derivative wrt position." begin
#     using InverseDesign
#     using DFTK
#     using ForwardDiff
#     using FiniteDiff
#     using LinearAlgebra
#     using Random, Distributions
#     
#     system = construct_diamond()
#     
#     # Create a simple calculator for the model.
#     model_kwargs = (; functionals = [:lda_x, :lda_c_pw], temperature = 1e-4)
#     basis_kwargs = (; kgrid = [4, 4, 4], Ecut = 30.0)
#     scf_kwargs = (; tol = 1e-5)
#     calculator = DFTKCalculator(; model_kwargs, basis_kwargs, scf_kwargs, verbose=true)
#     
#     # Compute at perturbed position (otherwise non-differentiable).
#     x0 = Vector(flatten(DFTK.parse_system(system).positions))
#     Random.seed!(1)
#     x0_pert = x0 + rand(Normal(0., 0.01), size(x0)) 
#     
#     dfx0 = ForwardDiff.gradient(x -> direct_bandgap(calculator, system, x), x0)
#     dfx0_finite = FiniteDiff.finite_difference_gradient(x -> direct_bandgap(calculator, system, x), x0)
#     
#     @test isapprox(norm(dfx0 - dfx0_finite), 0; atol=1e-3)
# end

@testitem "Test Gamma point band gap derivative wrt position. (ForwardDiff vs FiniteDiff)" setup=[TestCases] begin
    using InverseDesign
    using DFTK
    using ForwardDiff
    using FiniteDiff
    using LinearAlgebra
    using Random, Distributions
    
    system = construct_diamond()
    calculator = TestCases.calculator
    
    # Compute at perturbed position (otherwise non-differentiable).
    x0 = Vector(flatten(DFTK.parse_system(system).positions))
    Random.seed!(1)
    x0_pert = x0 + rand(Normal(0., 0.01), size(x0)) 
    
    dfx0_forward = ForwardDiff.gradient(x -> gamma_point_bandgap(calculator, system, x), x0_pert)
    dfx0_finite = FiniteDiff.finite_difference_gradient(
                                x -> gamma_point_bandgap(calculator, system, x), x0_pert)
    
    @test isapprox(dfx0_forward, dfx0_finite; rtol=1e-4)
end

@testitem "Test Gamma point band gap derivative wrt position. (ForwardDiff vs FiniteDifferences)" setup=[TestCases] begin
    using InverseDesign
    using DFTK
    using ForwardDiff
    using FiniteDifferences
    using LinearAlgebra
    using Random, Distributions
    
    system = construct_diamond()
    calculator = TestCases.calculator
    
    # Compute at perturbed position (otherwise non-differentiable).
    x0 = Vector(flatten(DFTK.parse_system(system).positions))
    Random.seed!(1)
    x0_pert = x0 + rand(Normal(0., 0.01), size(x0)) 
    
    dfx0_forward = ForwardDiff.gradient(x -> gamma_point_bandgap(calculator, system, x), x0_pert)
    dfx0_finite = FiniteDifferences.grad(central_fdm(7, 1, factor=1e6),
                                x -> gamma_point_bandgap(calculator, system, x), x0_pert)
    
    @test isapprox(dfx0_forward, dfx0_finite; rtol=1e-4)
end
