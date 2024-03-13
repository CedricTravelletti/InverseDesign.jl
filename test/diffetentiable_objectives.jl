#= Test Geometry Optimization on an aluminium supercell.
=#
@testitem "Test energy derivative wrt position." begin
    using InverseDesign
    using DFTK
    using ForwardDiff
    using FiniteDiff
    
    system = construct_silicon()
    
    # Create a simple calculator for the model.
    model_kwargs = (; functionals = [:lda_x, :lda_c_pw], temperature = 1e-4)
    basis_kwargs = (; kgrid = [4, 4, 4], Ecut = 30.0)
    scf_kwargs = (; tol = 1e-5)
    calculator = DFTKCalculator(; model_kwargs, basis_kwargs, scf_kwargs, verbose=true)
    
    # Compute energy at equilibrium position.
    # Note that we have to make x0 mutable.
    x0 = Vector(flatten(DFTK.parse_system(system).positions))
    
    dfx0 = ForwardDiff.gradient(x -> energy_wrt_pos(calculator, system, x), x0)
    dfx0_finite = FiniteDiff.finite_difference_gradient(x -> energy_wrt_pos(calculator, system, x), x0)
    
    @test isapprox(norm(dfx0 - dfx0_finite), 0; atol=1e-3)
end
