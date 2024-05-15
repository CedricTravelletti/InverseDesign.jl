""" Evaluate the Gamma point bandgap on a dense grid. 
This is used for later verifications.

"""
using StaticArrays
using JLD2
using InverseDesign
using DFTK


system = construct_diamond()

model_kwargs = (; functionals = [:lda_x, :lda_c_pw], temperature = 1e-4)
basis_kwargs = (; kgrid = [5, 5, 5], Ecut = 40.0)
scf_kwargs = (; tol = 1e-7)
calculator = DFTKCalculator(; model_kwargs, basis_kwargs, scf_kwargs, verbose=true)

# Compute energy at equilibrium position.
# Note that we have to make x0 mutable.
x0 = Vector(flatten(DFTK.parse_system(system).positions))

# Grid of strains along x-y.
strain_grid = -0.1 : 0.01 : 0.1
iterator = Iterators.product(strain_grid, strain_grid)
bandgaps = zeros(MVector{length(iterator)})
bandgaps_grad = zeros(MMatrix{length(iterator), 2})

for (i, x) in enumerate(iterator)
	res = InverseDesign.gamma_point_bandgap_vs_strain12(collect(x))
	bandgaps[i] = res[1]
	bandgaps_grad[i, :] = res[2:]
	if i % 10 == 0
	    save_object("./data/bandgaps.jld2", bandgaps)
	    save_object("./data/bandgaps_grad.jld2", bandgaps_grad)
    end
end
save_object("./data/bandgaps.jld2", bandgaps)
save_object("./data/bandgaps_grad.jld2", bandgaps_grad)
