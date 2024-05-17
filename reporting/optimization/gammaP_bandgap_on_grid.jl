""" Evaluate the Gamma point bandgap on a dense grid. 
This is used for later verifications.

"""

using StaticArrays
using JLD2
using InverseDesign
using DFTK


gammaP_bandgap_vs_strain12 = construct_gammaP_bandgap_vs_strain12()

# Grid of strains along x-y.
strain_grid = -0.1 : 0.01 : 0.1
iterator = Iterators.product(strain_grid, strain_grid)
bandgaps = zeros(MVector{length(iterator)})
bandgaps_grad = zeros(MMatrix{length(iterator), 2})

for (i, x) in enumerate(iterator)
	res = gammaP_bandgap_vs_strain12(collect(x))
	bandgaps[i] = res[1]
	bandgaps_grad[i, :] = res[2:end]
	if i % 10 == 0
	    save_object("./data/bandgaps.jld2", bandgaps)
	    save_object("./data/bandgaps_grad.jld2", bandgaps_grad)
    end
end
save_object("./data/bandgaps.jld2", bandgaps)
save_object("./data/bandgaps_grad.jld2", bandgaps_grad)
