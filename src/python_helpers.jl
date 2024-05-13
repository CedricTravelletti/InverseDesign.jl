# Helper functions for Jupyter notebooks.
using DFTK
using ForwardDiff
using ComponentArrays
using InverseDesign


system = construct_silicon()

# Create a simple calculator for the model.
model_kwargs = (; functionals = [:lda_x, :lda_c_pw], temperature = 1e-4)
basis_kwargs = (; kgrid = [4, 4, 4], Ecut = 30.0)
scf_kwargs = (; tol = 1e-7)
calculator = DFTKCalculator(; model_kwargs, basis_kwargs, scf_kwargs, verbose=true)

# Compute energy at equilibrium position.
# Note that we have to make x0 mutable.
x0 = Vector(flatten(DFTK.parse_system(system).positions))

f(x, p) = energy_wrt_pos(calculator, system, x)
function g!(G, x, p)
	G = ForwardDiff.gradient(x -> f(x, p), x0)
end

""" Helper function for gamma point bandgap versus the first two components 
of the strain. Everything done around equilibrium positions. 

"""
function _f_strain12(strain12)
	strain = [strain12; [0., 0, 0, 0]]
	positions_flat = ComponentVector(; atoms=x0, strain)
	gamma_point_bandgap(calculator, system, positions_flat)
end

function gamma_point_bandgap_vs_strain12(strain12)
	y = _f_strain12(strain12)
	y_grad = ForwardDiff.gradient(_f_strain12, strain12)
	[[y]; y_grad]
end
