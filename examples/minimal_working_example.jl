# Try energy derivative upon atomic positions updating.

using LinearAlgebra
using Unitful
using UnitfulAtomic
using DFTK
using GeometryOptimization
using ForwardDiff


# Basic silicon system.
a = 5.431u"angstrom"
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]];
Si = ElementPsp(:Si; psp=load_psp("hgh/lda/Si-q4"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]
system = periodic_system(lattice, atoms, positions)

# Create a simple calculator for the model.
model_kwargs = (; functionals = [:lda_x, :lda_c_pw], temperature = 1e-4)
basis_kwargs = (; kgrid = [4, 4, 4], Ecut = 30.0)
scf_kwargs = (; tol = 1e-5)
calculator = DFTKCalculator(; model_kwargs, basis_kwargs, scf_kwargs, verbose=true)


# Need to parse in order to use the model_DFT constructor.
parsed = DFTK.parse_system(system)

"""
Compute system total energy as a function of atomic positions.

Arguments:
- `positions_flat` (flat) vector of atomic positions (6-dimensional).

"""
function energy_wrt_pos(positions_flat)
	positions = collect.(eachcol(reshape(positions_flat, 3, :)))
	model = model_DFT(parsed.lattice, parsed.atoms, positions;
			  symmetries=false, model_kwargs...)
	basis = PlaneWaveBasis(model; basis_kwargs...)
	scfres = self_consistent_field(basis; scf_kwargs...)
	scfres.energies.total
end

# Compute energy at equilibrium position.
x0 = vcat(parsed.positions...)
energy_wrt_pos(x0)

# Try derivarive.
dfx0 = ForwardDiff.gradient(energy_wrt_pos, x0)
