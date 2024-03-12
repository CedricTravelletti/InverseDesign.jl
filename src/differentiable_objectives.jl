# 
# Set of target material properties with a differentiable implementation 
# for optimization.
#
using AtomsBase
using AtomsCalculators
using DFTK

"""
Updates system positions in a differentiable way.

Arguments:
- `positions_flat` (flat) vector of atomic positions (6-dimensional).

"""
function differentiable_update(calculator::DFTKCalculator,
		                                   system::AbstractSystem, positions_flat)
    positions = collect.(eachcol(reshape(positions_flat, 3, :)))
    # Original model. Symmetries disabled for differentiability.
    model = model_DFT(system; symmetries=false, calculator.params.model_kwargs...)
    new_model = Model(model; positions)
    new_basis = PlaneWaveBasis(new_model; calculator.basis_kwargs...)
    self_consistent_field(new_basis; calculator.params.scf_kwargs...)
	
"""
Compute system total energy as a function of atomic positions.

Arguments:
- `positions_flat` (flat) vector of atomic positions (6-dimensional).

"""
function energy_wrt_pos(calculator::DFTKCalculator, system::AbstractSystem, positions_flat)
    scfres_dual = differentiable_update(calculator, system, positions_flat)
    scfres_dual.energies.total
end

"""
Compute system direct  bandgap as a function of atomic positions.

Arguments:
- `positions_flat` (flat) vector of atomic positions (6-dimensional).

"""
function direct_bandgap_wrt_pos(calculator::DFTKCalculator, system::AbstractSystem, positions_flat)
    scfres_dual = differentiable_update(calculator, system, positions_flat)
    compute_band_gaps(scfres_dual)[:direct_bandgap]
end
