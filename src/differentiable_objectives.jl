# 
# Set of target material properties with a differentiable implementation 
# for optimization.
#
# This is currently DFTK-specific. 
# The philosophy is to compute the dual of the scf wrt. atomic positions 
# or strain updates.
#
using AtomsBase
using AtomsCalculators
using DFTK
using GeometryOptimization
using Unitful
using UnitfulAtomic
using ComponentArrays

function flatten(vector)
    vcat(vector...)
end

function unflatten(vector)
    collect.(eachcol(reshape(vector, 3, :)))
end

"""
Computes dual of the SCF by updating system positions in a differentiable way.

Arguments:
- `positions` flattened vector of atomic positions in fractional coordinates.

"""
function compute_scf_dual(calculator::DFTKCalculator,
		               system::AbstractSystem, positions)
    positions = unflatten(positions)
    # Original model. Symmetries disabled for differentiability.
    model = model_DFT(system; symmetries=false, calculator.params.model_kwargs...)
    new_model = Model(model; positions)
    new_basis = PlaneWaveBasis(new_model; calculator.params.basis_kwargs...)
    self_consistent_field(new_basis; calculator.params.scf_kwargs...)
end

"""
Computes dual of the SCF by updating system positions in a differentiable way.

Arguments:
- `positions` ComponenVector with components `atoms` containing the (flattened) 
              vector of atomic positions in fractional coordinates and component 
	      `strain` containing the strain to apply (in voigt notation).

"""
function compute_scf_dual(calculator::DFTKCalculator,
		               system::AbstractSystem, positions::ComponentVector)
    deformation_tensor = I + voigt_to_full(positions.strain)
    positions = [collect((x for x in deformation_tensor * position))
		 for position in unflatten(positions.atoms)]
    lattice = matrix_to_bbox(deformation_tensor * bbox_to_matrix(bounding_box(system)))
    # Original model. Symmetries disabled for differentiability.
    model = model_DFT(system; symmetries=false, calculator.params.model_kwargs...)
    new_model = Model(model; positions, lattice)
    new_basis = PlaneWaveBasis(new_model; calculator.params.basis_kwargs...)
    self_consistent_field(new_basis; calculator.params.scf_kwargs...)
end
	
"""
Compute system total energy as a function of atomic positions.

Arguments:
- `positions_flat` ComponenVector with components `atoms` containing the (flattened) 
                   vector of atomic positions in fractional coordinates and component 
		   `strain` containing the strain to apply (in voigt notation).

"""
function energy_wrt_pos(calculator::DFTKCalculator, system::AbstractSystem, positions_flat)
    scfres_dual = compute_scf_dual(calculator, system, positions_flat)
    scfres_dual.energies.total
end

"""
Returns the index of the highest occupied band. 

"""
function valence_band_index(scfres)
    sum(scfres.occupation[1] .> scfres.occupation_threshold)
end

"""
Compute system direct bandgap as a function of atomic positions.

Arguments:
- `positions_flat` (flattened) vector of atomic positions in fractional coordinates.

"""
function direct_bandgap(calculator::DFTKCalculator, system::AbstractSystem, positions_flat)
    scfres_dual = compute_scf_dual(calculator, system, positions_flat)
    vi = valence_band_index(scfres_dual)
    minimum([εk[vi + 1] - εk[vi] for εk in scfres_dual.eigenvalues])
end

"""
Compute system bandgap at the Gamma point as a function of atomic positions.

Arguments:
- `positions_flat` (flattened) vector of atomic positions in fractional coordinates.

"""
function gamma_point_bandgap(calculator::DFTKCalculator, system::AbstractSystem, positions_flat)
    scfres_dual = compute_scf_dual(calculator, system, positions_flat)
    vi = valence_band_index(scfres_dual)
    scfres_dual.eigenvalues[1][vi + 1] - scfres_dual.eigenvalues[1][vi]
end
function gamma_point_bandgap(calculator::DFTKCalculator, system::AbstractSystem, positions_flat::ComponentVector)
    scfres_dual = compute_scf_dual(calculator, system, positions_flat)
    vi = valence_band_index(scfres_dual)
    scfres_dual.eigenvalues[1][vi + 1] - scfres_dual.eigenvalues[1][vi]
end
