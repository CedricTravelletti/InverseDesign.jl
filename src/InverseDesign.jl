module InverseDesign

using AtomsBase
using AtomsCalculators
using DFTK

export flatten, unflatten, energy_wrt_pos, direct_bandgap, gamma_point_bandgap
include("differentiable_objectives.jl")

export load_bands_plotting, save_bands_plotting
include("utils.jl")

export construct_silicon, construct_diamond, construct_gammaP_bandgap_vs_strain12, construct_gammaP_bandgap_vs_strain14
include("example_systems.jl")

end
