module InverseDesign

using AtomsBase
using AtomsCalculators
using DFTK

export flatten, unflatten, energy_wrt_pos, direct_bandgap, gamma_point_bandgap
include("differentiable_objectives.jl")

export construct_silicon, construct_diamond
include("example_systems.jl")

end
