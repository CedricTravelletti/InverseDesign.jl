# 
# Compute the bandgap vs. strain relation on a dense grid, for diamond.
# 
using InverseDesign

system = construct_diamond()

min_strain, max_strain, step_strain = -0.1, 0.1, 0.05
strain_range = min_strain:step_strain:max_strain

grid_shape = ntuple(Returns(size(strain_range)[1]), 6)
results = zeros(grid_shape)

grid_iterator = Iterators.product(ntuple(Returns(strain_range), 6))
