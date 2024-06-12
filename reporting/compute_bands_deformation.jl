""" Compute and plot how bands, and in particularly the bandgap 
at the Gamma point deform when one atom gets moved. 

Here we only move the first coordinate of the first atom.


"""

using DFTK
setup_threading()

using InverseDesign
using JLD2
using Unitful
using UnitfulAtomic
using Random, Distributions

diamond = construct_diamond()

# Calculatro with loose tolerance.
model_kwargs = (; functionals = [:lda_x, :lda_c_pw], temperature = 1e-4)
basis_kwargs = (; kgrid = [5, 5, 5], Ecut=30.0)
scf_kwargs = (; tol = 1e-6)
calculator = DFTKCalculator(; model_kwargs, basis_kwargs, scf_kwargs, verbose=true)

f_diamond(x) = gamma_point_bandgap(calculator, diamond, x)

# Equilibrium positions.
x0_diamond = Vector(flatten(DFTK.parse_system(diamond).positions))

# Modify first coordinate of first atom.
x_offsets = range(-0.1, 0.1; length=50)

gammaP_bandgap_diamond_grid = Vector{Float64}()
x0_grid = [x0_diamond .+ [offset; zeros(length(x0_diamond) - 1)] for offset in x_offsets]

# Use same kpath for all deformations.
scfres = InverseDesign.compute_scf_dual(calculator, diamond, x0_diamond)
kpath = irrfbz_path(scfres.basis.model)
for (i, x) in enumerate(x0_grid)
    scfres = InverseDesign.compute_scf_dual(calculator, diamond, x)
    vi = InverseDesign.valence_band_index(scfres)
    gammaP_bandgap = scfres.eigenvalues[1][vi + 1] - scfres.eigenvalues[1][vi]
    push!(gammaP_bandgap_diamond_grid, gammaP_bandgap)
    save_bands_plotting("./data/diamond_grid_$(i)", scfres; kpath)
end
save_object("./data/gammaP_bandgap_diamond_grid.jld2", gammaP_bandgap_diamond_grid)

# Bandgap around perturbed equilibrium.
Random.seed!(1) # Setting the seed
x0_diamond_pert = x0_diamond + rand(Normal(0., 0.01), size(x0_diamond)) 

gammaP_bandgap_diamond_grid_pert = Vector{Float64}()
x0_grid_diamond_pert = [x0_diamond_pert .+ [offset; zeros(length(x0_diamond_pert) - 1)] for offset in x_offsets]

for (i, x) in enumerate(x0_grid_diamond_pert)
    scfres = InverseDesign.compute_scf_dual(calculator, diamond, x)
    vi = InverseDesign.valence_band_index(scfres)
    gammaP_bandgap = scfres.eigenvalues[1][vi + 1] - scfres.eigenvalues[1][vi]
    push!(gammaP_bandgap_diamond_grid_pert, gammaP_bandgap)
		save_bands_plotting("./data/diamond_grid_pert_$(i)", scfres)
end
save_object("./data/gammaP_bandgap_diamond_grid_pert.jld2",
			    gammaP_bandgap_diamond_grid_pert)
