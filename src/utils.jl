using DFTK
using JLD2
using MPI


""" Load DFTK bands for plotting. Meant to be used together 
with `save_bands_plotting`.

"""
function load_bands_plotting(base_path)
    scfres = load_scfres(base_path * "_scfres.jld2")
    bands = jldopen(base_path * "_bands.jld2")
    kinter = load_object(base_path * "_kinter.jld2")

    basis_args = (scfres.basis.model,
                  bands["Ecut"],
                  bands["fft_size"],
                  bands["variational"],
                  bands["kgrid"],
                  bands["symmetries_respect_rgrid"],
                  bands["use_symmetries_for_kpoint_reduction"])
    basis = PlaneWaveBasis(basis_args...,
			   MPI.COMM_WORLD, DFTK.CPU())
    (; occupation=bands["occupation"], εF=bands["εF"],
       basis, kinter,
       eigenvalues=collect.(eachcol(dropdims(bands["eigenvalues"], dims=3))))
end

""" Load DFTK bands for plotting. Meant to be used together 
with `save_bands_plotting`.

"""
function save_bands_plotting(base_path::String, scfres; kpath=nothing, kline_density=20)
    if isnothing(kpath)
	    kpath = irrfbz_path(scfres.basis.model)
    end
    bands = compute_bands(scfres, kpath; kline_density)

    save_bands(base_path * "_bands.jld2", bands)
    save_scfres(base_path * "_scfres.jld2", scfres;
                save_ψ=false, save_ρ=false)
    save_object(base_path * "_kinter.jld2", bands.kinter)
end
