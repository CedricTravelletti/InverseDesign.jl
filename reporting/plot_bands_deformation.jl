### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 6467b10b-5f66-4699-badd-184660d875a3
begin
	using Pkg
	Pkg.activate(Base.current_project())
end

# ╔═╡ 2e85c349-c626-4fd6-8210-5dd10f8bbf73
begin
	using DFTK
	using InverseDesign
	using Plots 
	using JLD2
	using Unitful
	using UnitfulAtomic
	using LaTeXStrings
	using PlutoUI
end

# ╔═╡ d12f389a-f325-11ee-14fb-699e8ba8988e
md"""
# Plot Gamma point bandgap evolution wrt. first coordinate of firs atom
"""

# ╔═╡ 91284845-73eb-4fda-8f37-617dcb7b913f
begin
	Nx = 25;
	x_offsets = range(-0.1, 0.1; length=Nx);
end

# ╔═╡ 2b81d718-cc53-446d-ab25-c8e4004e8716
begin
	gammaP_bandgap_diamond_grid = load_object("./data/gammaP_bandgap_diamond_grid.jld2")
	gammaP_bandgap_diamond_grid_pert = load_object("./data/gammaP_bandgap_diamond_grid_pert.jld2")
	plot(x_offsets, mark=:o, gammaP_bandgap_diamond_grid; xlabel="x-offset (first atom)", 
	title=L"$\Gamma$-point bandgap (diamond)", label="equilibrium")
	plot!(x_offsets, mark=:o, gammaP_bandgap_diamond_grid_pert; label="perturbed")
end

# ╔═╡ 55cef20a-5952-4495-b250-c603ac8bffdd
md"""
## On to Derivatives 
"""

# ╔═╡ 54f4d5df-6a9a-44dc-a6b3-5072eaadd3d6
# ╠═╡ disabled = true
#=╠═╡
dfx0_diamon_pert = ForwardDiff.gradient(f_diamond, x0_diamond_pert)
  ╠═╡ =#

# ╔═╡ aa3fa70c-19a2-4986-9cdc-2e763ec19ba4
# ╠═╡ disabled = true
#=╠═╡
dfx0_diamon_pert_finite = FiniteDiff.finite_difference_gradient(f_diamond, x0_diamond_pert)
  ╠═╡ =#

# ╔═╡ 1aac06f3-8b93-4cdb-b388-4ae9a8246628
# ╠═╡ disabled = true
#=╠═╡
dfx0_diamond_pert_finite_central = FiniteDifferences.grad(
	central_fdm(7, 1), f_diamond, x0_diamond_pert)
  ╠═╡ =#

# ╔═╡ b60602e2-2234-4ff6-9b8a-d5cad496c2ac
# ╠═╡ disabled = true
#=╠═╡
dfx0_diamon_pert_finite_central_noisy = FiniteDifferences.grad(central_fdm(7, 1, factor=1e6), f_diamond, x0_diamond_pert)
  ╠═╡ =#

# ╔═╡ c0faa3df-aa45-4f46-b531-3bf1b753bd59
# ╠═╡ disabled = true
#=╠═╡
dfx0_diamon_pert = ForwardDiff.gradient(f_diamond, x0_diamond_pert)
  ╠═╡ =#

# ╔═╡ 2290ba64-46aa-47f8-8231-b944dcb8054d
begin
	bands_loaded = []
	for i in 1:Nx
	push!(bands_loaded, load_bands_plotting("./data/diamond_grid_$(i)"))
	end
end

# ╔═╡ 1e6ae277-5ebc-4314-b15b-286b17298990
@bind i_plot Slider(1:1:Nx, default=1, show_value=true)

# ╔═╡ 4e9ff512-841f-41df-8dd2-163ca8d8508a
let			 	
	plot_bandstructure(bands_loaded[i_plot])
end

# ╔═╡ Cell order:
# ╟─d12f389a-f325-11ee-14fb-699e8ba8988e
# ╠═6467b10b-5f66-4699-badd-184660d875a3
# ╠═2e85c349-c626-4fd6-8210-5dd10f8bbf73
# ╠═91284845-73eb-4fda-8f37-617dcb7b913f
# ╠═2b81d718-cc53-446d-ab25-c8e4004e8716
# ╟─55cef20a-5952-4495-b250-c603ac8bffdd
# ╠═54f4d5df-6a9a-44dc-a6b3-5072eaadd3d6
# ╠═aa3fa70c-19a2-4986-9cdc-2e763ec19ba4
# ╠═1aac06f3-8b93-4cdb-b388-4ae9a8246628
# ╠═b60602e2-2234-4ff6-9b8a-d5cad496c2ac
# ╠═c0faa3df-aa45-4f46-b531-3bf1b753bd59
# ╠═2290ba64-46aa-47f8-8231-b944dcb8054d
# ╠═4e9ff512-841f-41df-8dd2-163ca8d8508a
# ╠═1e6ae277-5ebc-4314-b15b-286b17298990
