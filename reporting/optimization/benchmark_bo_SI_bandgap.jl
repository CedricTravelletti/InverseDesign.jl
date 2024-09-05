""" Benchmark gradient BO vs vanilla BO on gammaP bandgap of Silicon.

"""

using StaticArrays
using JLD2
using InverseDesign
using DFTK

using BayesianOptimization, GaussianProcesses, Distributions
using AbstractGPs
using GradientGPs
using ForwardDiff
using CovarianceFunctions
using Plots


gammaP_bandgap_vs_strain14 = construct_gammaP_bandgap_vs_strain14()

# -------------------
# GP model definition
# -------------------
sigma2 = 3.0^2
g = sigma2 * CovarianceFunctions.Lengthscale(CovarianceFunctions.MaternP(2), 4.0); # Matérn kernel with ν = 2.5

gv = CovarianceFunctions.ValueGradientKernel(g);
prior_m(x) = ValGrad(0.0, [0.0])
kernelD = GradientKernel(gv, 2)
mean_f = ZeroGradientMean{Float64}(2)
gpD = GradientGP(mean_f, kernelD)

model = BOGradientGP(gpD)

# Do not optimize hyperparams.
modeloptimizer = NoModelOptimizer()

myopt = GradientBOpt(gammaP_bandgap_vs_strain14,
           ValGrad{Float64},
           model,
           UpperConfidenceBound(),                   # type of acquisition
           modeloptimizer,                        
           [-0.05, -0.05], [0.05, 0.05],                     # lowerbounds, upperbounds         
           repetitions = 1,                          # evaluate the function for each input 5 times
           maxiterations = 20,                      # evaluate at 100 input positions
           sense = Min,                              # minimize the function
           acquisitionoptions = (method = :LN_NELDERMEAD, # run optimization of acquisition function with NLopts :LD_LBFGS method
                                 restarts = 5,       # run the NLopt method from 5 random initial conditions each time.
                                 maxtime = 300.0,      # run the NLopt method for at most 0.1 second each time
                                 maxeval = 10000),    # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl)
            verbosity = Progress,
            initializer_iterations=1)

result = boptimize!(myopt)

# Grid of strains along x-y.
strain_grid = -0.1 : 0.01 : 0.1
iterator = Iterators.product(strain_grid, strain_grid)
bandgaps = zeros(MVector{length(iterator)})
bandgaps_grad = zeros(MMatrix{length(iterator), 2})

for (i, x) in enumerate(iterator)
	res = gammaP_bandgap_vs_strain14(collect(x))
	bandgaps[i] = res[1]
	bandgaps_grad[i, :] = res[2:end]
	if i % 10 == 0
	    save_object("./data/bandgaps.jld2", bandgaps)
	    save_object("./data/bandgaps_grad.jld2", bandgaps_grad)
    end
end
save_object("./data/bandgaps.jld2", bandgaps)
save_object("./data/bandgaps_grad.jld2", bandgaps_grad)
