import numpy as np
from ase.build import bulk
import pickle
with open('atoms.pkl', 'rb') as f:
    atoms = pickle.load(f)

input = {}
with open('Input.txt', 'r') as file:
    for line in file:
        # Ignore lines starting with '#'
        if line.startswith('#'):
            continue
        key, value = line.strip().split(' : ')
        try:
            # Convert to integer if possible
            input[key] = int(value)
        except ValueError:
            # If not possible, store as string
            input[key] = value

# ## Set up evaluation function (pipe to ASE) for trial parameters suggested by Ax. 
# Note that this function can return additional keys that can be used in the `outcome_constraints` of the experiment.
def evaluate(parameters):
    x = np.array([parameters.get(f"x"), parameters.get(f"y"), parameters.get(f"z")])
    # Can put zeros since constraints are respected by set_positions.
    new_pos = np.vstack([np.zeros((atoms.get_number_of_atoms() - 1, 3)), x])
    atoms.set_positions(new_pos, apply_constraint=True)
    energy = atoms.get_potential_energy()
    
    # In our case, standard error is 0, since we are computing a synthetic function.
    return {"adsorption_energy": (energy, 0.0)} # We have 0 noise on the target.

from botorch.models import SingleTaskGP, ModelListGP, FixedNoiseGP
# Ax wrappers for BoTorch components
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition import knowledge_gradient
from botorch.acquisition import predictive_entropy_search
from botorch.acquisition import max_value_entropy_search
# model = SingleTaskGP(init_x, init_y)
# mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)

# Fit on init data. 
# from botorch import fit_gpytorch_model
# fit_gpytorch_model(mll)

import sys
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

model = BoTorchModel(
    # Optional `Surrogate` specification to use instead of default
    surrogate=Surrogate(
        # BoTorch `Model` type
        botorch_model_class=str_to_class(input['bo_surrogate']),
        # Optional, MLL class with which to optimize model parameters
        mll_class=ExactMarginalLogLikelihood,
        # Optional, dictionary of keyword arguments to underlying
        # BoTorch `Model` constructor
        model_options={},
    ),
    # Optional BoTorch `AcquisitionFunction` to use instead of default
    botorch_acqf_class=str_to_class(input['bo_acquisition_f']),
    # Optional dict of keyword arguments, passed to the input
    # constructor for the given BoTorch `AcquisitionFunction`
    acquisition_options={},
)

# ## Create client and initial sampling strategy to warm-up the GP model

import torch
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models


gs = GenerationStrategy(
    steps=[
        # 1. Initialization step (does not require pre-existing data and is well-suited for
        # initial sampling of the search space)
        GenerationStep(
            model=Models.SOBOL,
            num_trials=input['gs_init_steps'],  # How many trials should be produced from this generation step
            min_trials_observed=3,  # How many trials need to be completed to move to next model
            max_parallelism=5,  # Max parallelism for this step
            #model_kwargs={"seed": 999},  # Any kwargs you want passed into the model
            #model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
        ),
        # 2. Bayesian optimization step (requires data obtained from previous phase and learns
        # from all data available at the time of each new candidate generation call)
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,  # No limitation on how many trials should be produced from this step
            max_parallelism=3,  # Parallelism limit for this step, often lower than for Sobol
            # More on parallelism vs. required samples in BayesOpt:
            # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials
        ),
    ]
)