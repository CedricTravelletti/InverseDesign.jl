import numpy as np
from ase.build import bulk
import pickle
import torch

import gpytorch
from matplotlib import pyplot as plt

from botorch import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qKnowledgeGradient, qNoisyExpectedImprovement
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.models.gpytorch import GPyTorchModel
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.model import FantasizeMixin
from botorch.models.gpytorch import GPyTorchModel
from torch import Tensor
from typing import NoReturn, Optional
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.module import Module
from botorch.models.transforms.outcome import Log, OutcomeTransform
from botorch.models.transforms.input import InputTransform
from gpytorch.means.mean import Mean
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood

# Check if CUDA is available and set PyTorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup experiment parameters from input file
input = {}
#Read input file
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

with open('atoms.pkl', 'rb') as f:
    atoms = pickle.load(f)

# Define the dGP to include forces
class GPWithDerivatives(GPyTorchModel, gpytorch.models.ExactGP, FantasizeMixin):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        # Dimension of model
        dim = train_X.shape[-1] 
        # Multi-dimensional likelihood since we're modeling a function and its gradient
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1 + dim)
        super().__init__(train_X, train_Y, likelihood)
        # Gradient-enabled mean
        self.mean_module = gpytorch.means.ConstantMeanGrad() 
        # Gradient-enabled kernel
        self.base_kernel = gpytorch.kernels.RBFKernelGrad( 
            ard_num_dims=dim, # Separate lengthscale for each input dimension
        )
        # Adds lengthscale to the kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        # Output dimension is 1 (function value) + dim (number of partial derivatives)
        self._num_outputs = 1 + dim
        # Used to extract function value and not gradients during optimization
        self.scale_tensor = torch.tensor([1.0] + [0.0]*dim, dtype=torch.double)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# ## Set up evaluation function (pipe to ASE) for trial parameters suggested by Ax. 
# Note that this function can return additional keys that can be used in the `outcome_constraints` of the experiment.
def evaluate(parameters):
    x = torch.tensor([parameters.get(f"x"), parameters.get(f"y"), parameters.get(f"z")], device = device)
    x2 = torch.tensor([parameters.get(f"x2"), parameters.get(f"y2"), parameters.get(f"z2")], device = device)
    # Can put zeros since constraints are respected by set_positions.
    new_pos = torch.vstack([torch.zeros((atoms.get_number_of_atoms() - input['number_of_ads'], 3), device = device), x, x2])
    atoms.set_positions(new_pos.cpu().numpy(), apply_constraint=True)
    energy = atoms.get_potential_energy()
    dx,dy,dz = atoms.get_forces()[-2]
    dx2,dy2,dz2 = atoms.get_forces()[-1]
    # In our case, standard error is 0, since we are computing a synthetic function.
    return {"adsorption_energy": (energy, 0.0)} # We have 0 noise on the target.
    #return {"adsorption_energy": (energy, 0.0),"dx": (dx, 0.0), "dy": (dy, 0.0), "dz": (dz, 0.0), "dx2": (dx2, 0.0), "dy2": (dy2, 0.0), "dz2": (dz2, 0.0)} # We have 0 noise on the target.

#To add later ? "dx": (dx, 0.0), "dy": (dy, 0.0), "dz": (dz, 0.0), "dx2": (dx2, 0.0), "dy2": (dy2, 0.0), "dz2": (dz2, 0.0)

from botorch.models import SingleTaskGP, ModelListGP, FixedNoiseGP
# Ax wrappers for BoTorch components
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition import qMaxValueEntropy
from botorch.acquisition.analytic import UpperConfidenceBound

import sys
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

# ## Create client and initial sampling strategy to warm-up the GP model
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from botorch.models import MultiTaskGP

gs = GenerationStrategy(
    steps=[
        # 1. Initialization step (does not require pre-existing data and is well-suited for
        # initial sampling of the search space)
        GenerationStep(
            model=Models.SOBOL,
            num_trials=input['gs_init_steps'],  # How many trials should be produced from this generation step
            min_trials_observed=3,  # How many trials need to be completed to move to next model
            max_parallelism=5,  # Max parallelism for this step
            #model_kwargs={},  # Any kwargs you want passed into the model
            #model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
        ),
        # 2. Bayesian optimization step (requires data obtained from previous phase and learns
        # from all data available at the time of each new candidate generation call)
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,  # No limitation on how many trials should be produced from this step
            max_parallelism=3,
            model_kwargs={
                "torch_device": device,
                "surrogate": Surrogate(
                                        # BoTorch `Model` type
                                        botorch_model_class=SingleTaskGP,
                                        # Optional, MLL class with which to optimize model parameters
                                        mll_class=ExactMarginalLogLikelihood,
                                        # Optional, dictionary of keyword arguments to underlying
                                        # BoTorch `Model` constructor
                                        model_options={"torch_device": device},
                                        ),
                "botorch_acqf_class": str_to_class(input['bo_acquisition_f']),
                #"acquisition_options": {"torch_device": device},
                #"posterior_transform": ScalarizedPosteriorTransform(weights=torch.tensor([1.0] + [0.0]*input["number_of_ads"]*3, dtype=torch.double)),
                        },  # Any kwargs you want passed into the model
            model_gen_kwargs={"torch_device": device},
            # Parallelism limit for this step, often lower than for Sobol
            # More on parallelism vs. required samples in BayesOpt:
            # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials
        ),
    ]
)